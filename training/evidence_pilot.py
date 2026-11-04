"""Evidence-first partial-label pilot, with a versioned data and supervision gate."""

import json
from pathlib import Path

from annotation.cache import atomic_json
from scripts.common import sha256
from training.evidence_replay import replay
from training.model_io import make_batch
from training.partial import partial_labels, render_target
from training.pilot import PilotSession
from training.runtime import require_hf_token


class EvidencePilot(PilotSession):
    def __init__(self, root, output, settings, specs, source):
        super().__init__(root, output, settings, specs)
        self.source = Path(source)
        # A new replay cannot silently mix a changed source archive with earlier labels/masks.
        digest = {
            str(path.relative_to(self.source)): sha256(path.read_bytes())
            for path in sorted(self.source.rglob("*.json"))
            if "source_snapshot" not in path.parts
        }
        path = output / "evidence_source_identity.json"
        if path.exists() and json.loads(path.read_text()) != digest:
            raise ValueError("Replay source changed; choose a new run ID")
        atomic_json(path, digest)

    def preview(self):
        selections = self.output / "selector_choices.json"
        selected = json.loads(selections.read_text()) if selections.exists() else {}
        report = replay(
            self.source,
            self.inputs,
            self.specs,
            self.output,
            self.config.min_ocr_confidence,
            selected,
        )
        print(json.dumps(report, indent=2))
        return report

    def select(self):
        report = json.loads((self.output / "evidence_preview.json").read_text())
        if report["accepted_titles"] >= self.settings["minimum_accepted_titles"]:
            print("Offline replay meets title coverage; no selector model calls needed")
            return
        from training.selector import select_titles

        select_titles(self)
        self.preview()

    def annotate(self):
        report = self.preview()
        decisions = json.loads((self.output / "evidence_decisions.json").read_text())
        eligible = [row for row in decisions if row["partial_eligible"]]
        minimum = self.settings["minimum_accepted_titles"]
        ready = report["accepted_titles"] >= minimum and report["eligible_groups"] >= minimum
        coverage = {
            **report,
            "passed": ready,
            "minimum_titles": minimum,
            "minimum_eligible_groups": minimum,
            "partial_examples": len(eligible),
            "supervision_policy": "partial-fields-v1",
            "complete_records_required": False,
            "note": "Unknown fields must pass token-mask audit before training",
        }
        atomic_json(self.output / "annotation_coverage.json", coverage)
        atomic_json(self.output / "partial_examples.json", eligible)
        if not ready:
            raise RuntimeError("Evidence data gate failed; download reports, do not train")
        print(f"Partial data ready: {len(eligible)} examples; unknown fields will be masked")

    def audit_masks(self):
        from transformers import AutoProcessor

        self.require("annotate")
        spec = self.specs["student"]
        processor = AutoProcessor.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            token=require_hf_token(),
            trust_remote_code=False,
            min_pixels=256 * 28 * 28,
            max_pixels=768 * 768,
        )
        examples = json.loads((self.output / "partial_examples.json").read_text())
        images = {image.image_id: image for image, meta in self.inputs}
        rows = []
        for example in examples:
            raw = images[example["image_id"]].image_bytes
            text, _ = render_target(example)
            prompt = make_batch(processor, raw, self.extraction_prompt())
            batch = make_batch(processor, raw, self.extraction_prompt(), text)
            if batch["input_ids"].shape[1] > self.settings["max_training_tokens"]:
                raise RuntimeError("Mask audit exceeds training token budget; no truncation")
            labels, report = partial_labels(
                processor, prompt["input_ids"][0].tolist(), batch["input_ids"][0].tolist(), example
            )
            rows.append(
                {
                    "image_id": example["image_id"],
                    **report,
                    "input_tokens": len(labels),
                    "unknown_fields": [
                        n for n, f in example["fields"].items() if f["status"] == "unknown"
                    ],
                }
            )
        passed = bool(rows) and all(
            r["unknown_supervised_tokens"] == 0 and r["alignment_verified"] for r in rows
        )
        atomic_json(
            self.output / "partial_mask_audit.json",
            {
                "passed": passed,
                "rows": rows,
                "policy": "partial-fields-v1",
                "processor_revision": spec.revision,
            },
        )
        if not passed:
            raise RuntimeError("Partial supervision mask audit failed")
        print(f"Mask audit passed for {len(rows)} examples; no student weights were loaded")

    @staticmethod
    def extraction_prompt():
        from shelfscan.prompt import EXTRACT_PROMPT

        return EXTRACT_PROMPT

    def smoke(self):
        from training.smoke import run_smoke

        self.require("annotate", "mask_audit", "baseline")
        examples = json.loads((self.output / "partial_examples.json").read_text())
        images = {image.image_id: image for image, meta in self.inputs}
        settings = {**self.settings, "partial_supervision": True}
        report = run_smoke(
            self.specs["student"],
            [(images[e["image_id"]].image_bytes, e) for e in examples],
            self.output,
            settings,
        )
        if not report["ok"]:
            raise RuntimeError("Partial adapter reload differs; return reports")

    def finish(self):
        from training.runtime import export_reports

        coverage_path = self.output / "annotation_coverage.json"
        coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
        mask_path = self.output / "partial_mask_audit.json"
        masks = json.loads(mask_path.read_text()) if mask_path.exists() else {}
        actual_path = self.output / "smoke_supervision_report.json"
        actual = json.loads(actual_path.read_text()) if actual_path.exists() else {}
        checks = {
            "source_reused": (self.output / "source_import_report.json").exists(),
            "eight_title_gate": coverage.get("passed", False),
            "mask_audit": masks.get("passed", False),
            "baseline": self.progress.get("baseline") == "ok",
            "adapter_reload": self.progress.get("smoke") == "ok",
            "unknown_loss_zero": actual.get("unknown_supervised_tokens", -1) == 0,
            "gateway": self.progress.get("gateway") == "ok",
        }
        report = {
            "policy": "evidence-partial-v3",
            "passed": all(checks.values()),
            "checks": checks,
            "accepted_titles": coverage.get("accepted_titles", 0),
            "complete_records": coverage.get("complete_records", 0),
            "partial_examples": coverage.get("partial_examples", 0),
            "progress": self.progress,
            "note": "Partial-supervision feasibility; pseudo-labels are not ground truth",
        }
        atomic_json(self.output / "pilot_report.json", report)
        (self.output / "pilot_report.md").write_text(json.dumps(report, indent=2))
        return export_reports(self.output)
