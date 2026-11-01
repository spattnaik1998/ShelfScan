"""Phased Colab pilot: persistent OCR/A/B passes, adjudication, smoke and reports."""

import json
import traceback
from pathlib import Path

from annotation.cache import atomic_json
from annotation.contracts import AnnotationConfig, Candidate, OCRResult, RoleSpec, TeacherRequest
from annotation.orchestrator import ANNOTATE_PROMPT, AnnotationEngine, RoleFailed
from scripts.annotate import load_inputs, single_writer
from scripts.common import sha256
from shelfscan.prompt import EXTRACT_PROMPT
from shelfscan.schemas import BookRecord
from training.install import redact
from training.model_io import HubAdjudicator, HubTeacher, VisionSession
from training.ocr import EasyOCRWorker
from training.runtime import export_reports, load_json_object


class CacheOnly:
    def __getattr__(self, name):
        def unavailable(*args):
            raise RuntimeError("Required role pass is incomplete; run earlier notebook cells")

        return unavailable


def pilot_gate(progress: dict, titles: int, minimum: int, inputs: int) -> dict:
    checks = {
        "twelve_inputs": inputs == 12,
        "ocr_complete": progress.get("ocr") == "ok",
        "teacher_a_complete": progress.get("teacher_a") == "ok",
        "teacher_b_complete": progress.get("teacher_b") == "ok",
        "annotation_complete": progress.get("annotate") == "ok",
        "accepted_titles": titles >= minimum,
        "cache_resume": progress.get("resume") == "ok",
        "baseline_complete": progress.get("baseline") == "ok",
        "adapter_reload": progress.get("smoke") == "ok",
        "gateway_complete": progress.get("gateway") == "ok",
    }
    return {"passed": all(checks.values()), "checks": checks}


class PilotSession:
    def __init__(self, root: Path, output: Path, settings: dict, specs: dict[str, RoleSpec]):
        self.root, self.output, self.settings, self.specs = root, output, settings, specs
        self.config = AnnotationConfig(
            backend="colab",
            engine_version="colab-pilot-v1",
            input_path="artifacts/pilot_inputs.zip",
            image_root="data",
            roles={role: specs[role] for role in ("teacher_a", "teacher_b", "adjudicator")}
            | {
                "ocr": RoleSpec(
                    model_id="EasyOCR/en",
                    revision="1.7.2",
                    family="easyocr",
                    parameters={"rotations": ",".join(map(str, settings["ocr_rotations"]))},
                ),
            },
            seed=settings["seed"],
            max_total_attempts=settings["max_total_attempts"],
            max_seconds=settings["max_seconds"],
        )
        self.inputs, digest = load_inputs(self.config, root)
        identity = {
            "input_sha256": digest,
            "bundle_manifest_sha256": sha256((root / "handoff_manifest.json").read_bytes()),
            "models": {role: spec.model_dump(mode="json") for role, spec in specs.items()},
            "settings": {
                k: v
                for k, v in settings.items()
                if k
                not in (
                    "max_total_attempts",
                    "max_seconds",
                )
            },
            "runtime": json.loads((output / "runtime_manifest.json").read_bytes()),
        }
        path = output / "input_checksums.json"
        if path.exists() and json.loads(path.read_bytes()) != identity:
            raise ValueError(
                "Source, data, model, settings or runtime changed; choose a new run_id"
            )
        atomic_json(path, identity)
        self.progress = (
            json.loads((output / "progress.json").read_bytes())
            if (output / "progress.json").exists()
            else {}
        )

    def engine(self, **adapters):
        workers = {role: CacheOnly() for role in ("ocr", "teacher_a", "teacher_b", "adjudicator")}
        workers.update(adapters)
        return AnnotationEngine(self.config, self.output / "annotation", **workers)

    def step(self, name: str, action):
        with single_writer(self.output / "stage_lock"):
            if self.progress.get(name) == "ok":
                print(f"{name}: already complete; reused saved artifacts")
                return True
            try:
                action()
                self.progress[name] = "ok"
                print(f"{name}: complete")
                return True
            except Exception as exc:
                self.progress[name] = "failed"
                path = self.output / "stage_errors.json"
                errors = json.loads(path.read_bytes()) if path.exists() else {}
                # Redact credentials before exporting actionable traceback diagnostics.
                errors[name] = {
                    "error_type": type(exc).__name__,
                    "detail": redact(traceback.format_exc())[-12000:],
                    "retry": "Rerun failed cell",
                }
                atomic_json(path, errors)
                print(f"{name}: failed ({type(exc).__name__}); report bundle is still available")
                print(redact(str(exc))[:2000])
                return False
            finally:
                atomic_json(self.output / "progress.json", self.progress)

    def require(self, *names):
        if any(self.progress.get(name) != "ok" for name in names):
            raise RuntimeError("Required earlier stage has not passed")

    def ocr(self):
        worker = EasyOCRWorker(self.settings["ocr_languages"], self.settings["ocr_rotations"])
        engine = self.engine(ocr=worker)
        failures = 0
        for image, _ in self.inputs:
            try:
                engine.call("ocr", image, image, OCRResult, worker.read)
            except RoleFailed:
                failures += 1
        if failures:
            raise RuntimeError("OCR calls failed; inspect private role errors")

    def teacher(self, role: str, limit: int | None = None):
        self.require("ocr")
        session = VisionSession(self.specs[role])
        worker = HubTeacher(session, self.output / "teacher_diagnostics.json")
        engine = self.engine(**{role: worker})
        valid = 0
        try:
            for image, _ in self.inputs[:limit]:
                try:
                    engine.call(
                        role,
                        image,
                        TeacherRequest(image=image, prompt=ANNOTATE_PROMPT),
                        Candidate,
                        worker.extract,
                    )
                    valid += 1
                except RoleFailed as exc:
                    if valid == 0:
                        raise RuntimeError(
                            "First-image teacher check failed; inspect teacher_diagnostics.json"
                        ) from exc
                    continue
        finally:
            session.close()
            atomic_json(
                self.output / (f"{role}_check_report.json" if limit else f"{role}_report.json"),
                {
                    "valid": valid,
                    "total": len(self.inputs[:limit]),
                    "metrics": session.metrics() if hasattr(session, "metrics") else None,
                },
            )
        if valid == 0:
            raise RuntimeError("Teacher produced no valid annotation responses")

    def annotate(self):
        self.require("ocr", "teacher_a", "teacher_b")
        session = VisionSession(self.specs["adjudicator"])
        engine = self.engine(adjudicator=HubAdjudicator(session))
        decisions = []
        try:
            for image, metadata in self.inputs:
                decisions.append(engine.annotate(image, metadata))
                atomic_json(self.output / "pilot_decisions.json", decisions)
        finally:
            session.close()
            atomic_json(
                self.output / "adjudicator_report.json",
                {
                    "metrics": session.metrics() if hasattr(session, "metrics") else None,
                },
            )
        self.resume()

    def resume(self):
        # Recompute solely from validated cached responses; no GPU/worker calls allowed.
        engine = self.engine()
        decisions = [engine.annotate(image, metadata) for image, metadata in self.inputs]
        if engine.new_attempts:
            raise RuntimeError("Cached replay unexpectedly invoked workers")
        original = json.loads((self.output / "pilot_decisions.json").read_bytes())
        if decisions != original:
            raise RuntimeError("Replay decisions differ")
        atomic_json(
            self.output / "resume_report.json",
            {"ok": True, "new_attempts": 0, "cache_hits": engine.cache_hits},
        )
        self.progress["resume"] = "ok"

    def baseline(self):
        self.require("annotate")
        session = VisionSession(self.specs["student"])
        path = self.output / "baseline_predictions.json"
        predictions = json.loads(path.read_bytes()) if path.exists() else {}
        try:
            for image, metadata in self.inputs:
                if metadata["id"] in predictions:
                    continue
                text = session.generate(image.image_bytes, EXTRACT_PROMPT)
                try:
                    BookRecord.model_validate(load_json_object(text))
                    valid = True
                except ValueError:
                    valid = False
                predictions[metadata["id"]] = {"text": text, "schema_valid": valid}
                atomic_json(path, predictions)
        finally:
            session.close()

    def smoke(self):
        from training.smoke import run_smoke

        self.require("baseline", "annotate")
        records = {
            d["id"]: d["record"]
            for d in json.loads((self.output / "pilot_decisions.json").read_bytes())
            if d["record"] is not None
        }
        examples = [
            (image.image_bytes, records[meta["id"]])
            for image, meta in self.inputs
            if meta["id"] in records
        ]
        report = run_smoke(self.specs["student"], examples, self.output, self.settings)
        if not report["ok"]:
            raise RuntimeError("Clean adapter reload did not reproduce decoded output")

    def gateway(self):
        from training.gateway import gateway_smoke

        self.require("smoke")
        result = gateway_smoke(self.specs["student"], self.inputs[0][0].image_bytes, self.output)
        if not result["ok"]:
            raise RuntimeError("Authenticated gateway smoke did not pass")

    def finish(self) -> Path:
        path = self.output / "pilot_decisions.json"
        decisions = json.loads(path.read_bytes()) if path.exists() else []
        titles = sum(d["fields"]["title"]["status"] == "accepted" for d in decisions)
        gate = pilot_gate(
            self.progress, titles, self.settings["minimum_accepted_titles"], len(self.inputs)
        )
        report = {
            **gate,
            "accepted_titles": titles,
            "complete_records": sum(d["record"] is not None for d in decisions),
            "input_count": len(self.inputs),
            "progress": self.progress,
            "notes": [
                "Automated pseudo-labels, not human ground truth",
                "Pilot uses training-only inputs; smoke adapter is not the final trained model",
                "Teacher/adjudicator A share a checkpoint; no independent judge claim",
            ],
        }
        atomic_json(self.output / "pilot_report.json", report)
        (self.output / "pilot_report.md").write_text(
            "# ShelfScan GPU pilot\n\n" + f"Passed: {gate['passed']}; "
            f"accepted titles: {titles}/{len(self.inputs)}.\n\n"
            "Model quality is measured by user execution, not by local stub tests.\n",
            encoding="utf-8",
        )
        return export_reports(self.output)
