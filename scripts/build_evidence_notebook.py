"""Build the evidence-first partial-supervision notebook from verified setup cells."""

# Ruff exempts notebook prose and embedded cell source from physical line limits.
# ruff: noqa: E501

import json
from pathlib import Path

from scripts.build_annotation_recovery import build_recovery
from scripts.build_colab_handoff import code, markdown
from scripts.common import json_bytes


def build_evidence(root):
    build_recovery(root)
    notebook = json.loads((root / "notebooks/recover_annotations.ipynb").read_bytes())
    cells = notebook["cells"][:11]
    cells[0] = markdown("""
    # ShelfScan evidence-first pilot v3
    Run this NEW notebook from the top, one cell at a time. Keep the original results ZIP.
    This run starts at pilot-evidence-v3 and leaves prior runs untouched. It replays saved
    value proposals and attaches matching OCR regions, without asking models to author quotes.
    Null/unknown fields stay unknown. Accepted fields receive token-aligned training loss;
    all unknown field keys/values, prompt/image tokens and boundary-crossing tokens are masked.
    Source inputs: pilot-recovery-v2_results.zip (the latest FULL archive) and pilot_inputs.zip.
    Existing runtime copies are reused; upload only when prompted. Add HF_TOKEN in Colab Secrets.
    Select a GPU runtime for later smoke training. The preview makes zero model-generation calls.
    Original eight-title threshold remains, plus eight eligible distinct book groups.
    If a stage returns False, skip later model/training cells and use the final download cells.
    These are OCR-backed pseudo-labels, not human accuracy or final project acceptance.
    """)
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        if "RUN_ID =" in text:
            text = text.replace('"pilot-recovery-v2"', '"pilot-evidence-v3"')
            text = text.replace("'pilot-recovery-v2'", "'pilot-evidence-v3'")
            text = text.replace("/content/shelfscan_recovery_v2", "/content/shelfscan_evidence_v3")
            text = text.replace(
                "/content/shelfscan_results/pilot-v6",
                "/content/shelfscan_results/pilot-recovery-v2",
            )
        if "PILOT = PROJECT" in text:
            text = text.replace(
                "Path('/content/shelfscan_t03/artifacts/pilot_inputs.zip'),",
                "Path('/content/shelfscan_recovery_v2/artifacts/pilot_inputs.zip'),\n"
                "            Path('/content/shelfscan_t03/artifacts/pilot_inputs.zip'),",
            )
        if "runtime = preflight" in text:
            text = text.replace("pilot-v6", "pilot-recovery-v2")
            text = text.replace(
                "settings['run_id'] = RUN_ID",
                "settings['run_id'] = RUN_ID\n"
                "settings['partial_supervision'] = True\nsettings['engine_version'] = 'evidence-partial-v3'",
            )
            # Parent builder starts from unformatted cells; support both quote styles.
            text = text.replace(
                'settings["run_id"] = RUN_ID',
                'settings["run_id"] = RUN_ID\n'
                'settings["partial_supervision"] = True\nsettings["engine_version"] = "evidence-partial-v3"',
            )
        if "pilot = PilotSession" in text:
            text = text.replace(
                "from training.pilot import PilotSession",
                "from training.evidence_pilot import EvidencePilot",
            )
            text = text.replace(
                "pilot = PilotSession(PROJECT, OUTPUT, settings, specs)",
                "pilot = EvidencePilot(PROJECT, OUTPUT, settings, specs, SOURCE_OUTPUT)",
            )
        cell["source"] = text.splitlines(keepends=True)
    cells += [
        markdown(
            "## 1. Offline preview (no model calls)\nInspect accepted titles, eligible groups and unknown fields. Full records may remain zero."
        ),
        code("pilot.step('evidence_preview', pilot.preview)"),
        markdown(
            "## 2. Optional bounded selector\nAutomatically skipped if preview already has eight accepted titles. Otherwise selects only offered OCR-backed title IDs; cannot invent values."
        ),
        code("pilot.step('selection', pilot.select)"),
        markdown(
            "## 3. Freeze partial examples and enforce the data gate\nRequires eight accepted titles and eight distinct eligible groups. Unknown optional fields are never relabeled absent."
        ),
        code("pilot.step('annotate', pilot.annotate)"),
        markdown(
            "## 4. Audit EVERY supervision mask before student weights load\nUses the pinned processor/tokenizer on CPU. Checks full-sequence roundtrip, unique assistant alignment and zero unknown-field loss."
        ),
        code("pilot.step('mask_audit', pilot.audit_masks)"),
        markdown(
            "## 5. Baseline and two-step masked LoRA smoke\nProceed only after the data and mask gates passed. This is a training feasibility check, not a held-out improvement claim."
        ),
        code("pilot.step('baseline', pilot.baseline)"),
        code("pilot.step('smoke', pilot.smoke)"),
        markdown(
            "## 6. Authenticated serving check\nRequires saved adapter reload. Starts loopback-only services, validates JSON and authentication, then shuts them down."
        ),
        code("pilot.step('gateway', pilot.gateway)"),
        markdown(
            "## 7. Download reports and full artifacts\nAlways run these cells, including if an earlier gate failed. Keep both files before ending the runtime."
        ),
        code("from google.colab import files\nfiles.download(str(pilot.finish()))"),
        code(
            "import zipfile\nresults_zip = OUTPUT.parent / (OUTPUT.name + '_results.zip')\n"
            "with zipfile.ZipFile(results_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:\n"
            "    for path in sorted(OUTPUT.rglob('*')):\n"
            "        if path.is_file() and path.suffix not in ('.log', '.tmp') and not path.name.startswith('.'):\n"
            "            archive.write(path, path.relative_to(OUTPUT).as_posix())\n"
            "files.download(str(results_zip))"
        ),
    ]
    notebook["cells"] = [{**cell, "id": f"evidence-v3-{i:02d}"} for i, cell in enumerate(cells)]
    notebook["metadata"]["colab"]["name"] = "evidence_partial_pilot.ipynb"
    (root / "notebooks/evidence_partial_pilot.ipynb").write_bytes(json_bytes(notebook))
    return {"notebook": "notebooks/evidence_partial_pilot.ipynb", "cells": len(cells)}


if __name__ == "__main__":
    print(json.dumps(build_evidence(Path(__file__).resolve().parent.parent), indent=2))
