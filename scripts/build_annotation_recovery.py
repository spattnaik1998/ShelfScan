"""Build a self-contained recovery notebook by extending the canonical pilot setup."""

import json
from pathlib import Path

from scripts.build_colab_handoff import build, markdown
from scripts.common import json_bytes


def build_recovery(root):
    build(root)
    notebook = json.loads((root / "notebooks/00_pilot.ipynb").read_bytes())
    cells = notebook["cells"]
    cells[0] = markdown("""
    # ShelfScan: consolidated annotation recovery
    Run this notebook from the top. No manual patch cells or Drive mounting are needed.
    It creates a separate pilot-recovery-v2 run and leaves pilot-v6 results intact.
    It reuses verified OCR and successful teacher responses. Failed responses get
    at most two new structured generations per image and role; unresolved fields
    explicitly abstain. Adjudication has bounded context, tokens and retries.
    If the old runtime data is unavailable, upload pilot-v6_results.zip when prompted,
    followed by pilot_inputs.zip if requested. Both files already exist locally.
    HF_TOKEN is read from Colab Secrets. GPU loading/quality remain user-verified.
    Run ONE cell at a time. If a teacher or annotation stage returns False, skip GPU
    training/serving and run the final download cells. Do not bypass the coverage gate.
    """)
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "RUN_ID =" in source:
            source = source.replace('"pilot-v9"', '"pilot-recovery-v2"').replace(
                "'pilot-v9'", "'pilot-recovery-v2'"
            )
            source = source.replace("/content/shelfscan_t03", "/content/shelfscan_recovery_v2")
            source += (
                "\nSOURCE_OUTPUT = Path('/content/shelfscan_results/pilot-v6')\n"
                "for old_variable in ('pilot', 'active_pilot'):\n"
                "    globals().pop(old_variable, None)\n"
            )
            cell["source"] = source.splitlines(keepends=True)
        if "sys.path.insert" in source:
            # A paused runtime may still have old project classes/hooks loaded.
            source = source.replace(
                "from training.install import",
                "for module_name in list(sys.modules):\n"
                "    if module_name.split('.')[0] in (\n"
                "            'training','annotation','scripts','shelfscan','serving'):\n"
                "        sys.modules.pop(module_name)\nfrom training.install import",
            )
            cell["source"] = source.splitlines(keepends=True)
        if "PILOT = PROJECT" in source:
            source = source.replace(
                "if not pilot_ready:",
                "if not pilot_ready:\n"
                "    for candidate in (\n"
                "            Path('/content/shelfscan_t03/artifacts/pilot_inputs.zip'),\n"
                "            Path('/content/pilot_inputs.zip')):\n"
                "        if candidate.exists() and (\n"
                "                hashlib.sha256(candidate.read_bytes()).hexdigest()\n"
                "                == EXPECTED_PILOT_SHA256):\n"
                "            PILOT.parent.mkdir(parents=True, exist_ok=True)\n"
                "            PILOT.write_bytes(candidate.read_bytes())\n"
                "            pilot_ready = True\n"
                "            break\nif not pilot_ready:",
            )
            cell["source"] = source.splitlines(keepends=True)
        if "runtime = preflight" in source:
            source = source.replace(
                "specs = resolve_models",
                "from training.recovery import restore_results, pin_source_models\n"
                "if not (SOURCE_OUTPUT / 'annotation/raw').is_dir():\n"
                "    print('Upload the full pilot-v6_results.zip')\n"
                "    uploaded_results = files.upload()\n"
                "    if len(uploaded_results) != 1:\n"
                "        raise ValueError('Select exactly one full results archive')\n"
                "    import io\n"
                "    SOURCE_OUTPUT = Path('/content/shelfscan_imported/pilot-v6')\n"
                "    restore_results(\n"
                "        io.BytesIO(next(iter(uploaded_results.values()))), SOURCE_OUTPUT)\n"
                "    del uploaded_results\n"
                "pin_source_models(PROJECT, SOURCE_OUTPUT)\n"
                "specs = resolve_models",
            )
            cell["source"] = source.splitlines(keepends=True)
        if "pilot = PilotSession" in source:
            source += (
                "\nfrom training.recovery import import_successes\n"
                "print(import_successes(pilot, SOURCE_OUTPUT))\n"
            )
            cell["source"] = source.splitlines(keepends=True)
    # Teacher title retries and consensus support remain installed by the canonical setup.
    notebook["metadata"]["colab"]["name"] = "recover_annotations.ipynb"
    notebook["cells"] = [{**cell, "id": f"recovery-{i:02d}"} for i, cell in enumerate(cells)]
    (root / "notebooks/recover_annotations.ipynb").write_bytes(json_bytes(notebook))
    return {"notebook": "notebooks/recover_annotations.ipynb", "cells": len(cells)}


if __name__ == "__main__":
    print(json.dumps(build_recovery(Path(__file__).resolve().parent.parent), indent=2))
