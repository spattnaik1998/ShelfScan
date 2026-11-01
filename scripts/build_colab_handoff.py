"""Build a checksummed T03 source/input bundle and an output-free Colab notebook."""

import base64
import json
import textwrap
from pathlib import Path

from scripts.common import json_bytes, sha256
from scripts.split import write_archive


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    source = textwrap.dedent(text).strip()
    embedded_setup = "SOURCE_B64 =" in source
    if embedded_setup:
        source = "#@title Load included supporting code\n" + source
    return {
        "cell_type": "code",
        "metadata": {"cellView": "form"} if embedded_setup else {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build(root: Path) -> dict:
    files = {}
    for directory in ("annotation", "shelfscan", "scripts", "training", "serving"):
        for path in sorted((root / directory).glob("*.py")):
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    for directory, suffix in (("configs", "*.yaml"), ("requirements", "*")):
        for path in sorted((root / directory).glob(suffix)):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path.read_bytes()
    for name in (
        "pyproject.toml",
        "uv.lock",
        "data/split.provisional.json",
    ):
        files[name] = (root / name).read_bytes()
    manifest = {
        "task": "T03",
        "files": {name: sha256(data) for name, data in sorted(files.items())},
    }
    files["handoff_manifest.json"] = json_bytes(manifest)
    bundle = root / "artifacts/t03_notebook_source.zip"
    write_archive(bundle, files)
    digest = sha256(bundle.read_bytes())
    source_literal = "\n".join(
        "        " + repr(chunk)
        for chunk in textwrap.wrap(base64.b64encode(bundle.read_bytes()).decode(), 76)
    )
    pilot_digest = sha256((root / "artifacts/pilot_inputs.zip").read_bytes())
    cells = [
        markdown("""
        # ShelfScan T03 — real GPU pilot

        **Status: prepared and checked locally; GPU execution is yours.** This is a
        feasibility run, not full training. Use a fresh Colab GPU runtime (T4 16GB or
        better), and allow several GB of model/package downloads. No hosted inference
        subscription is needed: HF authenticates weight downloads, Colab runs inference.

        **No Drive or OneDrive mount. Supporting Python code is included in this notebook.**
        1. Run setup, then upload `pilot_inputs.zip` when the notebook prompts you.
        2. Add a Colab Secret named `HF_TOKEN`, grant notebook access. A read token is
           enough; accept any required model access terms on HF first.
        3. Run cells in order. If installation requests a session restart, restart and
           run again from the setup cells. Reuse the same RUN_ID to resume unchanged work.
        4. Download **pilot_reports.zip** and **pilot_results.zip** using the final cells.
           The results ZIP includes the saved model adapter and annotation progress.
           Results are saved under `/content/shelfscan_results/` in the current runtime.

        Teacher A/adjudicator: Qwen2.5-VL-7B (4-bit). Teacher B: SmolVLM2-2.2B.
        Student: Qwen2.5-VL-3B (4-bit). Revisions resolve once and are saved. A failed
        gate is an honest result: do not tune on test data or describe it as passed.
    """),
        code("""
        from pathlib import Path
        RUN_ID = 'pilot-v5'  # Change only for a deliberate new experiment.
        PROJECT = Path('/content/shelfscan_t03')
        OUTPUT = Path('/content/shelfscan_results') / RUN_ID
        OUTPUT.mkdir(parents=True, exist_ok=True)
    """),
        markdown("""
        ## Load the included supporting code
        Run this setup cell as-is. It unpacks the embedded, checksummed Python helpers
        into the runtime. No code archive upload, repository clone or mounted storage
        is required. No image data or credentials are embedded in the notebook.
    """),
        code(f"""
        import base64, hashlib, io, json, zipfile
        SOURCE_B64 = (
{source_literal}
        )
        EXPECTED_SHA256 = {digest!r}
        source_bytes = base64.b64decode(SOURCE_B64)
        if hashlib.sha256(source_bytes).hexdigest() != EXPECTED_SHA256:
            raise ValueError('Embedded notebook source checksum mismatch')
        PROJECT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
            manifest = json.loads(archive.read('handoff_manifest.json'))
            if set(archive.namelist()) != set(manifest['files']) | {{'handoff_manifest.json'}}:
                raise ValueError('Unexpected bundle members')
            for name, expected in manifest['files'].items():
                data = archive.read(name)
                path = (PROJECT / name).resolve()
                if not path.is_relative_to(PROJECT.resolve()):
                    raise ValueError('Unsafe bundle path')
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError('Bundle member checksum mismatch')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        (PROJECT / 'handoff_manifest.json').write_text(json.dumps(manifest))
        print('Supporting code is ready in', PROJECT)
    """),
        markdown("""
        ## Upload only the pilot images
        Select `pilot_inputs.zip` from the local `ShelfScan/artifacts/` folder. The
        notebook verifies it and reuses it on reruns in this runtime. Original full-size
        dataset images are unnecessary. This upload is image data only, not project code.
    """),
        code(f"""
        from google.colab import files
        PILOT = PROJECT / 'artifacts/pilot_inputs.zip'
        EXPECTED_PILOT_SHA256 = {pilot_digest!r}
        pilot_ready = PILOT.exists() and (
            hashlib.sha256(PILOT.read_bytes()).hexdigest() == EXPECTED_PILOT_SHA256)
        if not pilot_ready:
            uploaded = files.upload()
            if len(uploaded) != 1:
                raise ValueError('Select exactly one file: pilot_inputs.zip')
            image_archive = next(iter(uploaded.values()))
            if hashlib.sha256(image_archive).hexdigest() != EXPECTED_PILOT_SHA256:
                raise ValueError('Incorrect pilot_inputs.zip; use the supplied 12-image package')
            PILOT.parent.mkdir(parents=True, exist_ok=True)
            PILOT.write_bytes(image_archive)
            del uploaded, image_archive
        print('Pilot inputs ready.')
    """),
        markdown("""
        ## Install pinned GPU dependencies
        This may take several minutes and replaces incompatible preinstalled versions.
        Torch is not imported before installation. If told to restart the session,
        use Runtime → Restart session, then rerun from the top. Do not manually upgrade
        individual packages. The lock targets Linux Python 3.13 and includes project
        constraints. Installed versions and Python are recorded for diagnosis.
    """),
        code("""
        import sys
        sys.path.insert(0, str(PROJECT))
        from training.install import install_environment
        install_environment(PROJECT, OUTPUT)
    """),
        markdown("""
        ## Authenticate, inspect GPU and resolve immutable model revisions
        The token is read from Colab Secrets, never hardcoded, printed, saved to Git
        credentials, passed to the gateway or written into the reports.
    """),
        code("""
        import os, re, yaml
        from google.colab import userdata
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', RUN_ID):
            raise ValueError('Use a simple run ID')
        os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        from training.runtime import preflight, resolve_models, export_reports
        settings = yaml.safe_load((PROJECT / 'configs/colab.example.yaml').read_text())
        settings['run_id'] = RUN_ID
        runtime = preflight(OUTPUT, settings['minimum_gpu_gib'])
        specs = resolve_models(PROJECT / 'configs/models.example.yaml', OUTPUT)
        print({key: runtime[key] for key in ('gpu', 'gpu_memory_gib', 'bf16')})
        print({role: {'model': spec.model_id, 'revision': spec.revision[:12]}
               for role, spec in specs.items()})
    """),
        code("""
        import torch, random
        random.seed(settings['seed'])
        torch.manual_seed(settings['seed'])
        torch.cuda.manual_seed_all(settings['seed'])
        from training.pilot import PilotSession
        pilot = PilotSession(PROJECT, OUTPUT, settings, specs)
        print('Pilot inputs verified:', len(pilot.inputs))
    """),
        markdown("""
        ## Sequential annotation passes
        Each pass saves durable results before the next model loads. Rerunning a failed
        cell reuses successful calls; the persisted one-retry limit still applies. A failed
        stage prints a sanitized error type. Always run the final report-export cell,
        even if a gate fails. Fixes that change models/config/code require a new RUN_ID.
    """),
        code("pilot.step('ocr', pilot.ocr)"),
        markdown(
            "Run these one-image checks first. "
            "Continue to full teacher passes only when both return True. "
            "Teacher B validates seven individual fields with at most two attempts each. "
            "It saves each field and reuses validated work; no invalid field is filled in."
        ),
        code("pilot.step('teacher_a_check', lambda: pilot.teacher('teacher_a', limit=1))"),
        code("pilot.step('teacher_b_check', lambda: pilot.teacher('teacher_b', limit=1))"),
        code(
            "pilot.require('teacher_a_check', 'teacher_b_check')\n"
            "pilot.step('teacher_a', lambda: pilot.teacher('teacher_a'))"
        ),
        code(
            "pilot.require('teacher_a_check', 'teacher_b_check')\n"
            "pilot.step('teacher_b', lambda: pilot.teacher('teacher_b'))"
        ),
        code("pilot.step('annotate', pilot.annotate)"),
        markdown("""
        ## Student baseline and tiny LoRA save/reload check
        Baseline predictions cover the training-only pilot. The smoke trains for two
        steps on complete accepted records, masks the entire prompt and image prefix,
        saves the adapter/processor locally and reloads into a fresh base instance.
        It demonstrates feasibility, not improved accuracy or a final usable model.
    """),
        code("pilot.step('baseline', pilot.baseline)"),
        code("pilot.step('smoke', pilot.smoke)"),
        markdown("""
        ## Authenticated gateway smoke
        Starts a private loopback model endpoint and LiteLLM gateway, checks that missing
        credentials are rejected, requests one cover through `shelfscan-local`, validates
        its JSON, and shuts both services down. No public tunnel or HF publication occurs.
    """),
        code("pilot.step('gateway', pilot.gateway)"),
        markdown("""
        ## Export the handoff report — run even after a failed stage
        Passing requires all stages, cached replay, clean adapter reload, the authenticated
        gateway check and at least 8/12 accepted titles. Excluded optional fields can reduce
        complete records. The report distinguishes those counts. Download this small
        report bundle to return to Codex; the following cell saves your adapter separately.
    """),
        code("""
        if 'pilot' in globals():
            reports = pilot.finish()
            result = json.loads((OUTPUT / 'pilot_report.json').read_text())
            print({'passed': result['passed'], 'checks': result['checks'],
                   'accepted_titles': result['accepted_titles'],
                   'complete_records': result['complete_records']})
        else:
            # Setup can fail before any project dependencies are importable.
            import zipfile
            (OUTPUT / 'pilot_report.json').write_text(json.dumps({
                'passed': False, 'stage': 'setup', 'note': 'Setup did not complete'}))
            reports = OUTPUT / 'pilot_reports.zip'
            with zipfile.ZipFile(reports, 'w') as archive:
                for name in ('pilot_report.json', 'runtime_manifest.json',
                             'models.resolved.yaml', 'model_licenses.json'):
                    path = OUTPUT / name
                    if path.is_file():
                        archive.write(path, name)
            print('Setup incomplete; returning the available reports.')
        print('Saved report bundle:', reports)
        from google.colab import files
        files.download(str(reports))
    """),
        markdown("""
        ## Download all results and the saved model adapter
        This saves results, annotation progress, the LoRA adapter, tokenizer/processor
        and base-model revision. It excludes private logs and the large base weights.
        Download both ZIPs before ending the
        runtime: `/content/` is temporary storage. Full-data training follows T04 review.
    """),
        code("""
        import zipfile
        results_zip = OUTPUT.parent / (OUTPUT.name + '_results.zip')
        with zipfile.ZipFile(results_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(OUTPUT.rglob('*')):
                if path.is_file() and path.suffix not in ('.log', '.tmp'):
                    if not path.name.startswith('.'):
                        archive.write(path, path.relative_to(OUTPUT).as_posix())
        print('Saved results and available model adapter:', results_zip)
        files.download(str(results_zip))
    """),
    ]
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"name": "00_pilot.ipynb", "provenance": []},
        },
        "cells": [{**cell, "id": f"t03-{index:02d}"} for index, cell in enumerate(cells)],
    }
    directory = root / "notebooks"
    directory.mkdir(exist_ok=True)
    (directory / "00_pilot.ipynb").write_bytes(json_bytes(notebook))
    result = {
        "embedded_source_sha256": digest,
        "embedded_source_bytes": bundle.stat().st_size,
        "input_file": "artifacts/pilot_inputs.zip",
        "input_sha256": pilot_digest,
        "notebook": "notebooks/00_pilot.ipynb",
        "cells": len(cells),
    }
    (root / "artifacts/t03_handoff.json").write_bytes(json_bytes(result))
    return result


if __name__ == "__main__":
    print(json.dumps(build(Path(__file__).resolve().parent.parent), indent=2))
