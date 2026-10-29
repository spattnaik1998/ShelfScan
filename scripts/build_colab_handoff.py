"""Build a checksummed T03 source/input bundle and an output-free Colab notebook."""

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
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": textwrap.dedent(text).strip().splitlines(keepends=True),
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
        "artifacts/pilot_inputs.zip",
        "data/split.provisional.json",
    ):
        files[name] = (root / name).read_bytes()
    manifest = {
        "task": "T03",
        "files": {name: sha256(data) for name, data in sorted(files.items())},
    }
    files["handoff_manifest.json"] = json_bytes(manifest)
    bundle = root / "artifacts/t03_colab_inputs.zip"
    write_archive(bundle, files)
    digest = sha256(bundle.read_bytes())
    cells = [
        markdown("""
        # ShelfScan T03 — real GPU pilot

        **Status: prepared and checked locally; GPU execution is yours.** This is a
        feasibility run, not full training. Use a fresh Colab GPU runtime (T4 16GB or
        better), and allow several GB of model/package downloads. No hosted inference
        subscription is needed: HF authenticates weight downloads, Colab runs inference.

        1. Save `t03_colab_inputs.zip` to `MyDrive/shelfscan/t03_colab_inputs.zip`.
        2. Add a Colab Secret named `HF_TOKEN`, grant notebook access. A read token is
           enough; accept any required model access terms on HF first.
        3. Run cells in order. If installation requests a session restart, restart and
           run again from the setup cells. Reuse the same RUN_ID to resume unchanged work.
        4. Keep the smoke adapter on Drive and return **pilot_reports.zip** to Codex.

        Teacher A/adjudicator: Qwen2.5-VL-7B (4-bit). Teacher B: SmolVLM2-2.2B.
        Student: Qwen2.5-VL-3B (4-bit). Revisions resolve once and are saved. A failed
        gate is an honest result: do not tune on test data or describe it as passed.
    """),
        code("""
        from google.colab import drive
        drive.mount('/content/drive')
        from pathlib import Path
        RUN_ID = 'pilot-v1'  # Change only for a deliberate new experiment.
        BUNDLE = Path('/content/drive/MyDrive/shelfscan/t03_colab_inputs.zip')
        PROJECT = Path('/content/shelfscan_t03')
        OUTPUT = Path('/content/drive/MyDrive/shelfscan/runs') / RUN_ID / 'pilot'
        OUTPUT.mkdir(parents=True, exist_ok=True)
    """),
        markdown("""
        ## Verify and unpack the exact handoff
        This verifies the bundle before importing its code. The dataset and all role
        outputs stay on your runtime/Drive; tokens are never included in artifacts.
    """),
        code(f"""
        import hashlib, json, zipfile
        EXPECTED_SHA256 = {digest!r}
        if hashlib.sha256(BUNDLE.read_bytes()).hexdigest() != EXPECTED_SHA256:
            raise ValueError('Wrong or modified T03 bundle; use its matching notebook')
        PROJECT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BUNDLE) as archive:
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
        print('Verified source and 12-image pilot inputs.')
    """),
        markdown("""
        ## Install pinned GPU dependencies
        This may take several minutes and replaces incompatible preinstalled versions.
        Torch is not imported before installation. If told to restart the session,
        use Runtime → Restart session, then rerun from the top. Do not manually upgrade
        individual packages. The lock targets Linux Python 3.11; installed versions and
        the actual Python version are recorded for compatibility diagnosis.
    """),
        code("""
        import sys, subprocess, importlib.metadata as metadata
        tracked = {'torch': 'torch', 'numpy': 'numpy', 'PIL': 'Pillow',
                   'pydantic': 'pydantic', 'transformers': 'transformers'}
        loaded = {module: str(getattr(sys.modules[module], '__version__', ''))
                  for module in tracked if module in sys.modules}
        install_log = OUTPUT / 'install.private.log'
        with install_log.open('w') as log:
            result = subprocess.run([sys.executable, '-m', 'pip', 'install',
                '-r', str(PROJECT / 'requirements/colab.lock.txt'), '-e', str(PROJECT)],
                stdout=log, stderr=log)
        if result.returncode:
            raise RuntimeError('Install failed; inspect private log before sharing')
        stale = [module for module, version in loaded.items()
                 if version and version != metadata.version(tracked[module])]
        if stale:
            raise RuntimeError('Packages changed in memory: restart session, rerun from top')
        sys.path.insert(0, str(PROJECT))
        print('Pinned environment installed.')
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
        code("pilot.step('teacher_a', lambda: pilot.teacher('teacher_a'))"),
        code("pilot.step('teacher_b', lambda: pilot.teacher('teacher_b'))"),
        code("pilot.step('annotate', pilot.annotate)"),
        markdown("""
        ## Student baseline and tiny LoRA save/reload check
        Baseline predictions cover the training-only pilot. The smoke trains for two
        steps on complete accepted records, masks the entire prompt and image prefix,
        saves the adapter/processor on Drive and reloads into a fresh base instance.
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
        complete records. The report distinguishes those counts. Leave the adapter in
        `smoke_adapter/` on Drive; return only this small report bundle for the next task.
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
        If setup failed before `pilot` was constructed, collect available reports with:
        `export_reports(OUTPUT)` (after importing it from `training.runtime`). Do not
        paste HF_TOKEN or export environment variables. Reference APIs and limitations:
        see the repository's `docs/T03_COLAB.md`. Full-data annotation is T04, after
        this GPU pilot is reviewed.
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
        "bundle": bundle.name,
        "sha256": digest,
        "bytes": bundle.stat().st_size,
        "notebook": "notebooks/00_pilot.ipynb",
        "cells": len(cells),
    }
    (root / "artifacts/t03_handoff.json").write_bytes(json_bytes(result))
    return result


if __name__ == "__main__":
    print(json.dumps(build(Path(__file__).resolve().parent.parent), indent=2))
