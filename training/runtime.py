"""Runtime preflight, reproducible Hub access and safe report export."""

import importlib.metadata
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import yaml

from annotation.cache import atomic_json
from annotation.contracts import RoleSpec


def native_bf16_supported(torch) -> bool:
    """Require native GPU support; T4 must use FP16 rather than emulated BF16."""
    return torch.cuda.is_bf16_supported(including_emulation=False)


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token.startswith("hf_") or len(token) < 10:
        raise RuntimeError("Add HF_TOKEN in Colab Secrets and grant notebook access")
    return token


def preflight(output: Path, minimum_gpu_gib: float = 14.0) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab, then rerun")
    properties = torch.cuda.get_device_properties(0)
    report = {
        "python": platform.python_version(),
        "gpu": properties.name,
        "gpu_memory_gib": round(properties.total_memory / 1024**3, 2),
        "cuda": torch.version.cuda,
        "bf16": native_bf16_supported(torch),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "peft",
                "accelerate",
                "bitsandbytes",
                "huggingface-hub",
                "easyocr",
                "litellm",
                "pydantic",
                "pillow",
            )
        },
    }
    atomic_json(output / "runtime_manifest.json", report)
    (output / "dependency_snapshot.txt").write_text(
        subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True),
        encoding="utf-8",
    )
    if report["gpu_memory_gib"] < minimum_gpu_gib:
        raise RuntimeError("GPU memory is below configured pilot minimum; return runtime report")
    return report


def resolve_models(source: Path, output: Path) -> dict[str, RoleSpec]:
    from huggingface_hub import HfApi

    target = output / "models.resolved.yaml"
    requested = yaml.safe_load(source.read_text(encoding="utf-8"))
    if target.exists():
        resolved = yaml.safe_load(target.read_text(encoding="utf-8"))
        if set(resolved) != set(requested):
            raise ValueError("Resolved role set differs; use a new run_id")
        for role, spec in requested.items():
            old = resolved[role]
            if any(old[key] != spec[key] for key in ("model_id", "family", "parameters")):
                raise ValueError("Model config changed; use a new run_id")
            if spec["revision"] != "main" and old["revision"] != spec["revision"]:
                raise ValueError("Requested revision changed; use a new run_id")
    else:
        api = HfApi(token=require_hf_token())
        resolved = {}
        licenses = {}
        info_cache = {}
        for role, spec in requested.items():
            key = (spec["model_id"], spec["revision"])
            if key not in info_cache:
                info_cache[key] = api.model_info(spec["model_id"], revision=spec["revision"])
            info = info_cache[key]
            if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
                raise RuntimeError("Hub did not resolve an immutable model revision")
            resolved[role] = {**spec, "revision": info.sha}
            card = info.card_data
            card = card.to_dict() if hasattr(card, "to_dict") else dict(card or {})
            licenses[role] = card.get("license", "see model card")
        output.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
        atomic_json(output / "model_licenses.json", licenses)
    if any(not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]) for spec in resolved.values()):
        raise ValueError("Saved model revisions must be immutable Hub commit hashes")
    return {name: RoleSpec.model_validate(spec) for name, spec in resolved.items()}


def load_json_object(text: str) -> dict:
    """Accept a plain object or one fenced object; never evaluate model text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Unclosed model JSON fence")
        text = "\n".join(lines[1:-1])
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Expected one JSON object")
    return result


def export_reports(output: Path) -> Path:
    # Whitelist small sanitized reports; exclude weights, raw responses, env and credentials.
    names = [
        "pilot_report.json",
        "pilot_report.md",
        "runtime_manifest.json",
        "models.resolved.yaml",
        "model_licenses.json",
        "dependency_snapshot.txt",
        "smoke_reload_report.json",
        "gateway_report.json",
        "progress.json",
        "input_checksums.json",
        "stage_errors.json",
        "teacher_diagnostics.json",
        "teacher_a_diagnostics.json",
        "teacher_b_diagnostics.json",
        "teacher_b_field_report.json",
        "teacher_b_benchmark.json",
        "teacher_a_check_report.json",
        "teacher_b_check_report.json",
        "teacher_a_report.json",
        "teacher_b_report.json",
        "adjudicator_report.json",
        "resume_report.json",
    ]
    path = output / "pilot_reports.zip"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name in names:
            candidate = output / name
            if candidate.is_file():
                archive.write(candidate, name)
    return path
