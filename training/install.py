"""Dependency bootstrap diagnostics; stdlib only so failures remain inspectable."""

import importlib.metadata as metadata
import os
import platform
import re
import subprocess
import sys
from pathlib import Path


def redact(text: str) -> str:
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "GITHUB_TOKEN", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "<REDACTED>")
    text = re.sub(r"(?:hf_|gh[pousr]_)[A-Za-z0-9]{10,}", "<REDACTED>", text)
    text = re.sub(r"(https?://)[^/\s@]+@", r"\1<REDACTED>@", text)
    return re.sub(r"(?i)(Bearer\s+)\S+", r"\1<REDACTED>", text)


def install_environment(project: Path, output: Path) -> None:
    tracked = {
        "torch": "torch",
        "numpy": "numpy",
        "PIL": "Pillow",
        "pydantic": "pydantic",
        "transformers": "transformers",
    }
    loaded = {
        module: str(getattr(sys.modules[module], "__version__", ""))
        for module in tracked
        if module in sys.modules
    }
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "install.private.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(project / "requirements/colab.lock.txt"),
                "-e",
                str(project),
            ],
            stdout=log,
            stderr=log,
        )
    if result.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-70:])
        diagnostic = (
            f"Dependency installation failed (exit {result.returncode}).\n"
            f"Python: {platform.python_version()}; platform: {sys.platform}.\n"
            "Redacted pip log tail:\n" + redact(tail)
        )
        (output / "install_error.redacted.txt").write_text(diagnostic, encoding="utf-8")
        raise RuntimeError(diagnostic)
    stale = [
        module
        for module, version in loaded.items()
        if version and version != metadata.version(tracked[module])
    ]
    if stale:
        raise RuntimeError("Packages changed in memory: restart session, rerun from top")
    print("Pinned environment installed.")
