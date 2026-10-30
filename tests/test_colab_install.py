import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_install_failure_reports_pip_error_without_credentials(tmp_path, monkeypatch):
    notebook = json.loads(Path("notebooks/00_pilot.ipynb").read_bytes())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and (
            "install.private.log" in "".join(cell["source"])
            or "install_environment(" in "".join(cell["source"])
        )
    )
    secret = "hf_" + "privatevalue" * 4
    monkeypatch.setenv("HF_TOKEN", secret)

    def fail_install(*args, **kwargs):
        kwargs["stdout"].write(
            "ERROR: No matching distribution found for example-package\n"
            + secret
            + " https://reader:password@example.org/simple\n"
        )
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(subprocess, "run", fail_install)
    with pytest.raises(RuntimeError) as caught:
        exec(
            compile(source, "<installer-cell>", "exec"),
            {
                "PROJECT": tmp_path,
                "OUTPUT": tmp_path,
            },
        )
    message = str(caught.value)
    assert "No matching distribution found for example-package" in message
    assert secret not in message and "reader:password" not in message


def test_gpu_lock_satisfies_project_runtime_dependency_constraints():
    import re
    import tomllib

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    project = tomllib.loads(Path("pyproject.toml").read_text())
    locked = {}
    for line in Path("requirements/colab.lock.txt").read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^ ;]+)(?:.*)", line)
        if match:
            locked[canonicalize_name(match[1])] = match[2]
    for dependency in project["project"]["dependencies"]:
        requirement = Requirement(dependency)
        version = locked[canonicalize_name(requirement.name)]
        assert requirement.specifier.contains(version), (
            f"GPU lock pins {requirement.name}=={version}, incompatible with {dependency}"
        )
