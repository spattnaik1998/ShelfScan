import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import yaml

from training.model_io import VisionSession
from training.ocr import source_box, source_point
from training.pilot import pilot_gate
from training.runtime import export_reports, load_json_object, require_hf_token, resolve_models
from training.smoke import target_start


def test_token_is_required_but_not_printed(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="Colab Secrets"):
        require_hf_token()
    token = "hf_" + "dummy" * 8
    monkeypatch.setenv("HF_TOKEN", token)
    assert require_hf_token() == token
    assert token not in capsys.readouterr().out


@pytest.mark.parametrize("text", ["[]", "prefix {}", "```json\n{}", "__import__('os')"])
def test_parser_never_executes_or_repairs_arbitrary_model_text(text):
    with pytest.raises(ValueError):
        load_json_object(text)


def test_plain_and_fenced_json_objects():
    assert load_json_object('{"year":2020}') == {"year": 2020}
    assert load_json_object('```json\n{"year":2020}\n```') == {"year": 2020}


@pytest.mark.parametrize(
    "rotation,point", [(0, (0.2, 0.3)), (90, (0.7, 0.2)), (180, (0.8, 0.7)), (270, (0.3, 0.8))]
)
def test_rotated_ocr_coordinates_map_back_to_source(rotation, point):
    assert source_point(0.2, 0.3, rotation) == pytest.approx(point)
    box = source_box([[0, 0], [100, 0], [100, 50], [0, 50]], 100, 50, rotation)
    assert box.model_dump() == {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}


def test_loss_mask_requires_exact_image_prompt_prefix():
    assert target_start([10, 11, 12], [10, 11, 12, 40, 41]) == 3
    with pytest.raises(ValueError, match="prefix"):
        target_start([10, 11], [10, 99, 40])
    with pytest.raises(ValueError, match="No supervised"):
        target_start([10, 11], [10, 11])


def test_pilot_gate_requires_real_execution_not_just_coverage():
    progress = {
        name: "ok"
        for name in (
            "ocr",
            "teacher_a",
            "teacher_b",
            "annotate",
            "resume",
            "baseline",
            "smoke",
            "gateway",
        )
    }
    assert pilot_gate(progress, 8, 8, 12)["passed"]
    assert not pilot_gate(progress, 7, 8, 12)["passed"]
    assert not pilot_gate({**progress, "smoke": "failed"}, 12, 8, 12)["passed"]
    assert not pilot_gate({}, 12, 8, 12)["passed"]


def test_report_export_excludes_tokens_raw_outputs_and_weights(tmp_path):
    (tmp_path / "pilot_report.json").write_text('{"passed":false}')
    (tmp_path / ".env").write_text("SECRET")
    (tmp_path / "adapter.safetensors").write_bytes(b"weights")
    (tmp_path / "gateway.private.log").write_text("private")
    with ZipFile(export_reports(tmp_path)) as archive:
        assert archive.namelist() == ["pilot_report.json"]


def test_hub_revisions_are_resolved_once_without_serializing_token(tmp_path, monkeypatch):
    token = "hf_" + "dummy" * 8
    monkeypatch.setenv("HF_TOKEN", token)
    calls = []

    class FakeApi:
        def __init__(self, token):
            assert token == os.environ["HF_TOKEN"]

        def model_info(self, name, revision):
            calls.append((name, revision))
            return SimpleNamespace(
                sha="a" * 40,
                card_data=SimpleNamespace(
                    to_dict=lambda: {"license": "test-license"},
                ),
            )

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    config = tmp_path / "models.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "student": {
                    "model_id": "test/model",
                    "family": "qwen2_5_vl",
                    "revision": "main",
                    "parameters": {},
                }
            }
        )
    )
    output = tmp_path / "reports"
    first = resolve_models(config, output)
    second = resolve_models(config, output)
    assert first == second and len(calls) == 1
    assert first["student"].revision == "a" * 40
    assert token not in (output / "models.resolved.yaml").read_text()
    assert "test-license" in (output / "model_licenses.json").read_text()


def test_native_loader_passes_token_and_immutable_revision(monkeypatch, annotation_config):
    token = "hf_" + "dummy" * 8
    monkeypatch.setenv("HF_TOKEN", token)
    calls = []

    class Auto:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append((name, kwargs))
            return SimpleNamespace(eval=lambda: None)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            bfloat16="bf16",
            float16="fp16",
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda **kw: False),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=Auto,
            AutoModelForImageTextToText=Auto,
            BitsAndBytesConfig=lambda **kw: kw,
        ),
    )
    spec = annotation_config.roles["teacher_a"].model_copy(
        update={
            "revision": "a" * 40,
            "family": "qwen2_5_vl",
            "parameters": {"quantize": True},
        }
    )
    VisionSession(spec).load()
    assert len(calls) == 2
    assert all(
        kwargs["token"] == token
        and not kwargs["trust_remote_code"]
        and kwargs["revision"] == "a" * 40
        for _, kwargs in calls
    )
    assert calls[-1][1]["quantization_config"]["bnb_4bit_compute_dtype"] == "fp16"


def test_notebook_cells_are_valid_python_and_have_no_outputs():
    notebook = json.loads(Path("notebooks/00_pilot.ipynb").read_bytes())
    assert notebook["metadata"]["accelerator"] == "GPU"
    stages = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, "<notebook-cell>", "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
            stages.append(source)
    text = "\n".join(stages)
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "userdata"
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "HF_TOKEN"
        for source in stages
        for node in ast.walk(ast.parse(source))
    )
    assert "pilot.finish()" in text and "pilot.gateway" in text
    assert "teacher_a_check" in text and "teacher_b_check" in text
    assert "limit=1" in text
    assert "drive.mount" not in text and "/content/drive" not in text
    assert "files.upload()" in text and "results_zip" in text


def test_embedded_notebook_code_unpacks_without_external_archive(tmp_path):
    notebook = json.loads(Path("notebooks/00_pilot.ipynb").read_bytes())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "SOURCE_B64" in "".join(cell["source"])
    )
    project = tmp_path / "project"
    scope = {"PROJECT": project}
    exec(compile(source, "<notebook-bootstrap>", "exec"), scope)
    manifest = json.loads((project / "handoff_manifest.json").read_bytes())
    assert (project / "training/pilot.py").is_file()
    assert (project / "requirements/colab.lock.txt").is_file()
    assert (project / "requirements/colab.lock.txt").read_bytes() == Path(
        "requirements/colab.lock.txt"
    ).read_bytes()
    assert (project / "pyproject.toml").read_bytes() == Path("pyproject.toml").read_bytes()
    assert "artifacts/pilot_inputs.zip" not in manifest["files"]
    assert not any(name.endswith((".jpg", ".jpeg", ".png")) for name in manifest["files"])


def test_notebook_results_download_includes_adapter_and_excludes_private_logs(tmp_path):
    notebook = json.loads(Path("notebooks/00_pilot.ipynb").read_bytes())
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "results_zip = OUTPUT.parent" in "".join(cell["source"])
    )
    output = tmp_path / "pilot-v1"
    (output / "smoke_adapter").mkdir(parents=True)
    (output / "smoke_adapter/adapter.safetensors").write_bytes(b"test adapter")
    (output / "pilot_decisions.json").write_text("[]")
    (output / "gateway.private.log").write_text("private details")
    (output / ".writer.lock").write_text("lock")
    downloads = []
    exec(
        compile(source, "<notebook-results>", "exec"),
        {
            "OUTPUT": output,
            "files": SimpleNamespace(download=downloads.append),
        },
    )
    with ZipFile(downloads[0]) as archive:
        assert set(archive.namelist()) == {
            "smoke_adapter/adapter.safetensors",
            "pilot_decisions.json",
        }


def test_native_bf16_excludes_emulation():
    from training.runtime import native_bf16_supported

    calls = []
    cuda = SimpleNamespace(is_bf16_supported=lambda **kw: calls.append(kw) or False)
    assert not native_bf16_supported(SimpleNamespace(cuda=cuda))
    assert calls == [{"including_emulation": False}]


def test_teacher_prompt_has_explicit_envelope_and_no_product_schema():
    from annotation.orchestrator import ANNOTATE_PROMPT
    from training.model_io import teacher_prompt

    prompt = teacher_prompt(ANNOTATE_PROMPT)
    assert '"fields"' in prompt and '"title"' in prompt
    assert "category" not in prompt.lower()
    assert "BookRecord" not in prompt


def test_smolvlm_runtime_dependency_is_declared():
    assert "num2words==0.5.14" in Path("requirements/colab.in").read_text()


def test_teacher_invalid_response_is_saved_redacted(tmp_path, monkeypatch):
    from annotation.contracts import TeacherRequest, VisualInput
    from training.model_io import HubTeacher

    token = "hf_" + "secret" * 8
    monkeypatch.setenv("HF_TOKEN", token)
    session = SimpleNamespace(
        spec=SimpleNamespace(model_id="test/model"),
        generate=lambda *args: "invalid JSON " + token,
    )
    request = TeacherRequest(
        image=VisualInput(
            image_id="test", image_sha256="x", image_bytes=b"pixels", width=1, height=1
        ),
        prompt="ignored conflicting product format",
    )
    path = tmp_path / "teacher_diagnostics.json"
    with pytest.raises(ValueError):
        HubTeacher(session, path).extract(request)
    assert token not in path.read_text()
    assert "invalid JSON" in path.read_text()
    with ZipFile(export_reports(tmp_path)) as archive:
        assert "teacher_diagnostics.json" in archive.namelist()


@pytest.mark.parametrize("box", [[96, 107, 320, 140], "Title", "[365, 197, 472, 304]"])
def test_teacher_discards_untrusted_model_localization(box):
    from annotation.contracts import FIELDS, TeacherRequest, VisualInput
    from training.model_io import HubTeacher

    fields = {
        name: {"status": "unknown", "value": None, "evidence": [], "reason": "Uncertain"}
        for name in FIELDS
    }
    fields["title"] = {
        "status": "present",
        "value": "Example",
        "evidence": [{"quote": "Example", "box": box}],
        "reason": "Visible",
    }
    session = SimpleNamespace(generate=lambda *args: json.dumps({"fields": fields}))
    request = TeacherRequest(
        image=VisualInput(image_id="x", image_sha256="x", image_bytes=b"x", width=1, height=1),
        prompt="",
    )
    result = HubTeacher(session).extract(request)
    assert result.fields["title"].value == "Example"
    assert result.fields["title"].evidence[0].box is None


def test_compact_teacher_response_keeps_unknown_distinct_from_absent():
    from annotation.contracts import FIELDS
    from training.model_io import parse_candidate

    fields = {name: {"status": "unknown", "value": None, "quote": None} for name in FIELDS}
    fields["title"] = {"status": "present", "value": "Example", "quote": "Example"}
    fields["price"] = {"status": "absent", "value": None, "quote": None}
    result = parse_candidate(json.dumps({"fields": fields}))
    assert result.fields["title"].evidence[0].quote == "Example"
    assert result.fields["price"].status == "absent"
    assert result.fields["authors"].status == "unknown"
    fields["authors"]["value"] = "Invented"
    with pytest.raises(ValueError):
        parse_candidate(json.dumps({"fields": fields}))


def test_truncated_repeating_response_stays_invalid():
    from training.model_io import parse_candidate

    with pytest.raises(ValueError):
        parse_candidate('{"fields":{"authors":{"value":"' + "B." * 500)
