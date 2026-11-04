import json
from types import SimpleNamespace

import pytest

from annotation.contracts import FIELDS, Box, Candidate, OCRResult, OCRSpan


def test_python_literal_and_bad_optional_field_preserve_valid_title():
    from training.structured import parse_fields

    raw = {name: {"status": "absent", "value": None, "quote": None} for name in FIELDS}
    raw["title"] = {"status": "present", "value": "Deep Work", "quote": "Deep Work"}
    raw["publisher"] = {"status": "present", "value": "N", "quote": ""}
    fields, invalid = parse_fields(str({"fields": raw}))
    assert fields["title"].value == "Deep Work"
    assert fields["publisher"].status == "unknown"
    assert set(invalid) == {"publisher"}


def test_response_parser_never_executes_model_code():
    from training.structured import parse_fields

    fields, invalid = parse_fields("__import__('os').system('anything')")
    assert len(invalid) == 7
    assert all(claim.status == "unknown" for claim in fields.values())


def test_adjudicator_context_has_a_hard_character_bound():
    from annotation.contracts import AdjudicationRequest, FieldClaim, VisualInput
    from training.structured import adjudication_prompt

    candidate = Candidate(
        fields={
            name: FieldClaim(status="unknown", value=None, reason="uncertain") for name in FIELDS
        }
    )
    ocr = OCRResult(
        spans=[
            OCRSpan(
                text="long text " * 100, confidence=0.9, box=Box(x0=0.1, y0=0.1, x1=0.9, y1=0.2)
            )
            for _ in range(200)
        ]
    )
    request = AdjudicationRequest(
        image=VisualInput(image_id="x", image_sha256="a" * 64, image_bytes=b"x", width=1, height=1),
        prompt="",
        disputed_fields=["title"],
        candidates=[candidate, candidate],
        ocr=ocr,
        order_seed=1,
    )
    prompt = adjudication_prompt(request, "title")
    assert len(prompt) <= 4000
    assert "box" not in prompt


def test_targeted_retry_preserves_valid_fields_and_budget_on_resume(tmp_path):
    from annotation.contracts import RoleSpec, VisualInput
    from training.structured import StructuredTeacher

    calls = []
    first = {name: {"status": "absent", "value": None, "quote": None} for name in FIELDS}
    first["title"] = {"status": "present", "value": "Deep Work", "quote": "Deep Work"}
    first["publisher"] = {"status": "present", "value": "N", "quote": ""}

    class Session:
        spec = RoleSpec(model_id="test/model", revision="a" * 40, family="test")

        def generate(self, raw, prompt, **kwargs):
            calls.append(prompt)
            if len(calls) == 1:
                return str({"fields": first})
            return json.dumps(
                {
                    "fields": {
                        "publisher": {"status": "unknown", "value": None, "quote": None},
                        "title": {
                            "status": "present",
                            "value": "UNREQUESTED CHANGE",
                            "quote": "UNREQUESTED CHANGE",
                        },
                    }
                }
            )

    request = SimpleNamespace(
        image=VisualInput(image_id="x", image_sha256="b" * 64, image_bytes=b"x", width=1, height=1),
        prompt="",
    )
    session = Session()
    candidate = StructuredTeacher(session, tmp_path, "teacher_b").extract(request)
    assert candidate.fields["title"].value == "Deep Work"
    assert candidate.fields["publisher"].status == "unknown"
    assert len(calls) == 2
    assert StructuredTeacher(session, tmp_path, "teacher_b").extract(request) == candidate
    assert len(calls) == 2


def test_saved_archive_import_reuses_verified_roles_without_gpu(tmp_path):
    import shutil
    from pathlib import Path
    from zipfile import ZipFile

    import yaml

    from annotation.contracts import RoleSpec
    from training.pilot import PilotSession
    from training.recovery import import_successes, restore_results

    if not all(
        Path(name).exists()
        for name in (
            "pilot-v6_results.zip",
            "artifacts/t03_notebook_source.zip",
            "artifacts/pilot_inputs.zip",
        )
    ):
        pytest.skip("Optional integration check requires local returned Colab artifacts")
    source = tmp_path / "source"
    restore_results(Path("pilot-v6_results.zip"), source)
    root = tmp_path / "project"
    root.mkdir()
    with ZipFile("artifacts/t03_notebook_source.zip") as z:
        z.extractall(root)
    (root / "artifacts").mkdir(exist_ok=True)
    shutil.copyfile("artifacts/pilot_inputs.zip", root / "artifacts/pilot_inputs.zip")
    specs = {
        name: RoleSpec.model_validate(value)
        for name, value in yaml.safe_load((source / "models.resolved.yaml").read_text()).items()
    }
    output = tmp_path / "recovery"
    output.mkdir()
    (output / "runtime_manifest.json").write_text("{}")
    settings = yaml.safe_load(Path("configs/colab.example.yaml").read_text())
    pilot = PilotSession(root, output, settings, specs)
    report = import_successes(pilot, source)
    assert report["reused"] == {"ocr": 12, "teacher_a": 11, "teacher_b": 4}
    assert report["new_generation_calls"] == 0
    assert report["seeded_failures"] == {"teacher_a": 1, "teacher_b": 8}
    assert pilot.progress == {"ocr": "ok"}
    assert import_successes(pilot, source)["reused"] == report["reused"]


def test_recover_notebook_is_self_contained_and_uses_a_new_output():
    from pathlib import Path

    notebook = json.loads(Path("notebooks/recover_annotations.ipynb").read_bytes())
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            compile(source, "<recovery-cell>", "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None
            sources.append(source)
    text = "\n".join(sources)
    assert "pilot-recovery-v2" in text
    assert "/content/shelfscan_recovery_v2" in text
    assert "import_successes(pilot, SOURCE_OUTPUT)" in text
    assert "pin_source_models(PROJECT, SOURCE_OUTPUT)" in text
    assert "drive.mount" not in text


def test_import_rejects_archive_path_traversal(tmp_path):
    from io import BytesIO
    from zipfile import ZipFile

    from training.recovery import restore_results

    archive = BytesIO()
    with ZipFile(archive, "w") as z:
        z.writestr("../escape.json", "{}")
    archive.seek(0)
    with pytest.raises(ValueError, match="Unsafe"):
        restore_results(archive, tmp_path / "import")
    assert not (tmp_path / "escape.json").exists()


def test_gpu_failure_code_is_preserved_in_role_ledger(tmp_path, annotation_config, visual):
    from annotation.contracts import TeacherRequest
    from annotation.orchestrator import AnnotationEngine, RoleFailed
    from annotation.teachers import FakeAdjudicator, FakeOCR, FakeTeacher, ModelCallError

    engine = AnnotationEngine(
        annotation_config,
        tmp_path,
        ocr=FakeOCR(),
        teacher_a=FakeTeacher(),
        teacher_b=FakeTeacher(),
        adjudicator=FakeAdjudicator(),
    )

    def fail(request):
        raise ModelCallError("gpu_oom", retryable=False)

    with pytest.raises(RoleFailed):
        engine.call("teacher_a", visual, TeacherRequest(image=visual, prompt=""), Candidate, fail)
    state = engine.cache.load(next(engine.cache.root.glob("*.json")).stem)
    assert state["attempts"][0]["error_code"] == "gpu_oom"


def setup_pilot(dataset_config, annotation_config, monkeypatch):
    from pathlib import Path

    import yaml

    from scripts.prepare_images import prepare_dataset
    from scripts.split import build_split
    from training.pilot import PilotSession

    prepare_dataset(dataset_config)
    build_split(dataset_config)
    root = dataset_config.parent.parent
    (root / "handoff_manifest.json").write_text("{}")
    output = root / "reports"
    output.mkdir()
    (output / "runtime_manifest.json").write_text("{}")

    class Session:
        def __init__(self, spec):
            pass

        def close(self):
            pass

    monkeypatch.setattr("training.pilot.VisionSession", Session)
    settings = yaml.safe_load(Path("configs/colab.example.yaml").read_text())
    return PilotSession(root, output, settings, annotation_config.roles)


def test_full_teacher_phase_fails_when_one_image_is_invalid(
    dataset_config, annotation_config, monkeypatch
):
    from annotation.teachers import FakeTeacher

    pilot = setup_pilot(dataset_config, annotation_config, monkeypatch)
    pilot.progress["ocr"] = "ok"
    bad = pilot.inputs[3][0].image_id

    class Worker:
        def extract(self, request):
            if request.image.image_id == bad:
                raise ValueError("malformed response")
            return FakeTeacher().extract(request)

    monkeypatch.setattr("training.pilot.HubTeacher", lambda *args: Worker())
    assert not pilot.step("teacher_a", lambda: pilot.teacher("teacher_a"))
    assert pilot.progress["teacher_a"] == "failed"
    assert json.loads((pilot.output / "teacher_a_report.json").read_text())["valid"] == 11


def test_annotation_quality_failure_preserves_decisions_and_blocks_training(
    dataset_config, annotation_config, monkeypatch
):
    pilot = setup_pilot(dataset_config, annotation_config, monkeypatch)
    pilot.progress.update({"ocr": "ok", "teacher_a": "ok", "teacher_b": "ok"})

    def annotate(image, meta):
        return {**meta, "record": None, "fields": {name: {"status": "unknown"} for name in FIELDS}}

    monkeypatch.setattr(pilot, "engine", lambda **kwargs: SimpleNamespace(annotate=annotate))
    monkeypatch.setattr(pilot, "resume", lambda: None)
    assert not pilot.step("annotate", pilot.annotate)
    report = json.loads((pilot.output / "annotation_coverage.json").read_text())
    assert report["processed"] == 12 and not report["passed"]
    with pytest.raises(RuntimeError, match="earlier stage"):
        pilot.baseline()


def test_recovery_notebook_bootstrap_contains_current_helpers(tmp_path):
    from pathlib import Path

    notebook = json.loads(Path("notebooks/recover_annotations.ipynb").read_bytes())
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "SOURCE_B64" in "".join(c["source"])
    )
    project = tmp_path / "project"
    exec(compile(source, "<recovery-bootstrap>", "exec"), {"PROJECT": project})
    for name in ("training/structured.py", "training/recovery.py", "annotation/orchestrator.py"):
        assert (project / name).read_bytes() == Path(name).read_bytes()


def test_canonical_present_claim_without_evidence_is_not_valid():
    from training.structured import parse_fields

    fields, invalid = parse_fields(
        '{"title":{"status":"present","value":"Deep Work","reason":"visible"}}', ["title"]
    )
    assert "title" in invalid and fields["title"].status == "unknown"


@pytest.mark.parametrize("oversized", [True, False])
def test_input_budget_precedes_gpu_transfer_and_oom_code_survives(monkeypatch, oversized):
    import sys

    from annotation.contracts import RoleSpec
    from annotation.teachers import ModelCallError
    from training.model_io import VisionSession

    class OOM(Exception):
        pass

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM))
    )

    class Batch(dict):
        def to(self, **kwargs):
            if oversized:
                pytest.fail("Oversized input must not transfer tensors to GPU")
            raise OOM("test failure")

    batch = Batch(input_ids=SimpleNamespace(shape=(1, 5000 if oversized else 10)))
    monkeypatch.setattr("training.model_io.make_batch", lambda *args: batch)
    spec = RoleSpec(
        model_id="test/model",
        revision="a" * 40,
        family="test",
        parameters={"max_prompt_tokens": 3072},
    )
    session = VisionSession(spec)
    session.load = lambda: session
    session.model = SimpleNamespace(device="test")
    with pytest.raises(ModelCallError) as caught:
        session.generate(b"x", "prompt")
    assert caught.value.code == ("input_token_budget" if oversized else "gpu_oom")
