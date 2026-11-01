import json
from pathlib import Path

import pytest
import yaml

from annotation.cache import RoleCache
from annotation.contracts import RoleSpec
from annotation.teachers import FakeAdjudicator, FakeOCR, FakeTeacher
from scripts.prepare_images import prepare_dataset
from scripts.split import build_split
from training.pilot import PilotSession


@pytest.mark.parametrize("fieldwise", [False, True])
def test_phased_role_passes_and_replay_without_gpu(
    dataset_config, annotation_config, monkeypatch, fieldwise
):
    prepare_dataset(dataset_config)
    build_split(dataset_config)
    root = dataset_config.parent.parent
    (root / "handoff_manifest.json").write_text('{"test_bundle":true}')
    output = root / "reports"
    output.mkdir()
    (output / "runtime_manifest.json").write_text('{"gpu":"stub-only"}')
    settings = yaml.safe_load(Path("configs/colab.example.yaml").read_text())
    specs = {
        **annotation_config.roles,
        "student": RoleSpec(
            model_id="test/student",
            revision="a" * 40,
            family="test-student",
        ),
    }

    specs["teacher_b"] = specs["teacher_b"].model_copy(
        update={"parameters": {"fieldwise": fieldwise}}
    )

    class StubSession:
        def __init__(self, spec):
            self.spec = spec

        def close(self):
            pass

    monkeypatch.setattr("training.pilot.VisionSession", StubSession)
    monkeypatch.setattr(
        "training.pilot.HubTeacher",
        lambda session, diagnostics: FakeTeacher(
            disagree=session.spec.family == "teacher_b",
        ),
    )
    monkeypatch.setattr(
        "training.pilot.FieldwiseTeacher", lambda session, output: FakeTeacher(disagree=True)
    )
    monkeypatch.setattr("training.pilot.HubAdjudicator", lambda _: FakeAdjudicator())
    monkeypatch.setattr("training.pilot.EasyOCRWorker", lambda *args: FakeOCR())
    pilot = PilotSession(root, output, settings, specs)
    assert pilot.step("ocr", pilot.ocr)
    assert pilot.step("teacher_a_check", lambda: pilot.teacher("teacher_a", limit=1))
    assert pilot.step("teacher_b_check", lambda: pilot.teacher("teacher_b", limit=1))
    assert json.loads((output / "teacher_b_check_report.json").read_text())["total"] == 1
    assert pilot.step("teacher_a", lambda: pilot.teacher("teacher_a"))
    assert pilot.step("teacher_b", lambda: pilot.teacher("teacher_b"))
    assert pilot.step("annotate", pilot.annotate)
    ledger = RoleCache(output / "annotation/raw")
    assert ledger.total_attempts() == 48
    # Fake Teacher B deliberately disagrees with OCR; schema validity alone must fail.
    with pytest.raises(RuntimeError, match="benchmark failed"):
        pilot.teacher_b_benchmark()
    monkeypatch.setattr("training.pilot.evidence_support", lambda *args: (True, True))
    pilot.teacher_b_benchmark()
    benchmark = json.loads((output / "teacher_b_benchmark.json").read_text())
    assert benchmark["passed"] and benchmark["ocr_supported_titles"] == 3
    monkeypatch.setattr("training.pilot.evidence_support", lambda *args: (True, False))
    with pytest.raises(RuntimeError, match="benchmark failed"):
        pilot.teacher_b_benchmark()
    assert not json.loads((output / "teacher_b_benchmark.json").read_text())["passed"]
    pilot.resume()
    assert ledger.total_attempts() == 48
    assert json.loads((output / "resume_report.json").read_bytes())["new_attempts"] == 0
    pilot.finish()
    report = json.loads((output / "pilot_report.json").read_bytes())
    assert report["accepted_titles"] == 12
    assert not report["passed"]  # No baseline, training or gateway execution in this stub test.


def test_stage_errors_preserve_redacted_tracebacks(dataset_config, annotation_config, monkeypatch):
    prepare_dataset(dataset_config)
    build_split(dataset_config)
    root = dataset_config.parent.parent
    (root / "handoff_manifest.json").write_text("{}")
    output = root / "reports"
    output.mkdir()
    (output / "runtime_manifest.json").write_text("{}")
    settings = yaml.safe_load(Path("configs/colab.example.yaml").read_text())
    pilot = PilotSession(root, output, settings, annotation_config.roles)

    token = "hf_" + "secret" * 8
    monkeypatch.setenv("HF_TOKEN", token)

    def fail():
        raise ImportError("Missing num2words; token=" + token)

    assert not pilot.step("teacher_a", fail)
    assert token not in (output / "stage_errors.json").read_text()
    assert "Missing num2words" in (output / "stage_errors.json").read_text()
    assert json.loads((output / "progress.json").read_bytes())["teacher_a"] == "failed"
