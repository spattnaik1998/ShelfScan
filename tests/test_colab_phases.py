import json
from pathlib import Path

import yaml

from annotation.cache import RoleCache
from annotation.contracts import RoleSpec
from annotation.teachers import FakeAdjudicator, FakeOCR, FakeTeacher
from scripts.prepare_images import prepare_dataset
from scripts.split import build_split
from training.pilot import PilotSession


def test_phased_role_passes_and_replay_without_gpu(dataset_config, annotation_config, monkeypatch):
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

    class StubSession:
        def __init__(self, spec):
            self.spec = spec

        def close(self):
            pass

    monkeypatch.setattr("training.pilot.VisionSession", StubSession)
    monkeypatch.setattr(
        "training.pilot.HubTeacher",
        lambda session: FakeTeacher(
            disagree=session.spec.family == "teacher_b",
        ),
    )
    monkeypatch.setattr("training.pilot.HubAdjudicator", lambda _: FakeAdjudicator())
    monkeypatch.setattr("training.pilot.EasyOCRWorker", lambda *args: FakeOCR())
    pilot = PilotSession(root, output, settings, specs)
    assert pilot.step("ocr", pilot.ocr)
    assert pilot.step("teacher_a", lambda: pilot.teacher("teacher_a"))
    assert pilot.step("teacher_b", lambda: pilot.teacher("teacher_b"))
    assert pilot.step("annotate", pilot.annotate)
    ledger = RoleCache(output / "annotation/raw")
    assert ledger.total_attempts() == 48
    pilot.resume()
    assert ledger.total_attempts() == 48
    assert json.loads((output / "resume_report.json").read_bytes())["new_attempts"] == 0
    pilot.finish()
    report = json.loads((output / "pilot_report.json").read_bytes())
    assert report["accepted_titles"] == 12
    assert not report["passed"]  # No baseline, training or gateway execution in this stub test.


def test_stage_errors_persist_only_sanitized_types(dataset_config, annotation_config):
    prepare_dataset(dataset_config)
    build_split(dataset_config)
    root = dataset_config.parent.parent
    (root / "handoff_manifest.json").write_text("{}")
    output = root / "reports"
    output.mkdir()
    (output / "runtime_manifest.json").write_text("{}")
    settings = yaml.safe_load(Path("configs/colab.example.yaml").read_text())
    pilot = PilotSession(root, output, settings, annotation_config.roles)

    def fail():
        raise RuntimeError("private credential details")

    assert not pilot.step("teacher_a", fail)
    assert "private credential details" not in (output / "stage_errors.json").read_text()
    assert json.loads((output / "progress.json").read_bytes())["teacher_a"] == "failed"
