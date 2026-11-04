import json
from types import SimpleNamespace

from annotation.contracts import FIELDS, Box, Candidate, OCRResult, OCRSpan, RoleSpec, VisualInput
from training.title_recovery import TitleCorrector


def test_retry_changes_only_title_and_preserves_original_claims(tmp_path, monkeypatch):
    spec = RoleSpec(model_id="test/model", revision="a" * 40, family="idefics3")
    pilot = SimpleNamespace(
        output=tmp_path, specs={"teacher_b": spec}, config=SimpleNamespace(min_ocr_confidence=0.5)
    )
    calls = []

    class Session:
        def __init__(self, spec):
            pass

        def generate(self, *args, **kw):
            calls.append(1)
            return json.dumps(
                {
                    "status": "present",
                    "value": "The Administrative Tribunals Act 1985",
                    "quote": "The Administrative Tribunals Act 1985",
                }
            )

    monkeypatch.setattr("training.title_recovery.model_io.VisionSession", Session)
    fields = {
        name: {"status": "unknown", "value": None, "evidence": [], "reason": "unknown"}
        for name in FIELDS
    }
    fields["title"] = {
        "status": "present",
        "value": "The Administrator's Act 1985",
        "evidence": [{"quote": "The Administrator's Act 1985"}],
        "reason": "visible",
    }
    original = Candidate.model_validate({"fields": fields})
    image = VisualInput(
        image_id="x", image_sha256="b" * 64, image_bytes=b"image", width=1, height=1
    )
    ocr = OCRResult(
        spans=[
            OCRSpan(
                text="The Administrative Tribunals Act, 1985",
                confidence=0.9,
                box=Box(x0=0.1, y0=0.1, x1=0.9, y1=0.2),
            )
        ]
    )
    fixed = TitleCorrector(pilot).repair(image, original, ocr)
    assert fixed.fields["title"].value == "The Administrative Tribunals Act 1985"
    assert original.fields["title"].value == "The Administrator's Act 1985"
    assert fixed.fields["authors"] == original.fields["authors"]
    assert TitleCorrector(pilot).repair(image, original, ocr) == fixed
    assert len(calls) == 1


def test_unsupported_retry_is_bounded_across_reruns(tmp_path, monkeypatch):
    spec = RoleSpec(model_id="test/model", revision="a" * 40, family="idefics3")
    pilot = SimpleNamespace(
        output=tmp_path, specs={"teacher_b": spec}, config=SimpleNamespace(min_ocr_confidence=0.5)
    )
    calls = []

    class Session:
        def __init__(self, spec):
            pass

        def generate(self, *args, **kw):
            calls.append(1)
            return json.dumps({"status": "present", "value": "Wrong title", "quote": "Wrong title"})

    monkeypatch.setattr("training.title_recovery.model_io.VisionSession", Session)
    fields = {
        name: {"status": "unknown", "value": None, "evidence": [], "reason": "unknown"}
        for name in FIELDS
    }
    original = Candidate.model_validate({"fields": fields})
    image = VisualInput(
        image_id="x", image_sha256="b" * 64, image_bytes=b"image", width=1, height=1
    )
    for _ in range(2):
        assert TitleCorrector(pilot).repair(image, original, OCRResult(spans=[])) == original
    assert len(calls) == 2


def test_paused_pilot_repairs_only_unsupported_cached_title(
    dataset_config, annotation_config, monkeypatch
):
    import hashlib
    from pathlib import Path

    import yaml

    import annotation.orchestrator as orchestration
    import training.pilot as runtime
    from annotation.cache import RoleCache
    from annotation.teachers import FakeOCR, FakeTeacher
    from scripts.prepare_images import prepare_dataset
    from scripts.split import build_split
    from training.title_recovery import install_title_recovery

    prepare_dataset(dataset_config)
    build_split(dataset_config)
    root = dataset_config.parent.parent
    (root / "handoff_manifest.json").write_text("{}")
    output = root / "reports"
    output.mkdir()
    (output / "runtime_manifest.json").write_text("{}")
    settings = yaml.safe_load(Path("configs/colab.example.yaml").read_text())
    calls = []

    class Session:
        def __init__(self, spec):
            self.spec = spec

        def close(self):
            pass

        def metrics(self):
            return {"generation_calls": len(calls)}

        def generate(self, raw, *args, **kwargs):
            calls.append(1)
            title = "FAKE COVER " + hashlib.sha256(raw).hexdigest()[:8]
            return json.dumps({"status": "present", "value": title, "quote": title})

    monkeypatch.setattr(runtime, "VisionSession", Session)
    monkeypatch.setattr("training.title_recovery.model_io.VisionSession", Session)
    monkeypatch.setattr(runtime, "HubTeacher", lambda *args: FakeTeacher())
    monkeypatch.setattr(runtime, "EasyOCRWorker", lambda *args: FakeOCR())
    # Register originals so pytest restores globals changed by the installer.
    monkeypatch.setattr(runtime, "evidence_support", runtime.evidence_support)
    monkeypatch.setattr(orchestration, "verify", orchestration.verify)
    pilot = runtime.PilotSession(root, output, settings, annotation_config.roles)
    assert pilot.step("ocr", pilot.ocr)
    pilot.teacher("teacher_b", limit=3)
    cache = RoleCache(output / "annotation/raw")
    for path in cache.root.glob("*.json"):
        state = cache.load(path.stem)
        if state["provenance"]["role"] == "teacher_b":
            state["result"]["fields"]["title"]["value"] = "Wrong title"
            state["result"]["fields"]["title"]["evidence"] = [{"quote": "Wrong title", "box": None}]
            cache.save(path.stem, state)
            break
    install_title_recovery(pilot)
    assert pilot.step("teacher_b_check", pilot.teacher_b_benchmark)
    report = json.loads((output / "teacher_b_benchmark.json").read_text())
    assert report["ocr_supported_titles"] == 3
    assert all("dataset_id" in row for row in report["rows"])
    assert len(calls) == 1
    assert pilot.step("teacher_b_check", pilot.teacher_b_benchmark)
    assert len(calls) == 1
