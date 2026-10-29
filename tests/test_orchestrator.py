import json

import pytest
import yaml

from annotation.contracts import Adjudication, Candidate, OCRResult, TeacherRequest
from annotation.orchestrator import ANNOTATE_PROMPT, AnnotationEngine, BudgetStop, RoleFailed
from annotation.teachers import (
    FakeAdjudicator,
    FakeOCR,
    FakeTeacher,
    ModelCallError,
    fake_candidate,
)
from scripts.annotate import run, single_writer
from scripts.prepare_images import prepare_dataset
from scripts.split import build_split


def make_engine(config, path, **overrides):
    adapters = {
        "ocr": FakeOCR(),
        "teacher_a": FakeTeacher(),
        "teacher_b": FakeTeacher(disagree=True),
        "adjudicator": FakeAdjudicator(),
    }
    adapters.update(overrides)
    return AnnotationEngine(config, path, **adapters)


METADATA = {"id": "test-001", "category": "fiction", "group_id": "book-001", "split": "train"}


def test_end_to_end_disagreement_is_adjudicated_once_then_cached(
    annotation_config, tmp_path, visual
):
    engine = make_engine(annotation_config, tmp_path)
    first = engine.annotate(visual, METADATA)
    assert first["disposition"] == "accepted"
    assert first["fields"]["title"]["tier"] == "adjudicated_image"
    assert first["training_eligible"] is False and first["is_synthetic"]
    assert engine.new_attempts == 4
    second = make_engine(annotation_config, tmp_path)
    assert second.annotate(visual, METADATA) == first
    assert second.new_attempts == 0 and second.cache_hits == 4


def test_technical_retry_is_bounded_and_persistent(annotation_config, tmp_path, visual):
    calls = []

    def broken(_):
        calls.append(1)
        raise ModelCallError()

    engine = make_engine(annotation_config, tmp_path)
    request = TeacherRequest(image=visual, prompt=ANNOTATE_PROMPT)
    for _ in range(2):
        with pytest.raises(RoleFailed):
            engine.call("teacher_a", visual, request, Candidate, broken)
    assert len(calls) == 2
    resumed = make_engine(annotation_config, tmp_path)
    with pytest.raises(RoleFailed):
        resumed.call("teacher_a", visual, request, Candidate, broken)
    assert len(calls) == 2


def test_transient_error_can_succeed_on_single_retry(annotation_config, tmp_path, visual):
    calls = []

    def intermittent(request):
        calls.append(1)
        if len(calls) == 1:
            raise ModelCallError()
        return fake_candidate(request.image)

    engine = make_engine(annotation_config, tmp_path)
    request = TeacherRequest(image=visual, prompt=ANNOTATE_PROMPT)
    assert engine.call("teacher_a", visual, request, Candidate, intermittent)
    assert len(calls) == 2


def test_malformed_output_is_not_retried_forever(annotation_config, tmp_path, visual):
    engine = make_engine(annotation_config, tmp_path)
    request = TeacherRequest(image=visual, prompt=ANNOTATE_PROMPT)
    with pytest.raises(RoleFailed):
        engine.call("teacher_a", visual, request, Candidate, lambda _: {"category": "fiction"})
    assert engine.new_attempts == 1


def test_inflight_interruption_consumes_attempt_on_resume(annotation_config, tmp_path, visual):
    engine = make_engine(annotation_config, tmp_path)
    request = TeacherRequest(image=visual, prompt=ANNOTATE_PROMPT)

    def interrupt(_):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        engine.call("teacher_a", visual, request, Candidate, interrupt)
    resumed = make_engine(annotation_config, tmp_path)
    result = resumed.call("teacher_a", visual, request, Candidate, lambda _: fake_candidate(visual))
    assert result and resumed.cache.total_attempts() == 2


def test_wall_budget_checked_before_worker_call(annotation_config, tmp_path, visual):
    engine = make_engine(annotation_config, tmp_path)
    engine.started -= 1000
    with pytest.raises(BudgetStop):
        engine.call("ocr", visual, visual, OCRResult, FakeOCR().read)
    assert engine.new_attempts == 0


def test_teachers_only_receive_blind_requests(annotation_config, tmp_path, visual):
    seen = []

    class SpyTeacher:
        def extract(self, request):
            seen.append(request)
            return fake_candidate(request.image)

    engine = make_engine(
        annotation_config, tmp_path, teacher_a=SpyTeacher(), teacher_b=SpyTeacher()
    )
    engine.annotate(visual, METADATA)
    assert len(seen) == 2 and seen[0] == seen[1]
    assert set(seen[0].model_dump()) == {"image", "prompt"}
    assert set(seen[0].image.model_dump()) == {
        "image_id",
        "image_sha256",
        "image_bytes",
        "width",
        "height",
    }
    assert "book-001" not in seen[0].prompt


def test_abstention_quarantines_instead_of_defaulting_null(annotation_config, tmp_path, visual):
    class Abstain:
        def resolve(self, request):
            return Adjudication(resolutions={})

    engine = make_engine(annotation_config, tmp_path, adjudicator=Abstain())
    decision = engine.annotate(visual, METADATA)
    assert decision["disposition"] == "quarantined" and decision["record"] is None
    assert decision["fields"]["title"]["status"] == "unknown"
    assert decision["supervision_mask"]["title"] is False


def test_total_budget_stop_and_restart_resume_without_duplicate_calls(
    dataset_config, annotation_config
):
    prepare_dataset(dataset_config)
    build_split(dataset_config)
    config_path = dataset_config.with_name("annotation.yaml")
    data = annotation_config.model_dump(mode="json")
    data.update(input_path="artifacts/pilot_inputs.zip", max_total_attempts=1)
    config_path.write_text(yaml.safe_dump(data))
    stopped, directory = run(config_path, backend="fake")
    assert stopped["status"] == "budget_stopped"
    assert stopped["total_attempts"] == 1
    assert stopped["simulated_dispositions"]["pending"] == 12
    data["max_total_attempts"] = 100
    config_path.write_text(yaml.safe_dump(data))
    resumed, resumed_dir = run(config_path, backend="fake")
    assert resumed_dir == directory and resumed["status"] == "complete"
    assert resumed["cache_hits"] >= 1
    assert resumed["training_eligible"] == 0
    assert (directory / "labels.jsonl").read_text() == ""
    repeated, _ = run(config_path, backend="fake")
    assert repeated["new_attempts"] == 0
    decisions = json.loads((directory / "decisions.json").read_bytes())
    assert len({d["id"] for d in decisions}) == 12
    assert all(d["training_eligible"] is False for d in decisions)
    assert "FAKE" in (directory / "annotation_report.md").read_text()


def test_single_writer_rejects_competing_runner(tmp_path):
    with single_writer(tmp_path):
        with pytest.raises(RuntimeError, match="locked"), single_writer(tmp_path):
            pytest.fail("Second writer acquired lock")
    assert not (tmp_path / ".writer.lock").exists()
