import json
from types import SimpleNamespace

import pytest

from annotation.contracts import FIELDS, TeacherRequest, VisualInput
from annotation.teachers import ModelCallError


def request():
    return TeacherRequest(
        image=VisualInput(
            image_id="x", image_sha256="a" * 64, image_bytes=b"pixels", width=1, height=1
        ),
        prompt="",
    )


def session(generate):
    return SimpleNamespace(
        spec=SimpleNamespace(
            model_dump=lambda **kw: {"model_id": "test/model", "revision": "a" * 40}
        ),
        generate=generate,
    )


def test_fieldwise_teacher_retries_bad_claim_then_resumes_without_calls(tmp_path):
    from training.field_teacher import FieldwiseTeacher

    calls = []

    def generate(raw, prompt, **kw):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({"status": "present", "value": "Example", "quote": ""})
        return (
            json.dumps({"status": "present", "value": "Example", "quote": "Example"})
            if len(calls) == 2
            else json.dumps({"status": "unknown", "value": None, "quote": None})
        )

    worker = FieldwiseTeacher(session(generate), tmp_path)
    result = worker.extract(request())
    assert len(calls) == 8
    assert set(result.fields) == set(FIELDS)
    assert result.fields["title"].value == "Example"
    assert "Previous response failed" in calls[1]
    assert FieldwiseTeacher(session(generate), tmp_path).extract(request()) == result
    assert len(calls) == 8


def test_fieldwise_retry_budget_survives_restart_and_never_invents_unknown(tmp_path):
    from training.field_teacher import FieldwiseTeacher

    calls = []

    def generate(*args, **kw):
        calls.append(1)
        return '{"status":"unknown","value":"1958","quote":""}'

    for _ in range(2):
        with pytest.raises(ModelCallError):
            FieldwiseTeacher(session(generate), tmp_path).extract(request())
    assert len(calls) == 2


def test_fieldwise_global_budget_prevents_generation(tmp_path):
    from training.field_teacher import FieldwiseTeacher

    def generate(*args, **kw):
        pytest.fail("Exhausted budget must not call a model")

    with pytest.raises(ModelCallError):
        FieldwiseTeacher(session(generate), tmp_path, max_calls=0).extract(request())


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "present", "value": "Title", "quote": ""},
        {"status": "unknown", "value": "1958", "quote": ""},
        {"status": "unknown", "value": None, "quote": None, "extra": 1},
        {"fields": [{"field": "title", "status": "present", "value": "Title", "quote": ""}]},
    ],
)
def test_single_field_does_not_repair_bad_evidence_or_status(payload):
    from training.field_teacher import validate_field

    with pytest.raises(ValueError):
        validate_field("title", json.dumps(payload))


def test_fieldwise_diagnostics_redact_credentials(tmp_path, monkeypatch):
    from training.field_teacher import FieldwiseTeacher

    token = "hf_" + "secret" * 8
    monkeypatch.setenv("HF_TOKEN", token)

    def generate(*args, **kw):
        return "bad JSON " + token

    with pytest.raises(ModelCallError):
        FieldwiseTeacher(session(generate), tmp_path).extract(request())
    assert token not in (tmp_path / "teacher_b_diagnostics.json").read_text()
    assert token not in (tmp_path / "teacher_b_fields.json").read_text()
