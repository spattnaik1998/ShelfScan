import json
from types import SimpleNamespace

import pytest

from annotation.cache import RoleCache
from annotation.contracts import FIELDS, RoleSpec, VisualInput
from training import model_io
from training.parser_recovery import recover_year_quote


@pytest.mark.parametrize("matching_model", [True, False])
def test_recovery_revalidates_saved_response_without_generation(
    tmp_path, monkeypatch, matching_model
):
    monkeypatch.setattr(model_io, "parse_candidate", model_io.parse_candidate)
    spec = RoleSpec(model_id="test/model", revision="a" * 40, family="idefics3")
    image = VisualInput(
        image_id="x", image_sha256="b" * 64, image_bytes=b"pixels", width=1, height=1
    )
    cache = RoleCache(tmp_path / "annotation/raw")
    provenance = {
        "role": "teacher_b",
        "model": spec.model_dump(mode="json"),
        "image_id": "x",
        "image_sha256": image.image_sha256,
    }
    key = cache.key(provenance)
    cache.save(
        key,
        {
            "provenance": provenance,
            "result": None,
            "attempts": [{"status": "error", "retryable": False}],
        },
    )
    fields = {name: {"status": "unknown", "value": None, "quote": None} for name in FIELDS}
    fields["year"] = {"status": "present", "value": 2016, "quote": 2016}
    (tmp_path / "teacher_b_diagnostics.json").write_text(
        json.dumps(
            [
                {
                    "image_id": "x",
                    "model_id": "test/model" if matching_model else "wrong/model",
                    "response": json.dumps({"fields": fields}),
                }
            ]
        )
    )
    pilot = SimpleNamespace(output=tmp_path, specs={"teacher_b": spec}, inputs=[(image, {})])
    report = recover_year_quote(pilot)
    assert report["new_generation_calls"] == 0
    state = cache.load(key)
    assert len(state["attempts"]) == 1
    if matching_model:
        assert state["result"]["fields"]["year"]["evidence"][0]["quote"] == "2016"
        assert recover_year_quote(pilot)["recovered_images"] == []
    else:
        assert state["result"] is None
