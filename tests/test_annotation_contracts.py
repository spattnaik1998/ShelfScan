import pytest
from pydantic import ValidationError

from annotation.contracts import Box, Candidate, FieldClaim
from annotation.teachers import fake_candidate


def test_unknown_and_absent_are_distinct_claims():
    unknown = FieldClaim(status="unknown", value=None, reason="Unreadable")
    absent = FieldClaim(status="absent", value=None, reason="Not printed")
    assert unknown != absent
    with pytest.raises(ValidationError):
        FieldClaim(status="unknown", value="guessed", reason="Unreadable")
    with pytest.raises(ValidationError):
        FieldClaim(status="present", value=None, reason="No text")


@pytest.mark.parametrize(
    "box",
    [
        {"x0": 0.8, "x1": 0.2, "y0": 0.0, "y1": 1.0},
        {"x0": -0.1, "x1": 0.2, "y0": 0.0, "y1": 1.0},
    ],
)
def test_evidence_coordinates_must_be_inside_image(box):
    with pytest.raises(ValidationError):
        Box(**box)


def test_candidates_forbid_category_and_require_all_visible_fields(visual):
    data = fake_candidate(visual).model_dump()
    data["fields"]["category"] = {"status": "present", "value": "fiction", "reason": "Guess"}
    with pytest.raises(ValidationError):
        Candidate.model_validate(data)
    del data["fields"]["category"]
    del data["fields"]["publisher"]
    with pytest.raises(ValidationError):
        Candidate.model_validate(data)


@pytest.mark.parametrize("field,value", [("year", "2020"), ("authors", "Writer"), ("price", 50)])
def test_field_specific_types_validated(visual, field, value):
    data = fake_candidate(visual).model_dump()
    data["fields"][field] = {"status": "present", "value": value, "reason": "Visible"}
    with pytest.raises(ValidationError):
        Candidate.model_validate(data)
