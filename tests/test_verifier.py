import pytest

from annotation.adjudicate import apply_resolutions, blind_candidates
from annotation.contracts import Adjudication, Candidate, Evidence, FieldClaim, OCRResult
from annotation.curate import curate
from annotation.teachers import FAKE_BOX, FakeOCR, fake_candidate
from annotation.verify import contains, verify


def test_clear_agreement_and_supported_absence(visual):
    candidate = fake_candidate(visual)
    fields = verify(candidate, candidate, FakeOCR().read(visual))
    assert fields["title"].tier == "ocr_agreement"
    assert fields["publisher"].status == "absent"
    record, reasons = curate(fields, "fiction")
    assert record and record.authors == [] and record.publisher is None
    assert reasons == []


def test_missing_ocr_routes_to_adjudication_not_automatic_acceptance(visual):
    candidate = fake_candidate(visual)
    fields = verify(candidate, candidate, OCRResult(spans=[]))
    assert fields["title"].status == "unknown"
    assert curate(fields, "fiction")[0] is None
    resolved = apply_resolutions(
        fields,
        Adjudication(resolutions={"title": candidate.fields["title"]}),
        OCRResult(spans=[]),
        min_confidence=0.5,
    )
    assert resolved["title"].tier == "adjudicated_image"


def test_unquoted_or_unlocated_adjudication_is_not_evidence(visual):
    candidate = fake_candidate(visual)
    fields = verify(candidate, candidate, OCRResult(spans=[]))
    unsupported = FieldClaim(
        status="present",
        value="Imaginary Book",
        evidence=[Evidence(quote="Imaginary Book")],
        reason="Model says so",
    )
    result = apply_resolutions(
        fields,
        Adjudication(resolutions={"title": unsupported}),
        OCRResult(spans=[]),
        min_confidence=0.5,
    )
    assert result["title"].status == "unknown"


def test_unknown_optional_field_excludes_whole_example(visual):
    a = fake_candidate(visual)
    data = a.model_dump()
    data["fields"]["publisher"] = FieldClaim(
        status="unknown",
        value=None,
        reason="Unreadable publisher",
    ).model_dump()
    fields = verify(a, Candidate.model_validate(data), FakeOCR().read(visual))
    assert fields["publisher"].status == "unknown"
    record, reasons = curate(fields, "fiction")
    assert record is None and reasons == ["Unresolved field: publisher"]


def test_adjudicator_cannot_overwrite_undisputed_field(visual):
    candidate = fake_candidate(visual)
    fields = verify(candidate, candidate, FakeOCR().read(visual))
    with pytest.raises(ValueError, match="overwrite"):
        apply_resolutions(
            fields,
            Adjudication(
                resolutions={
                    "price": FieldClaim(
                        status="present",
                        value="FREE",
                        evidence=[Evidence(quote="FREE", box=FAKE_BOX)],
                        reason="Sticker",
                    )
                }
            ),
            FakeOCR().read(visual),
            min_confidence=0.5,
        )


def test_author_order_and_exact_price_are_not_normalized_away(visual):
    from annotation.verify import equivalent

    assert not equivalent("authors", ["Alice", "Bob"], ["Bob", "Alice"], 95.0)
    assert not equivalent("price", "$10", "10", 95.0)
    assert not contains("Printed 20190", "2019")


def test_blind_candidate_order_reproducible_without_role_metadata(visual):
    a, b = fake_candidate(visual), fake_candidate(visual, disagree=True)
    assert blind_candidates(a, b, 42) == blind_candidates(a, b, 42)
    assert all(set(c.model_dump()) == {"fields"} for c in blind_candidates(a, b, 42))
