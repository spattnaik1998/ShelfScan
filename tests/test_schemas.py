import json

import pytest
from pydantic import ValidationError

from shelfscan.schemas import BookRecord, GuardReport


def record() -> dict:
    return {
        "title": "Example",
        "authors": ["A. Writer"],
        "publisher": None,
        "edition": None,
        "year": None,
        "category": "fiction",
        "price": None,
        "other_text": None,
    }


def test_all_eight_keys_required_even_nullable_fields():
    for key in record():
        data = record()
        del data[key]
        with pytest.raises(ValidationError):
            BookRecord.model_validate(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("year", "2020"),
        ("year", True),
        ("year", 1700),
        ("year", 2101),
        ("category", "unknown"),
        ("authors", "A. Writer"),
        ("title", "   "),
        ("title", "x" * 201),
        ("other_text", "x" * 301),
        ("price", 9.99),
    ],
)
def test_invalid_types_and_values_rejected(field, value):
    data = record()
    data[field] = value
    with pytest.raises(ValidationError):
        BookRecord.model_validate_json(json.dumps(data))


def test_explicit_absence_and_extra_key_contract():
    data = record()
    data.update(authors=[], extra_untrusted="ignore")
    parsed = BookRecord.model_validate_json(json.dumps(data))
    assert parsed.authors == []
    assert "extra_untrusted" not in parsed.model_dump()
    assert len(parsed.model_dump()) == 8


def test_guard_reason_lists_not_shared():
    first, second = GuardReport(status="clean"), GuardReport(status="clean")
    first.reasons.append("test")
    assert second.reasons == []
