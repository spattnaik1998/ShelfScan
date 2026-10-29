"""MVP curation excludes every partial example; unknown is never supervised as null."""

from annotation.contracts import FieldDecision
from shelfscan.schemas import BookRecord, Category


def curate(
    fields: dict[str, FieldDecision], category: Category
) -> tuple[BookRecord | None, list[str]]:
    unresolved = sorted(name for name, item in fields.items() if item.status == "unknown")
    if unresolved:
        return None, [f"Unresolved field: {name}" for name in unresolved]
    if fields["title"].status != "accepted":
        return None, ["Title is required"]
    values = {name: item.value for name, item in fields.items()}
    if fields["authors"].status == "absent":
        values["authors"] = []
    return BookRecord.model_validate({**values, "category": category}), []
