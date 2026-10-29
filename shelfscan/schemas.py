"""Strict inference contracts. Annotation unknowns will use a separate contract."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Category = Literal[
    "british_law", "fiction", "history", "indian_bare_act", "indian_law", "technical"
]
CATEGORY_MAP: dict[str, Category] = {
    "British_law_books": "british_law",
    "Fiction": "fiction",
    "History": "history",
    "Indian_bare_acts": "indian_bare_act",
    "Indian_law_books": "indian_law",
    "Technical": "technical",
}
NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Contract(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")


class BookRecord(Contract):
    # Every key is required, even when its explicit value is null/empty.
    title: NonemptyText = Field(max_length=200)
    authors: list[NonemptyText] = Field(max_length=10)
    publisher: str | None
    edition: str | None
    year: int | None = Field(ge=1800, le=2100)
    category: Category
    price: str | None
    other_text: str | None = Field(max_length=300)


class GuardReport(Contract):
    status: Literal["clean", "flagged", "blocked"]
    reasons: list[str] = Field(default_factory=list)


class Attempt(Contract):
    alias: str
    ok: bool
    error: str | None = None
    latency_ms: int = Field(ge=0)


class ScanResult(Contract):
    id: int | None
    record: BookRecord | None
    route: Literal["local", "cloud", "cloud-alt"]
    model_alias: str
    via: Literal["gateway", "direct"]
    attempts: list[Attempt]
    latency_ms: int = Field(ge=0)
    guard: GuardReport
    blurb: str | None = None


class ErrorDetail(Contract):
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class ErrorResponse(Contract):
    error: ErrorDetail
