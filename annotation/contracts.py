"""Separate unknown/absent claims from the fully supervised inference record."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FIELDS = ("title", "authors", "publisher", "edition", "year", "price", "other_text")
Role = Literal["ocr", "teacher_a", "teacher_b", "adjudicator"]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class Box(StrictModel):
    """Normalized coordinates in the prepared (EXIF-corrected) image frame."""

    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError("Evidence box must have positive area")
        return self


class Evidence(StrictModel):
    quote: str = Field(min_length=1, max_length=1000)
    box: Box | None = None


class FieldClaim(StrictModel):
    status: Literal["present", "absent", "unknown"]
    value: str | list[str] | int | None
    evidence: list[Evidence] = Field(default_factory=list, max_length=20)
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def consistent(self):
        if self.status == "present" and self.value in (None, "", []):
            raise ValueError("Present claims need a nonempty value")
        if self.status != "present" and self.value is not None:
            raise ValueError("Absent/unknown claims must carry null, never guessed text")
        return self


class Candidate(StrictModel):
    # Category is deliberately not an annotator output or input.
    fields: dict[str, FieldClaim]

    @model_validator(mode="after")
    def complete_and_typed(self):
        if set(self.fields) != set(FIELDS):
            raise ValueError("Candidate requires exactly the seven visible metadata fields")
        for name, claim in self.fields.items():
            if claim.status != "present":
                continue
            value = claim.value
            if name == "authors":
                if not isinstance(value, list) or not 1 <= len(value) <= 10:
                    raise ValueError("Authors must be an ordered nonempty list of <=10 names")
                if any(not author.strip() for author in value):
                    raise ValueError("Author names must be nonempty")
            elif name == "year":
                if type(value) is not int or not 1800 <= value <= 2100:
                    raise ValueError("Year must be an integer from 1800 through 2100")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
            elif name in ("title", "other_text") and len(value) > (200 if name == "title" else 300):
                raise ValueError(f"{name} exceeds inference schema limits")
        return self


class OCRSpan(StrictModel):
    text: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0, le=1)
    box: Box
    rotation: Literal[0, 90, 180, 270] = 0
    crop_id: str = "full"


class OCRResult(StrictModel):
    spans: list[OCRSpan]


class Adjudication(StrictModel):
    resolutions: dict[str, FieldClaim]


class VisualInput(StrictModel):
    """Only these pixels/identifiers reach role adapters, never source paths/categories."""

    image_id: str
    image_sha256: str
    image_bytes: bytes
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class TeacherRequest(StrictModel):
    image: VisualInput
    prompt: str


class AdjudicationRequest(StrictModel):
    image: VisualInput
    prompt: str
    disputed_fields: list[str]
    candidates: list[Candidate]
    ocr: OCRResult
    order_seed: int


class FieldDecision(StrictModel):
    status: Literal["accepted", "absent", "unknown"]
    value: str | list[str] | int | None
    tier: Literal["ocr_agreement", "consensus_absence", "adjudicated_image", "unresolved"]
    reasons: list[str]
    evidence: list[Evidence] = Field(default_factory=list)


class RoleSpec(StrictModel):
    model_id: str
    revision: str
    family: str
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class AnnotationConfig(StrictModel):
    engine_version: str = "annotation-v1"
    backend: Literal["fake"] = "fake"
    input_kind: Literal["pilot", "inventory"] = "pilot"
    input_path: str = "data/pilot_manifest.json"
    image_root: str = "artifacts/pilot_inputs"
    split_path: str = "data/split.provisional.json"
    output_root: str = "data/annotation_fake"
    seed: int = 42
    agreement_threshold: float = Field(default=95.0, ge=0, le=100)
    min_ocr_confidence: float = Field(default=0.5, ge=0, le=1)
    max_total_attempts: int = Field(default=1000, ge=1)
    max_seconds: float = Field(default=300.0, gt=0)
    roles: dict[Role, RoleSpec]

    @model_validator(mode="after")
    def all_roles(self):
        if set(self.roles) != {"ocr", "teacher_a", "teacher_b", "adjudicator"}:
            raise ValueError("Exactly four role specifications required")
        if self.roles["teacher_a"].family == self.roles["teacher_b"].family:
            raise ValueError("Independent teacher families required in this configuration")
        return self
