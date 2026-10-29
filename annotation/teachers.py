"""Replaceable model interface and explicitly synthetic offline demonstration backend."""

from typing import Protocol

from annotation.contracts import (
    FIELDS,
    Adjudication,
    AdjudicationRequest,
    Box,
    Candidate,
    Evidence,
    FieldClaim,
    OCRResult,
    OCRSpan,
    TeacherRequest,
    VisualInput,
)


class ModelCallError(RuntimeError):
    """Sanitized technical failure; adapters must not put secrets in diagnostics."""

    def __init__(self, code: str = "transport_error", *, retryable: bool = True):
        super().__init__(code)
        self.code = code
        self.retryable = retryable


class VisionAnnotator(Protocol):
    def extract(self, request: TeacherRequest) -> Candidate: ...


class Adjudicator(Protocol):
    def resolve(self, request: AdjudicationRequest) -> Adjudication: ...


FAKE_BOX = Box(x0=0.1, y0=0.1, x1=0.9, y1=0.8)


def fake_candidate(image: VisualInput, *, disagree: bool = False) -> Candidate:
    # Intentionally NOT OCR or inferred metadata: never use these labels for training.
    title = "FAKE COVER " + image.image_sha256[:8]
    if disagree:
        title = "DISAGREEMENT " + image.image_sha256[:8]
    fields = {
        name: FieldClaim(status="absent", value=None, reason="Synthetic absent-field fixture")
        for name in FIELDS
    }
    fields["title"] = FieldClaim(
        status="present",
        value=title,
        evidence=[Evidence(quote=title, box=FAKE_BOX)],
        reason="Synthetic title fixture; not read from pixels",
    )
    return Candidate(fields=fields)


class FakeOCR:
    def read(self, image: VisualInput) -> OCRResult:
        title = fake_candidate(image).fields["title"].value
        return OCRResult(spans=[OCRSpan(text=title, confidence=1.0, box=FAKE_BOX)])


class FakeTeacher:
    def __init__(self, *, disagree: bool = False):
        self.disagree = disagree

    def extract(self, request: TeacherRequest) -> Candidate:
        return fake_candidate(request.image, disagree=self.disagree)


class FakeAdjudicator:
    def resolve(self, request: AdjudicationRequest) -> Adjudication:
        claims = fake_candidate(request.image).fields
        return Adjudication(resolutions={name: claims[name] for name in request.disputed_fields})
