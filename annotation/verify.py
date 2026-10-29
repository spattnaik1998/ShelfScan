"""Deterministic agreement and quote/OCR checks; consensus is not truth."""

import re
import unicodedata
from difflib import SequenceMatcher

from annotation.contracts import Candidate, FieldClaim, FieldDecision, OCRResult


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def equivalent(name: str, left, right, threshold: float) -> bool:
    if name == "authors":
        return len(left) == len(right) and all(
            equivalent("title", a, b, threshold) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, str) and isinstance(right, str):
        a, b = normalize(left), normalize(right)
        if name == "title":
            return 100 * SequenceMatcher(None, a, b, autojunk=False).ratio() >= threshold
        return a == b
    return type(left) is type(right) and left == right


def contains(text: str, value: str) -> bool:
    # Prevent e.g. year 2019 being supported by the text 20190.
    return bool(re.search(r"(?<!\w)" + re.escape(normalize(value)) + r"(?!\w)", normalize(text)))


def evidence_support(claim: FieldClaim, ocr: OCRResult, min_confidence: float) -> tuple[bool, bool]:
    values = claim.value if isinstance(claim.value, list) else [str(claim.value)]
    spans = [span.text for span in ocr.spans if span.confidence >= min_confidence]
    text = " ".join(spans)
    quoted = all(any(contains(e.quote, value) for e in claim.evidence) for value in values)
    supported = quoted and all(
        any(contains(e.quote, value) and contains(text, e.quote) for e in claim.evidence)
        for value in values
    )
    return quoted, supported


def unresolved(reason: str) -> FieldDecision:
    return FieldDecision(status="unknown", value=None, tier="unresolved", reasons=[reason])


def verify(
    a: Candidate,
    b: Candidate,
    ocr: OCRResult,
    *,
    threshold: float = 95.0,
    min_confidence: float = 0.5,
) -> dict[str, FieldDecision]:
    decisions = {}
    for name, left in a.fields.items():
        right = b.fields[name]
        if left.status == right.status == "absent" and name != "title":
            decisions[name] = FieldDecision(
                status="absent",
                value=None,
                tier="consensus_absence",
                reasons=["Both independent teachers report absence"],
            )
        elif (
            left.status == right.status == "present"
            and equivalent(name, left.value, right.value, threshold)
            and evidence_support(left, ocr, min_confidence)[1]
            and evidence_support(right, ocr, min_confidence)[1]
        ):
            decisions[name] = FieldDecision(
                status="accepted",
                value=left.value,
                tier="ocr_agreement",
                reasons=["Independent agreement with supporting OCR quotes"],
                evidence=left.evidence + right.evidence,
            )
        else:
            decisions[name] = unresolved("Disagreement, unknown value, or missing OCR support")
    return decisions
