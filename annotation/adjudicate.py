"""One selective resolution round with logged, anonymous candidate ordering."""

import random

from annotation.contracts import Adjudication, Candidate, FieldDecision, OCRResult
from annotation.verify import evidence_support, unresolved


def blind_candidates(a: Candidate, b: Candidate, seed: int) -> list[Candidate]:
    candidates = [a, b]
    random.Random(seed).shuffle(candidates)
    return candidates


def apply_resolutions(
    decisions: dict[str, FieldDecision],
    result: Adjudication,
    ocr: OCRResult,
    *,
    min_confidence: float,
) -> dict[str, FieldDecision]:
    disputed = {name for name, decision in decisions.items() if decision.status == "unknown"}
    if not set(result.resolutions).issubset(disputed):
        raise ValueError("Adjudicator tried to overwrite an undisputed or unknown schema field")
    output = dict(decisions)
    for name in sorted(disputed):
        claim = result.resolutions.get(name)
        if claim is None or claim.status == "unknown":
            output[name] = unresolved("Adjudicator abstained")
            continue
        # Validate the resolved field type using the same full-candidate contract.
        from annotation.contracts import FIELDS, FieldClaim

        placeholders = {
            field: FieldClaim(status="unknown", value=None, reason="Not part of adjudication")
            for field in FIELDS
        }
        placeholders[name] = claim
        Candidate(fields=placeholders)
        if claim.status == "absent" and name != "title":
            output[name] = FieldDecision(
                status="absent",
                value=None,
                tier="adjudicated_image",
                reasons=[claim.reason],
            )
        elif claim.status == "present":
            quoted, supported = evidence_support(claim, ocr, min_confidence)
            # A valid source-image box is required for lower-evidence visual-only decisions.
            boxed = bool(claim.evidence) and all(e.box is not None for e in claim.evidence)
            if quoted and (supported or boxed):
                output[name] = FieldDecision(
                    status="accepted",
                    value=claim.value,
                    tier="adjudicated_image",
                    reasons=[claim.reason, "Model adjudication; not independently verified truth"],
                    evidence=claim.evidence,
                )
            else:
                output[name] = unresolved("Adjudication lacks traceable quote/image evidence")
        else:
            output[name] = unresolved("A book title cannot be supervised as absent")
    return output
