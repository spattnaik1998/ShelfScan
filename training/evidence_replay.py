"""Replay saved independent value proposals; attach OCR evidence with full provenance."""

import json
from collections import Counter

from annotation.cache import RoleCache, atomic_json
from annotation.contracts import FIELDS, Candidate, FieldDecision, OCRResult
from annotation.curate import curate
from annotation.evidence import abstain, ground, proposal, value_key
from annotation.verify import unresolved
from scripts.common import sha256
from training.structured import wire_fields


def variants(name, raw):
    if isinstance(raw, list) and raw and all(isinstance(v, dict) for v in raw):
        return [proposal(name, v) for v in raw]
    return [proposal(name, raw)]


def decoded(text, expected):
    try:
        return wire_fields(text, expected)
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return {}


def load_proposals(source, inputs, specs):
    images = {image.image_id: image for image, meta in inputs}
    values = {im: {role: {} for role in ("teacher_a", "teacher_b", "adjudicator")} for im in images}
    origins = {im: {role: {} for role in values[im]} for im in images}
    ocr = {}
    authoritative_unknowns = set()

    def assign(im, role, name, raw, origin):
        if name not in FIELDS:
            return
        values[im][role][name] = variants(name, raw)
        origins[im][role][name] = origin

    # Preserve typed successes; recover format-induced abstentions from raw proposals.
    cache = RoleCache(source / "annotation/raw")
    for path in sorted(cache.root.glob("*.json")):
        state = cache.load(path.stem)
        p = state["provenance"]
        im = p["image_id"]
        role = p["role"]
        if im not in images:
            continue
        if p["image_sha256"] != images[im].image_sha256:
            raise ValueError("Replay image mismatch")
        if role == "ocr" and state["result"] is not None:
            ocr[im] = OCRResult.model_validate(state["result"])
        elif role in ("teacher_a", "teacher_b"):
            current = specs[role]
            if (p["model"]["model_id"], p["model"]["revision"]) != (
                current.model_id,
                current.revision,
            ):
                raise ValueError("Replay model mismatch")
            if state["result"] is not None:
                candidate = Candidate.model_validate(state["result"])
                for name, claim in candidate.fields.items():
                    if claim.status == "unknown" and not claim.reason.startswith(
                        "Missing or invalid"
                    ):
                        authoritative_unknowns.add((im, role, name))
                    assign(
                        im,
                        role,
                        name,
                        claim.model_dump(mode="json"),
                        {"kind": "typed_cache", "sha256": sha256(path.read_bytes())},
                    )
    if set(ocr) != set(images):
        raise ValueError("Every replay image requires verified OCR")
    # Source seed proposals precede the correction attempts. A valid newer typed claim wins below.
    typed = {im: {r: dict(values[im][r]) for r in values[im]} for im in values}
    typed_origins = {im: {r: dict(origins[im][r]) for r in origins[im]} for im in origins}
    seeds = source / "recovery_seed_responses.json"
    if seeds.exists():
        for role, records in json.loads(seeds.read_text()).items():
            if role not in ("teacher_a", "teacher_b"):
                continue
            for im, text in records.items():
                if im not in images:
                    continue
                for name, raw in decoded(text, list(FIELDS)).items():
                    assign(
                        im,
                        role,
                        name,
                        raw,
                        {"kind": "seed_value", "response_sha256": sha256(text.encode())},
                    )
    for role in ("teacher_a", "teacher_b", "adjudicator"):
        path = source / f"{role}_structured_ledger.json"
        if not path.exists():
            continue
        for state in json.loads(path.read_text()).values():
            im = state["image_id"]
            if im not in images:
                continue
            model = state["identity"]["model"]
            if (model["model_id"], model["revision"]) != (
                specs[role].model_id,
                specs[role].revision,
            ):
                raise ValueError("Structured ledger model mismatch")
            if state["identity"]["image_sha256"] != images[im].image_sha256:
                raise ValueError("Structured ledger image mismatch")
            for attempt_index, attempt in enumerate(state["attempts"]):
                text = attempt.get("response", "")
                expected = attempt.get("requested_fields", list(FIELDS))
                for name, raw in decoded(text, expected).items():
                    assign(
                        im,
                        role,
                        name,
                        raw,
                        {
                            "kind": "saved_response",
                            "attempt": attempt_index + 1,
                            "response_sha256": sha256(text.encode()),
                        },
                    )
    for im in values:
        for role in ("teacher_a", "teacher_b"):
            for name, claims in typed[im][role].items():
                # Technical masking is not a semantic retraction of the original value proposal.
                original = next((s for s in claims), abstain())
                # Typed unknowns are only authoritative when not caused by a format failure.
                # Re-open typed source reason (proposal intentionally does not preserve its reason).
                if original.status != "unknown" or (im, role, name) in authoritative_unknowns:
                    values[im][role][name] = claims
                    origins[im][role][name] = typed_origins[im][role][name]
    return values, origins, ocr


def replay(source, inputs, specs, output, minimum=0.5, selections=None):
    values, origins, ocr = load_proposals(source, inputs, specs)
    decisions = []
    choices = {}
    audit = []
    selections = selections or {}
    for image, meta in inputs:
        im = image.image_id
        fields = {}
        image_choices = {}
        for name in FIELDS:
            grounded = {}
            for role in values[im]:
                supported = {}
                for claim in values[im][role].get(name, []):
                    match = ground(name, claim, ocr[im], minimum)
                    if match["supported"]:
                        supported[value_key(claim)] = match
                grounded[role] = supported
            a = grounded["teacher_a"]
            b = grounded["teacher_b"]
            judge = grounded["adjudicator"]
            common = set(a) & set(b)
            claim = None
            tier = None
            why = None
            provenance = None
            if len(common) == 1:
                key = next(iter(common))
                match = a[key]
                claim = match["claim"]
                provenance = match["provenance"]
                tier = "ocr_agreement"
                why = "Independent teacher values agree; evidence attached from OCR"
            else:
                absent = all(
                    any(c.status == "absent" for c in values[im][role].get(name, []))
                    for role in ("teacher_a", "teacher_b")
                )
                if absent and name != "title":
                    fields[name] = FieldDecision(
                        status="absent",
                        value=None,
                        tier="consensus_absence",
                        reasons=[
                            "Both teachers explicitly declared absence; unknown is not absence"
                        ],
                    )
                    continue
                corroborated = set(judge) & (set(a) | set(b))
                if len(judge) == 1 and len(corroborated) == 1:
                    key = next(iter(corroborated))
                    match = judge[key]
                    claim = match["claim"]
                    provenance = match["provenance"]
                    tier = "adjudicated_image"
                    why = "Saved adjudicator value corroborates a teacher and exact OCR region"
            pool = {}
            for role, supported in grounded.items():
                for key, match in supported.items():
                    item = pool.setdefault(
                        key,
                        {
                            "claim": match["claim"].model_dump(mode="json"),
                            "ocr_provenance": match["provenance"],
                            "roles": [],
                        },
                    )
                    item["roles"].append(role)
            options = {f"C{i}": value for i, value in enumerate(pool.values())}
            image_choices[name] = options
            selected = selections.get(im, {}).get(name)
            if claim is None and selected in options:
                option = options[selected]
                claim = proposal(name, option["claim"])
                match = ground(name, claim, ocr[im], minimum)
                if match["supported"]:
                    claim = match["claim"]
                    provenance = match["provenance"]
                    tier = "adjudicated_image"
                    why = "Image selector chose an OCR-backed candidate value"
            if claim is None:
                fields[name] = unresolved(
                    "No unambiguous corroborated OCR-backed value; masked from training"
                )
            else:
                fields[name] = FieldDecision(
                    status="accepted",
                    value=claim.value,
                    tier=tier,
                    reasons=[why, "Pseudo-label evidence; not human ground truth"],
                    evidence=claim.evidence,
                )
                audit.append(
                    {
                        "image_id": im,
                        "field": name,
                        "tier": tier,
                        "ocr": provenance,
                        "origins": {role: origins[im][role].get(name) for role in origins[im]},
                    }
                )
        choices[im] = image_choices
        record, reasons = curate(fields, meta["category"])
        decisions.append(
            {
                **meta,
                "image_id": im,
                "record": record.model_dump(mode="json") if record else None,
                "fields": {name: field.model_dump(mode="json") for name, field in fields.items()},
                "partial_eligible": fields["title"].status == "accepted",
                "reasons": reasons,
            }
        )
    atomic_json(output / "evidence_decisions.json", decisions)
    atomic_json(output / "evidence_options.json", choices)
    atomic_json(output / "evidence_provenance.json", audit)
    eligible = [d for d in decisions if d["partial_eligible"]]
    report = {
        "policy": "ocr-values-v3",
        "input_count": len(decisions),
        "accepted_titles": len(eligible),
        "eligible_groups": len({d["group_id"] for d in eligible}),
        "complete_records": sum(d["record"] is not None for d in decisions),
        "field_statuses": {
            name: dict(Counter(d["fields"][name]["status"] for d in decisions)) for name in FIELDS
        },
        "new_generation_calls": 0,
        "note": "Offline proposal replay, not human accuracy or a completed GPU gate",
    }
    atomic_json(output / "evidence_preview.json", report)
    return report
