"""Resume an existing pilot after the narrowly scoped numeric-year quote fix."""

import hashlib
import json

from annotation.cache import RoleCache, atomic_json
from training import model_io
from training.runtime import load_json_object


def normalize_year_quote(text):
    data = load_json_object(text)
    fields = data.get("fields", data)
    if not isinstance(fields, dict):
        return text
    year = fields.get("year")
    if isinstance(year, dict) and (
        year.get("status") == "present"
        and type(year.get("quote")) is int
        and type(year.get("value")) is int
        and year["quote"] == year["value"]
        and 1800 <= year["quote"] <= 2100
    ):
        year["quote"] = str(year["quote"])
        return json.dumps(data)
    return text


def recover_year_quote(pilot):
    # Patch the existing module object: old HubTeacher instances resolve this global too.
    previous = model_io.parse_candidate

    def patched(text):
        return previous(normalize_year_quote(text))

    model_io.parse_candidate = patched
    diagnostics = pilot.output / "teacher_b_diagnostics.json"
    records = json.loads(diagnostics.read_text()) if diagnostics.exists() else []
    inputs = {image.image_id: image.image_sha256 for image, _ in pilot.inputs}
    cache = RoleCache(pilot.output / "annotation/raw")
    recovered = []
    for path in cache.root.glob("*.json"):
        state = cache.load(path.stem)  # verifies envelope checksum and provenance
        provenance = state["provenance"]
        if (
            state["result"] is not None
            or provenance["role"] != "teacher_b"
            or provenance["model"] != pilot.specs["teacher_b"].model_dump(mode="json")
            or inputs.get(provenance["image_id"]) != provenance["image_sha256"]
        ):
            continue
        for record in reversed(records):
            if (
                record.get("image_id") != provenance["image_id"]
                or record.get("model_id") != provenance["model"]["model_id"]
            ):
                continue
            text = record.get("response", "")
            try:
                normalized = normalize_year_quote(text)
                if normalized == text:
                    continue
                result = patched(text)
            except (ValueError, TypeError):
                continue
            state["result"] = result.model_dump(mode="json")
            state["parser_revalidation"] = {
                "patch_version": "numeric-year-quote-v1",
                "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "new_generation_calls": 0,
            }
            cache.save(path.stem, state)
            recovered.append(provenance["image_id"])
            break
    report = {
        "patch_version": "numeric-year-quote-v1",
        "recovered_images": recovered,
        "new_generation_calls": 0,
        "note": "Benchmark and evidence gates still required",
    }
    atomic_json(pilot.output / "parser_recovery_report.json", report)
    return report
