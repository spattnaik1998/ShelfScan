"""Validated import into a new recovery run; original runs are never rewritten."""

import json
from pathlib import Path
from zipfile import ZipFile


def restore_results(archive, destination):
    destination = Path(destination)
    with ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist()) > 100 * 1024 * 1024:
            raise ValueError("Results archive exceeds recovery size limit")
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("Duplicate results archive members")
        for info in z.infolist():
            path = (destination / info.filename).resolve()
            if not path.is_relative_to(destination.resolve()) or "\\" in info.filename:
                raise ValueError("Unsafe results archive path")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlinks are not allowed")
            if info.file_size > 16 * 1024 * 1024:
                raise ValueError("Oversized recovery member")
            if info.is_dir() or path.suffix not in (".json", ".yaml"):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(info)
            if path.exists() and path.read_bytes() != data:
                raise ValueError("Import target already has different source results")
            path.write_bytes(data)
    if not (destination / "models.resolved.yaml").is_file():
        raise ValueError("Use the full results ZIP, not the small pilot_reports.zip archive")
    if not (destination / "annotation/raw").is_dir():
        raise ValueError("Full results archive is missing role caches")


def pin_source_models(project, source):
    import re

    import yaml

    path = project / "configs/models.example.yaml"
    target = yaml.safe_load(path.read_text())
    old = yaml.safe_load((source / "models.resolved.yaml").read_text())
    for role, spec in target.items():
        prior = old[role]
        if spec["model_id"] != prior["model_id"] or spec["family"] != prior["family"]:
            raise ValueError("Recovery requires the same base checkpoints as the source run")
        if not re.fullmatch("[0-9a-f]{40}", prior["revision"]):
            raise ValueError("Source model revision must be immutable")
        spec["revision"] = prior["revision"]
    path.write_text(yaml.safe_dump(target), encoding="utf-8")


def import_successes(pilot, source):
    import shutil

    import yaml

    from annotation.cache import RoleCache, atomic_json
    from annotation.contracts import Candidate, OCRResult, TeacherRequest
    from annotation.orchestrator import ANNOTATE_PROMPT
    from scripts.common import sha256

    if source.resolve() == pilot.output.resolve():
        raise ValueError("Recovery output must differ from the source run")
    prior_models = yaml.safe_load((source / "models.resolved.yaml").read_text())
    for role in ("teacher_a", "teacher_b", "student", "adjudicator"):
        prior = prior_models[role]
        current = pilot.specs[role]
        if (prior["model_id"], prior["revision"], prior["family"]) != (
            current.model_id,
            current.revision,
            current.family,
        ):
            raise ValueError("Source/target base-model provenance mismatch")
    source_cache = RoleCache(source / "annotation/raw")
    engine = pilot.engine()
    images = {image.image_id: image for image, _ in pilot.inputs}
    imported = {"ocr": 0, "teacher_a": 0, "teacher_b": 0}
    seeds = {"teacher_a": {}, "teacher_b": {}}
    diagnostics = {}
    for role in seeds:
        path = source / f"{role}_diagnostics.json"
        diagnostics[role] = json.loads(path.read_text()) if path.exists() else []
    for path in sorted(source_cache.root.glob("*.json")):
        state = source_cache.load(path.stem)  # Hashes and provenance checked before reuse.
        p = state["provenance"]
        role = p["role"]
        if role not in imported or p["image_id"] not in images:
            continue
        image = images[p["image_id"]]
        current = pilot.config.roles[role]
        old = p["model"]
        if (old["model_id"], old["revision"], old["family"]) != (
            current.model_id,
            current.revision,
            current.family,
        ):
            raise ValueError("Cached model identity differs from target")
        if p["image_sha256"] != image.image_sha256:
            raise ValueError("Cached image digest differs from verified pilot input")
        for key in ("quantize", "image_longest_edge", "rotations"):
            if old["parameters"].get(key) != current.parameters.get(key):
                raise ValueError("Pixel processing, quantization or OCR rotation changed")
        snapshot = pilot.output / "source_snapshot/annotation/raw" / path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            shutil.copyfile(path, snapshot)
        if state["result"] is None:
            if role in seeds:
                for record in reversed(diagnostics[role]):
                    if (
                        record.get("image_id") == image.image_id
                        and record.get("model_id") == current.model_id
                    ):
                        seeds[role][image.image_id] = record["response"]
                        break
            continue
        schema = OCRResult if role == "ocr" else Candidate
        result = schema.model_validate(state["result"])
        request = image if role == "ocr" else TeacherRequest(image=image, prompt=ANNOTATE_PROMPT)
        provenance = engine.provenance(role, image, request, schema)
        key = engine.cache.key(provenance)
        if engine.cache.load(key) is None:
            engine.cache.save(
                key,
                {
                    "provenance": provenance,
                    "attempts": [],
                    "result": result.model_dump(mode="json"),
                    "imported_from": {
                        "source_key": path.stem,
                        "source_sha256": sha256(path.read_bytes()),
                        "source_provenance": p,
                        "transforms": state.get("title_repair"),
                    },
                },
            )
        imported[role] += 1
    seed_path = pilot.output / "recovery_seed_responses.json"
    if not seed_path.exists():
        atomic_json(seed_path, seeds)
    report = {
        "source": str(source),
        "reused": imported,
        "seeded_failures": {role: len(values) for role, values in seeds.items()},
        "policy": "structured-v2",
        "new_generation_calls": 0,
        "input_checksums_sha256": sha256((pilot.output / "input_checksums.json").read_bytes()),
    }
    atomic_json(pilot.output / "source_import_report.json", report)
    if imported["ocr"] == len(pilot.inputs):
        pilot.progress["ocr"] = "ok"
        atomic_json(pilot.output / "progress.json", pilot.progress)
    return report
