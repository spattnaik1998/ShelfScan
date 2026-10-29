"""Inventory originals without modification and create versioned, portable JPEG copies."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import __version__ as pillow_version

from scripts.common import json_bytes, load_config, sha256, write_json
from shelfscan.images import ImageInputError, prepare_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def prepare_dataset(config_path: str | Path) -> dict:
    config, root = load_config(config_path)
    source = (root / config.source_root).resolve()
    output = root / config.output_root
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        "long_edge": config.long_edge,
        "quality": config.jpeg_quality,
        "max_bytes": config.max_upload_mb * 1024 * 1024,
        "max_pixels": config.max_pixels,
    }
    config_hash = sha256(json_bytes(settings))
    rows = []
    ignored = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            ignored.append(relative)
            continue
        # Refuse links outside the named dataset rather than reading unrelated files.
        if not path.resolve().is_relative_to(source):
            raise ValueError(f"Source image link escapes dataset: {relative}")
        raw = path.read_bytes()
        source_hash = sha256(raw)
        image_id = sha256(relative.encode() + b"\0" + source_hash.encode())[:24]
        row = {
            "id": image_id,
            "file": relative,
            "source_sha256": source_hash,
            "source_bytes": len(raw),
            "category": None,
            "status": "error",
            "preprocessing_sha256": config_hash,
        }
        folder = path.relative_to(source).parts[0]
        if folder not in config.category_map:
            row.update(error="unknown_category", message="Image not in a configured category")
        else:
            row["category"] = config.category_map[folder]
            try:
                image = prepare_image(raw, **settings)
                # Content/settings ID prevents a rerun from mutating an earlier prepared artifact.
                name = f"{image_id}-{sha256(image.jpeg)[:12]}.jpg"
                prepared_path = output / "images_1024" / name
                prepared_path.parent.mkdir(parents=True, exist_ok=True)
                prepared_path.write_bytes(image.jpeg)
                row.update(
                    status="ok",
                    prepared_file=f"images_1024/{name}",
                    prepared_sha256=sha256(image.jpeg),
                    prepared_bytes=len(image.jpeg),
                    source_width=image.source_width,
                    source_height=image.source_height,
                    exif_orientation=image.exif_orientation,
                    width=image.width,
                    height=image.height,
                    pixel_sha256=image.pixel_sha256,
                    dhash=image.dhash,
                )
            except ImageInputError as exc:
                row.update(error=exc.code, message=str(exc))
        # Detect concurrent edits to the originals before publishing the inventory.
        if sha256(path.read_bytes()) != source_hash:
            raise ValueError(f"Original changed during preparation: {relative}")
        rows.append(row)
    if not rows:
        raise ValueError("No input images found")
    lines = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    (output / "inventory.jsonl").write_text(lines, encoding="utf-8")
    counts = Counter(row["category"] for row in rows if row["status"] == "ok")
    report = {
        "input_count": len(rows),
        "expected_count": config.expected_images,
        "count_matches_expected": len(rows) == config.expected_images,
        "prepared_count": sum(r["status"] == "ok" for r in rows),
        "failed_count": sum(r["status"] != "ok" for r in rows),
        "category_counts": dict(sorted(counts.items())),
        "ignored_files": ignored,
        "original_bytes": sum(r["source_bytes"] for r in rows),
        "prepared_bytes": sum(r.get("prepared_bytes", 0) for r in rows),
        "pillow_version": pillow_version,
        "settings": settings,
        "inventory_sha256": sha256((output / "inventory.jsonl").read_bytes()),
        "originals_verified_unchanged": True,
    }
    write_json(output / "preparation_report.json", report)
    table = "\n".join(f"| {category} | {counts[category]} |" for category in sorted(counts))
    failures = (
        "\n".join(f"- {r['file']}: {r['error']}" for r in rows if r["status"] != "ok") or "None."
    )
    (output / "preparation_report.md").write_text(
        "# Image preparation report\n\n"
        f"Inputs: {len(rows)} (expected {config.expected_images}); "
        f"prepared: {report['prepared_count']}; failed: {report['failed_count']}.\n\n"
        f"Originals: {report['original_bytes'] / 1024**2:.2f} MiB; "
        f"prepared copies: {report['prepared_bytes'] / 1024**2:.2f} MiB.\n\n"
        "Original hashes rechecked unchanged. Outputs omit original EXIF metadata.\n\n"
        f"| Category | Prepared |\n|---|---:|\n{table}\n\n"
        f"## Decode/category failures\n\n{failures}\n\n"
        "Split and book grouping remain provisional until OCR/title checks in T04.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    report = prepare_dataset(args.config)
    print(json.dumps(report, indent=2))
    if not report["count_matches_expected"] or report["failed_count"]:
        raise SystemExit("Preparation requires attention; inspect data/preparation_report.md")


if __name__ == "__main__":
    main()
