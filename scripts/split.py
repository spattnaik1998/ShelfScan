"""Provisional visual grouping, stratified group splits and a portable pilot archive."""

import argparse
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.common import json_bytes, load_config, read_inventory, sha256, write_json
from shelfscan.prompt import EXTRACT_PROMPT, PROMPT_SHA256, PROMPT_VERSION


def group_images(rows: list[dict], distance: int) -> tuple[list[dict], list[dict]]:
    """Conservative leakage prevention; near groups are explicitly unverified."""
    valid = sorted((r for r in rows if r["status"] == "ok"), key=lambda r: r["id"])
    parent = {row["id"]: row["id"] for row in valid}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    edges = []
    for left, right in combinations(valid, 2):
        delta = (int(left["dhash"], 16) ^ int(right["dhash"], 16)).bit_count()
        reason = None
        if left["source_sha256"] == right["source_sha256"]:
            reason = "identical_source"
        elif left["pixel_sha256"] == right["pixel_sha256"]:
            reason = "identical_prepared_pixels"
        elif delta <= distance:
            reason = "near_visual_candidate"
        if reason:
            a, b = find(left["id"]), find(right["id"])
            parent[max(a, b)] = min(a, b)
            edges.append(
                {
                    "left": left["id"],
                    "right": right["id"],
                    "reason": reason,
                    "dhash_distance": delta,
                }
            )
    members = defaultdict(list)
    for row in valid:
        members[find(row["id"])].append(row)
    groups = []
    for items in members.values():
        ids = sorted(r["id"] for r in items)
        categories = sorted({r["category"] for r in items})
        groups.append(
            {
                "group_id": "book-" + sha256(json_bytes(ids))[:16],
                "members": ids,
                "categories": categories,
                "category_conflict": len(categories) > 1,
                "verified_book_identity": False,
            }
        )
    return sorted(groups, key=lambda g: g["group_id"]), edges


def assign_splits(
    rows: list[dict],
    groups: list[dict],
    *,
    seed: int,
    pilot_per_category: int,
) -> tuple[dict, dict]:
    by_id = {r["id"]: r for r in rows if r["status"] == "ok"}
    if len(by_id) != sum(len(g["members"]) for g in groups):
        raise ValueError("Group membership does not cover valid inventory")
    rng = random.Random(seed)
    by_category = defaultdict(list)
    for group in groups:
        # Mixed categories cannot be independently stratified: keep the group together in train.
        key = group["categories"][0] if not group["category_conflict"] else "__mixed__"
        by_category[key].append(group)
    assignments = {}
    pilot_ids = []
    exceptions = []
    for category in sorted({row["category"] for row in by_id.values()}):
        candidates = [r for r in by_id.values() if r["category"] == category]
        candidates.sort(key=lambda r: r["id"])
        rng.shuffle(candidates)
        # Prefer distinct visual groups so the pilot is not redundant.
        group_of = {m: g["group_id"] for g in groups for m in g["members"]}
        selected = []
        seen = set()
        for row in candidates:
            if group_of[row["id"]] not in seen:
                selected.append(row["id"])
                seen.add(group_of[row["id"]])
            if len(selected) == pilot_per_category:
                break
        if len(selected) < pilot_per_category:
            extras = [r["id"] for r in candidates if r["id"] not in selected]
            selected.extend(extras[: pilot_per_category - len(selected)])
            exceptions.append(f"{category}: fewer than {pilot_per_category} distinct visual groups")
        if len(selected) < pilot_per_category:
            exceptions.append(f"{category}: only {len(selected)} valid pilot photos")
        pilot_ids.extend(selected)
    pilot_set = set(pilot_ids)
    for category, category_groups in sorted(by_category.items()):
        category_groups = sorted(category_groups, key=lambda g: g["group_id"])
        reserved = [g for g in category_groups if pilot_set.intersection(g["members"])]
        others = [g for g in category_groups if g not in reserved]
        rng.shuffle(others)
        count = len(category_groups)
        if category == "__mixed__":
            train_count, val_count = count, 0
            exceptions.append("Mixed-category visual groups reserved for train pending T04 review")
        elif count >= 3:
            val_count = max(1, round(count * 0.15))
            test_count = max(1, round(count * 0.15))
            train_count = max(len(reserved), count - val_count - test_count)
            val_count = min(val_count, count - train_count)
        else:
            train_count, val_count = count, 0
            exceptions.append(f"{category}: too few groups for three nonempty partitions")
        ordered = reserved + others
        for index, group in enumerate(ordered):
            split = (
                "train"
                if index < train_count
                else ("validation" if index < train_count + val_count else "test")
            )
            assignments[group["group_id"]] = split
    ids_by_split = {name: [] for name in ("train", "validation", "test")}
    for group in groups:
        ids_by_split[assignments[group["group_id"]]].extend(group["members"])
    for ids in ids_by_split.values():
        ids.sort()
    if not pilot_set.issubset(ids_by_split["train"]):
        raise AssertionError("Pilot group leaked outside training")
    split = {
        "status": "provisional",
        "seed": seed,
        "target_ratios": [0.7, 0.15, 0.15],
        "assignments": assignments,
        "groups": groups,
        **ids_by_split,
        "category_counts": {
            name: dict(sorted(Counter(by_id[i]["category"] for i in ids).items()))
            for name, ids in ids_by_split.items()
        },
        "exceptions": exceptions,
    }
    split["split_sha256"] = sha256(json_bytes(split))
    return split, {"ids": sorted(pilot_ids), "exceptions": exceptions}


def write_archive(path: Path, files: dict[str, bytes]) -> None:
    """Stable member ordering/timestamps for byte-reproducible handoffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)


def build_split(config_path: str | Path) -> dict:
    config, root = load_config(config_path)
    output = root / config.output_root
    inventory_path = output / "inventory.jsonl"
    rows = read_inventory(inventory_path)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate inventory IDs")
    # Packaging must not silently accept edited or missing prepared images.
    for row in rows:
        if row["status"] == "ok":
            path = (output / row["prepared_file"]).resolve()
            if not path.is_relative_to(output.resolve()):
                raise ValueError("Prepared image path escapes output directory")
            if sha256(path.read_bytes()) != row["prepared_sha256"]:
                raise ValueError(f"Prepared image hash mismatch: {row['id']}")
    groups, edges = group_images(rows, config.near_duplicate_distance)
    split, pilot = assign_splits(
        rows,
        groups,
        seed=config.seed,
        pilot_per_category=config.pilot_per_category,
    )
    group_of = {m: g["group_id"] for g in groups for m in g["members"]}
    for row in rows:
        row["group_id"] = group_of.get(row["id"])
    # Inventory stays a preparation artifact; grouped membership is a separate immutable view.
    write_json(
        output / "groups.provisional.json",
        {
            "status": "provisional",
            "groups": groups,
            "evidence": edges,
            "near_duplicate_distance": config.near_duplicate_distance,
            "inventory_sha256": sha256(inventory_path.read_bytes()),
            "limitation": (
                "Visual similarity can over/under-group books; refine with OCR/title in T04."
            ),
        },
    )
    split["inventory_sha256"] = sha256(inventory_path.read_bytes())
    # Hash includes every field except the hash itself.
    del split["split_sha256"]
    split["split_sha256"] = sha256(json_bytes(split))
    write_json(output / "split.provisional.json", split)
    by_id = {r["id"]: r for r in rows}
    inputs = []
    files = {}
    for image_id in pilot["ids"]:
        row = by_id[image_id]
        name = f"images/{image_id}.jpg"
        files[name] = (output / row["prepared_file"]).read_bytes()
        inputs.append(
            {
                "id": image_id,
                "file": name,
                "category": row["category"],
                "group_id": row["group_id"],
                "source_sha256": row["source_sha256"],
                "sha256": row["prepared_sha256"],
                "width": row["width"],
                "height": row["height"],
                "split": "train",
            }
        )
    manifest = {
        "status": "provisional",
        "seed": config.seed,
        "inputs": inputs,
        "split_sha256": split["split_sha256"],
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "exceptions": pilot["exceptions"],
        "category_policy": "For curation only; do not pass category or manifest to teachers.",
    }
    write_json(output / "pilot_manifest.json", manifest)
    files["pilot_manifest.json"] = json_bytes(manifest)
    files["extract_prompt.txt"] = EXTRACT_PROMPT.encode()
    files["ATTRIBUTION.txt"] = (
        b"Rex Jovakim Joseph. Book Cover Image Dataset (2026). CC BY 4.0.\n"
        b"https://creativecommons.org/licenses/by/4.0/\n"
        b"Prepared derivatives: EXIF correction, resize and JPEG recompression.\n"
    )
    files["README.txt"] = (
        b"ShelfScan pilot inputs: no labels, no model, no notebook yet.\n"
        b"Verify each image SHA-256 against pilot_manifest.json before use.\n"
        b"All selected source groups are reserved for training. Split is provisional.\n"
        b"Load images using opaque paths; never pass the category-bearing manifest to a teacher.\n"
    )
    files["checksums.json"] = json_bytes(
        {name: sha256(data) for name, data in sorted(files.items())}
    )
    archive_path = root / config.archive_path
    write_archive(archive_path, files)
    summary = {
        "status": "provisional",
        "valid_images": len(by_id) - sum(r["status"] != "ok" for r in rows),
        "groups": len(groups),
        "group_edges": len(edges),
        "split_counts": {s: len(split[s]) for s in ("train", "validation", "test")},
        "category_counts": split["category_counts"],
        "pilot_count": len(inputs),
        "pilot_category_counts": dict(sorted(Counter(i["category"] for i in inputs).items())),
        "archive_sha256": sha256(archive_path.read_bytes()),
        "split_sha256": split["split_sha256"],
        "exceptions": pilot["exceptions"],
    }
    write_json(output / "split_report.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--provisional", action="store_true", required=True)
    args = parser.parse_args()
    import json

    print(json.dumps(build_split(args.config), indent=2))


if __name__ == "__main__":
    main()
