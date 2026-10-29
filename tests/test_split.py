import json
from zipfile import ZipFile

import pytest
import yaml

from scripts.common import json_bytes, read_inventory, sha256
from scripts.prepare_images import prepare_dataset
from scripts.split import assign_splits, build_split, group_images


def test_duplicate_images_never_cross_group_split(dataset_config):
    project = dataset_config.parent.parent
    source = project.parent / "originals/Fiction"
    (source / "duplicate.jpg").write_bytes((source / "photo-0.jpg").read_bytes())
    prepare_dataset(dataset_config)
    build_split(dataset_config)
    split = json.loads((project / "data/split.provisional.json").read_text())
    inventory = read_inventory(project / "data/inventory.jsonl")
    originals = [
        r for r in inventory if r["file"] in ("Fiction/photo-0.jpg", "Fiction/duplicate.jpg")
    ]
    assert len(originals) == 2
    ids = {r["id"] for r in originals}
    matching_groups = [g for g in split["groups"] if ids.intersection(g["members"])]
    assert len(matching_groups) == 1
    assert ids.issubset(matching_groups[0]["members"])
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["validation"]) & set(split["test"])
    assert not set(split["train"]) & set(split["validation"])
    assert sum(len(split[s]) for s in ("train", "validation", "test")) == 31


def test_archive_is_self_contained_and_byte_reproducible(dataset_config):
    project = dataset_config.parent.parent
    prepare_dataset(dataset_config)
    first = build_split(dataset_config)
    archive_path = project / "artifacts/pilot_inputs.zip"
    first_bytes = archive_path.read_bytes()
    second = build_split(dataset_config)
    assert first == second
    assert archive_path.read_bytes() == first_bytes
    assert first["pilot_count"] == 12
    assert set(first["pilot_category_counts"].values()) == {2}
    split = json.loads((project / "data/split.provisional.json").read_text())
    saved_hash = split.pop("split_sha256")
    assert sha256(json_bytes(split)) == saved_hash
    with ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("pilot_manifest.json"))
        checksums = json.loads(archive.read("checksums.json"))
        for name, digest in checksums.items():
            assert sha256(archive.read(name)) == digest
        assert set(manifest["inputs"][i]["id"] for i in range(12)).issubset(split["train"])
        for row in manifest["inputs"]:
            assert row["category"] not in row["file"]
            assert sha256(archive.read(row["file"])) == row["sha256"]


def test_tampered_prepared_file_blocks_packaging(dataset_config):
    project = dataset_config.parent.parent
    prepare_dataset(dataset_config)
    inventory = read_inventory(project / "data/inventory.jsonl")
    (project / "data" / inventory[0]["prepared_file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_split(dataset_config)


def test_output_path_cannot_overwrite_source(dataset_config):
    config = yaml.safe_load(dataset_config.read_text())
    config["output_root"] = "../originals"
    dataset_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="inside the project"):
        prepare_dataset(dataset_config)


def test_near_visual_candidates_are_provisional_and_seeded(dataset_config):
    project = dataset_config.parent.parent
    prepare_dataset(dataset_config)
    rows = read_inventory(project / "data/inventory.jsonl")[:3]
    # Force a nonidentical near pair independent of fixture image encoding.
    rows[0]["dhash"], rows[1]["dhash"], rows[2]["dhash"] = (
        "0000000000000000",
        "0000000000000001",
        "ffffffffffffffff",
    )
    groups, edges = group_images(rows, 1)
    assert len(groups) == 2
    assert edges[0]["reason"] == "near_visual_candidate"
    assert not any(g["verified_book_identity"] for g in groups)
    split, pilot = assign_splits(rows, groups, seed=42, pilot_per_category=2)
    assert set(pilot["ids"]).issubset(split["train"])
    assert split["status"] == "provisional"
