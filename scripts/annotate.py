"""Run bounded fake annotation on a verified pilot archive or prepared inventory."""

import argparse
import io
import json
import platform
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import yaml
from PIL import Image

from annotation.cache import atomic_json
from annotation.contracts import AnnotationConfig, VisualInput
from annotation.orchestrator import AnnotationEngine, BudgetStop, save_reports
from annotation.teachers import FakeAdjudicator, FakeOCR, FakeTeacher
from scripts.common import json_bytes, read_inventory, sha256
from shelfscan.prompt import PROMPT_SHA256
from shelfscan.schemas import CATEGORY_MAP


def inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("Annotation paths must stay inside the project")
    return path


def verified_split(path: Path) -> dict:
    split = json.loads(path.read_bytes())
    payload = {k: v for k, v in split.items() if k != "split_sha256"}
    if sha256(json_bytes(payload)) != split["split_sha256"]:
        raise ValueError("Split checksum mismatch")
    partitions = [set(split[name]) for name in ("train", "validation", "test")]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Split overlap")
    seen = set()
    for group in split["groups"]:
        members = set(group["members"])
        if seen & members or len(members) != len(group["members"]):
            raise ValueError("Duplicate group membership")
        seen |= members
        assigned = split["assignments"][group["group_id"]]
        if not members.issubset(split[assigned]):
            raise ValueError("Group crosses split partitions")
    if seen != set.union(*partitions):
        raise ValueError("Groups do not exactly cover the split")
    return split


def load_inputs(config: AnnotationConfig, root: Path) -> tuple[list[tuple[VisualInput, dict]], str]:
    split = verified_split(inside(root, config.split_path))
    group_of = {m: g["group_id"] for g in split["groups"] for m in g["members"]}
    source = inside(root, config.input_path)
    image_root = inside(root, config.image_root)
    if config.input_kind == "pilot":
        with ZipFile(source) as archive:
            if len(archive.namelist()) != len(set(archive.namelist())):
                raise ValueError("Duplicate archive members")
            if any(info.file_size > 16 * 1024 * 1024 for info in archive.infolist()):
                raise ValueError("Oversized pilot member")
            if sum(info.file_size for info in archive.infolist()) > 64 * 1024 * 1024:
                raise ValueError("Oversized pilot archive")
            checksums = json.loads(archive.read("checksums.json"))
            if set(archive.namelist()) != set(checksums) | {"checksums.json"}:
                raise ValueError("Unexpected archive members")
            for name, digest in checksums.items():
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name:
                    raise ValueError("Unsafe archive path")
                if sha256(archive.read(name)) != digest:
                    raise ValueError("Pilot member checksum mismatch")
            manifest = json.loads(archive.read("pilot_manifest.json"))
            if manifest["split_sha256"] != split["split_sha256"]:
                raise ValueError("Pilot belongs to a different split")
            if manifest["prompt_sha256"] != PROMPT_SHA256:
                raise ValueError("Pilot uses a different extraction prompt")
            rows = manifest["inputs"]
            raw_images = {row["id"]: archive.read(row["file"]) for row in rows}
    else:
        rows = read_inventory(source)
        if any(row["status"] != "ok" for row in rows):
            raise ValueError("Resolve inventory decode failures before annotation")
        if split["inventory_sha256"] != sha256(source.read_bytes()):
            raise ValueError("Inventory belongs to a different split")
        rows = [{**r, "sha256": r["prepared_sha256"]} for r in rows]
        raw_images = {
            row["id"]: inside(image_root, row["prepared_file"]).read_bytes() for row in rows
        }
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Inputs must have unique nonempty identifiers")
    results = []
    for row in sorted(rows, key=lambda r: r["id"]):
        raw = raw_images[row["id"]]
        if sha256(raw) != row["sha256"]:
            raise ValueError("Input image checksum mismatch")
        if row["category"] not in CATEGORY_MAP.values() or row["id"] not in group_of:
            raise ValueError("Invalid input category or split membership")
        partition = next(
            name for name in ("train", "validation", "test") if row["id"] in split[name]
        )
        if config.input_kind == "pilot" and partition != "train":
            raise ValueError("Pilot group must be training-only")
        with Image.open(io.BytesIO(raw)) as image:
            if image.size != (row["width"], row["height"]) or max(image.size) > 1024:
                raise ValueError("Prepared image dimensions mismatch")
            image.load()
        visual = VisualInput(
            image_id=sha256(row["id"].encode())[:24],
            image_sha256=row["sha256"],
            image_bytes=raw,
            width=row["width"],
            height=row["height"],
        )
        metadata = {
            "id": row["id"],
            "image_sha256": row["sha256"],
            "source_sha256": row["source_sha256"],
            "category": row["category"],
            "group_id": group_of[row["id"]],
            "split": partition,
        }
        results.append((visual, metadata))
    digest = sha256(json_bytes({"split": split["split_sha256"], "inputs": [m for _, m in results]}))
    return results, digest


@contextmanager
def single_writer(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / ".writer.lock"
    try:
        handle = lock.open("x")
    except FileExistsError as exc:
        raise RuntimeError(
            "Run is locked; confirm no worker is active before removing stale lock"
        ) from exc
    try:
        with handle:
            handle.write(
                "Single-writer annotation run; remove only after confirming worker stopped.\n"
            )
        yield
    finally:
        lock.unlink(missing_ok=True)


def run(
    config_path: str | Path, *, backend: str, adapters: dict | None = None
) -> tuple[dict, Path]:
    config_path = Path(config_path).resolve()
    config = AnnotationConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    if backend != config.backend:
        raise ValueError("Backend argument must match configuration")
    root = config_path.parent.parent
    inputs, input_hash = load_inputs(config, root)
    identity_config = config.model_dump(
        mode="json",
        exclude={
            "max_total_attempts",
            "max_seconds",
            "output_root",
        },
    )
    code_files = [
        *Path(__file__).parent.glob("*.py"),
        *Path(__file__).parent.parent.joinpath("annotation").glob("*.py"),
    ]
    source_hash = sha256(
        json_bytes(
            {
                f"{path.parent.name}/{path.name}": sha256(path.read_bytes())
                for path in sorted(code_files)
            }
        )
    )
    identity = sha256(
        json_bytes(
            {
                "config": identity_config,
                "inputs": input_hash,
                "source_sha256": source_hash,
            }
        )
    )
    run_dir = inside(root, config.output_root) / identity[:24]
    adapters = adapters or {
        "ocr": FakeOCR(),
        "teacher_a": FakeTeacher(),
        "teacher_b": FakeTeacher(
            disagree=bool(config.roles["teacher_b"].parameters.get("disagree"))
        ),
        "adjudicator": FakeAdjudicator(),
    }
    with single_writer(run_dir):
        engine = AnnotationEngine(config, run_dir, **adapters)
        decisions = [
            {**metadata, "disposition": "pending", "is_synthetic": True, "training_eligible": False}
            for _, metadata in inputs
        ]
        atomic_json(
            run_dir / "input_manifest.json",
            {
                "input_sha256": input_hash,
                "run_identity": identity,
                "source_sha256": source_hash,
                "prompt_sha256": PROMPT_SHA256,
                "python_version": platform.python_version(),
                "inputs": [m for _, m in inputs],
            },
        )
        stopped = None
        for index, (image, metadata) in enumerate(inputs):
            try:
                decisions[index] = engine.annotate(image, metadata)
            except BudgetStop as exc:
                stopped = str(exc)
                break
            save_reports(engine, decisions, stopped="Checkpoint; remaining images pending")
        summary = save_reports(engine, decisions, stopped=stopped)
        atomic_json(
            run_dir / "artifact_checksums.json",
            {
                path.name: sha256(path.read_bytes())
                for path in sorted(run_dir.iterdir())
                if path.is_file() and path.name not in ("artifact_checksums.json", ".writer.lock")
            },
        )
    return summary, run_dir


def main(argv: list[str] | None = None, *, pilot_only: bool = False) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/annotation.yaml")
    parser.add_argument("--backend", choices=["fake"], required=True)
    args = parser.parse_args(argv)
    if pilot_only:
        raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        if raw_config.get("input_kind", "pilot") != "pilot":
            parser.error("pilot command requires input_kind: pilot")
    summary, directory = run(args.config, backend=args.backend)
    print(
        json.dumps(
            {
                "warning": summary["warning"],
                "status": summary["status"],
                "output": str(directory),
                "input_count": summary["input_count"],
                "simulated_dispositions": summary["simulated_dispositions"],
                "new_attempts": summary["new_attempts"],
                "total_attempts": summary["total_attempts"],
                "cache_hits": summary["cache_hits"],
                "training_eligible": 0,
            },
            indent=2,
        )
    )
    if summary["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
