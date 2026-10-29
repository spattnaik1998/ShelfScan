"""Validated preparation configuration and deterministic artifact serialization."""

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from shelfscan.schemas import CATEGORY_MAP


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_root: str
    output_root: str = "data"
    archive_path: str = "artifacts/pilot_inputs.zip"
    expected_images: int = Field(default=128, ge=1)
    seed: int = 42
    long_edge: int = Field(default=1024, ge=1, le=4096)
    jpeg_quality: int = Field(default=85, ge=1, le=95)
    max_upload_mb: int = Field(default=15, ge=1)
    max_pixels: int = Field(default=40_000_000, ge=1)
    pilot_per_category: int = Field(default=2, ge=1)
    near_duplicate_distance: int = Field(default=4, ge=0, le=8)
    category_map: dict[str, str] = Field(default_factory=lambda: dict(CATEGORY_MAP))


def load_config(path: str | Path) -> tuple[DataConfig, Path]:
    path = Path(path).resolve()
    config = DataConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if config.category_map != CATEGORY_MAP:
        raise ValueError("category_map must match the shared six-category contract")
    root = path.parent.parent
    source = (root / config.source_root).resolve()
    output = (root / config.output_root).resolve()
    archive = (root / config.archive_path).resolve()
    if not source.is_dir():
        raise ValueError(f"Dataset directory not found: {source}")
    if not output.is_relative_to(root) or not archive.is_relative_to(root):
        raise ValueError("Generated output/archive must stay inside the project")
    if output == root or archive == root:
        raise ValueError("Generated targets must name a project subdirectory/file")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and output directories must be disjoint")
    return config, root


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def read_inventory(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
