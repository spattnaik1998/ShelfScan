"""Atomic role-state cache; attempts are persisted before invoking an adapter."""

import json
import os
from pathlib import Path
from uuid import uuid4

from scripts.common import json_bytes, sha256


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(json_bytes(value))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class CorruptCache(ValueError):
    pass


class RoleCache:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(provenance: dict) -> str:
        return sha256(json_bytes(provenance))

    def load(self, key: str) -> dict | None:
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_bytes())
            payload = envelope["payload"]
            if sha256(json_bytes(payload)) != envelope["sha256"]:
                raise ValueError("Checksum mismatch")
            if self.key(payload["provenance"]) != key:
                raise ValueError("Provenance mismatch")
            return payload
        except (ValueError, KeyError, TypeError) as exc:
            # Never silently reinvoke an expensive model after cache corruption.
            raise CorruptCache(f"Invalid cached role state: {key}") from exc

    def save(self, key: str, state: dict) -> None:
        atomic_json(
            self.root / f"{key}.json",
            {
                "payload": state,
                "sha256": sha256(json_bytes(state)),
            },
        )

    def total_attempts(self) -> int:
        return sum(len(self.load(p.stem)["attempts"]) for p in self.root.glob("*.json"))
