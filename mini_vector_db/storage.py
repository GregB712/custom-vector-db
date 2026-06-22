"""Persistence backends for custom-vector-db."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .exceptions import StorageError
from .index import VectorRecord


class StorageBackend(Protocol):
    """Interface for persistence backends."""

    def save(self, path: str | Path, payload: dict[str, Any]) -> None:
        """Persist a serializable database payload."""

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a database payload."""


class LocalStorage:
    """JSON-on-disk storage backend.

    The database path is a directory containing ``manifest.json``. Vectors are
    serialized as lists, which is simple and inspectable for portfolio use.
    """

    filename = "manifest.json"

    def save(self, path: str | Path, payload: dict[str, Any]) -> None:
        directory = Path(path)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / self.filename).open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, sort_keys=True)
        except OSError as exc:
            raise StorageError(f"Failed to save database to {directory}.") from exc

    def load(self, path: str | Path) -> dict[str, Any]:
        directory = Path(path)
        file_path = directory / self.filename
        try:
            with file_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except OSError as exc:
            raise StorageError(f"Failed to load database from {file_path}.") from exc
        except json.JSONDecodeError as exc:
            raise StorageError(f"Invalid database JSON in {file_path}.") from exc

        if not isinstance(payload, dict):
            raise StorageError(f"Invalid database payload in {file_path}.")
        return payload


def record_to_payload(record: VectorRecord) -> dict[str, Any]:
    """Convert a record into JSON-serializable data."""

    return {
        "id": record.id,
        "vector": record.vector.tolist(),
        "metadata": record.metadata,
    }
