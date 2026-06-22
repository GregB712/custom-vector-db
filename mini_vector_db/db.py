"""Public database API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from .exceptions import (
    DimensionMismatchError,
    DuplicateIDError,
    MissingIDError,
    StorageError,
)
from .filters import FilterExpression
from .index import ApproxGraphIndex, ExactIndex, SearchIndex, SearchResult, VectorRecord
from .metrics import Metric, validate_metric
from .storage import LocalStorage, StorageBackend, record_to_payload


class VectorDB:
    """A small vector database with exact and approximate search."""

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        index_type: str = "exact",
        storage: StorageBackend | None = None,
        max_neighbors: int = 12,
        ef_search: int = 64,
    ) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1.")
        self.dim = dim
        self.metric: Metric = validate_metric(metric)
        self.index_type = index_type
        self.storage = storage or LocalStorage()
        self.records: dict[str, VectorRecord] = {}
        self.index = self._create_index(index_type, max_neighbors, ef_search)

    def insert(
        self,
        id: str,
        vector: Iterable[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert a vector with a unique string ID and optional metadata."""

        if not isinstance(id, str) or not id:
            raise ValueError("id must be a non-empty string.")
        if id in self.records:
            raise DuplicateIDError(f"Record with id {id!r} already exists.")

        array = self._coerce_vector(vector)
        record = VectorRecord(id=id, vector=array, metadata=dict(metadata or {}))
        self.records[id] = record
        self.index.insert(record)

    def get(self, id: str) -> VectorRecord:
        """Return a stored record by ID."""

        try:
            return self.records[id]
        except KeyError as exc:
            raise MissingIDError(f"Record with id {id!r} does not exist.") from exc

    def delete(self, id: str) -> None:
        """Delete a record by ID."""

        if id not in self.records:
            raise MissingIDError(f"Record with id {id!r} does not exist.")
        del self.records[id]
        self.index.delete(id)

    def search(
        self,
        query_vector: Iterable[float],
        top_k: int = 5,
        filters: FilterExpression | None = None,
    ) -> list[dict[str, Any]]:
        """Search for nearest vectors and return dictionaries.

        Scores are always sorted descending. For Euclidean search the score is
        ``1 / (1 + distance)``, so closer vectors receive higher scores.
        """

        query = self._coerce_vector(query_vector)
        results = self.index.search(query, top_k=top_k, filters=filters)
        return [result.to_dict() for result in results]

    def search_raw(
        self,
        query_vector: Iterable[float],
        top_k: int = 5,
        filters: FilterExpression | None = None,
    ) -> list[SearchResult]:
        """Search and return ``SearchResult`` dataclass instances."""

        query = self._coerce_vector(query_vector)
        return self.index.search(query, top_k=top_k, filters=filters)

    def save(self, path: str | Path) -> None:
        """Save the database to disk."""

        payload: dict[str, Any] = {
            "version": 1,
            "dim": self.dim,
            "metric": self.metric,
            "index_type": self.index_type,
            "records": [
                record_to_payload(record) for record in self.records.values()
            ],
        }
        if isinstance(self.index, ApproxGraphIndex):
            payload["index_state"] = self.index.export_state()

        self.storage.save(path, payload)

    @classmethod
    def load(
        cls,
        path: str | Path,
        storage: StorageBackend | None = None,
    ) -> "VectorDB":
        """Load a database from disk."""

        backend = storage or LocalStorage()
        payload = backend.load(path)

        try:
            dim = int(payload["dim"])
            metric = str(payload["metric"])
            index_type = str(payload.get("index_type", "exact"))
            records_payload = payload["records"]
        except (KeyError, TypeError, ValueError) as exc:
            raise StorageError("Database payload is missing required fields.") from exc

        db = cls(dim=dim, metric=metric, index_type=index_type, storage=backend)

        if not isinstance(records_payload, list):
            raise StorageError("Database records must be a list.")

        for item in records_payload:
            if not isinstance(item, dict):
                raise StorageError("Database record entries must be dictionaries.")
            db.insert(
                id=str(item["id"]),
                vector=item["vector"],
                metadata=dict(item.get("metadata") or {}),
            )

        if isinstance(db.index, ApproxGraphIndex) and "index_state" in payload:
            db.index.import_state(payload["index_state"])

        return db

    def __len__(self) -> int:
        return len(self.records)

    def _create_index(
        self, index_type: str, max_neighbors: int, ef_search: int
    ) -> SearchIndex:
        if index_type == "exact":
            return ExactIndex(metric=self.metric)
        if index_type == "approx":
            return ApproxGraphIndex(
                metric=self.metric,
                max_neighbors=max_neighbors,
                ef_search=ef_search,
            )
        raise ValueError("index_type must be 'exact' or 'approx'.")

    def _coerce_vector(self, vector: Iterable[float]) -> NDArray[np.float64]:
        array = np.asarray(list(vector), dtype=np.float64)
        if array.ndim != 1:
            raise DimensionMismatchError("Vectors must be one-dimensional.")
        if array.shape[0] != self.dim:
            raise DimensionMismatchError(
                f"Expected vector dimension {self.dim}, got {array.shape[0]}."
            )
        return array
