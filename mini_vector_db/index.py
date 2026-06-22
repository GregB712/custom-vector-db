"""Search index implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .filters import FilterExpression, matches_filter
from .metrics import Metric, score_vector


@dataclass(slots=True)
class VectorRecord:
    """A stored vector and its user metadata."""

    id: str
    vector: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search hit returned by an index."""

    id: str
    score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to the public dictionary representation."""

        return {"id": self.id, "score": self.score, "metadata": self.metadata}


class SearchIndex(Protocol):
    """Common interface for exact and approximate indexes."""

    def insert(self, record: VectorRecord) -> None:
        """Insert a record into the index."""

    def delete(self, id: str) -> None:
        """Delete a record from the index."""

    def rebuild(self, records: dict[str, VectorRecord]) -> None:
        """Rebuild the index from records."""

    def search(
        self,
        query_vector: NDArray[np.float64],
        top_k: int,
        filters: FilterExpression | None = None,
    ) -> list[SearchResult]:
        """Search for nearest records."""


class ExactIndex:
    """Brute-force exact nearest-neighbor index."""

    def __init__(self, metric: Metric) -> None:
        self.metric = metric
        self.records: dict[str, VectorRecord] = {}

    def insert(self, record: VectorRecord) -> None:
        self.records[record.id] = record

    def delete(self, id: str) -> None:
        self.records.pop(id, None)

    def rebuild(self, records: dict[str, VectorRecord]) -> None:
        self.records = dict(records)

    def search(
        self,
        query_vector: NDArray[np.float64],
        top_k: int,
        filters: FilterExpression | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []

        scored: list[SearchResult] = []
        for record in self.records.values():
            if not matches_filter(record.metadata, filters):
                continue
            scored.append(
                SearchResult(
                    id=record.id,
                    score=score_vector(query_vector, record.vector, self.metric),
                    metadata=dict(record.metadata),
                )
            )

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]


class ApproxGraphIndex:
    """Small educational graph-based approximate nearest-neighbor index.

    This is inspired by navigable small-world indexes such as HNSW, but it is
    deliberately much simpler:

    - every vector is a node in one graph layer;
    - inserted nodes connect to their nearest existing neighbors;
    - search starts from an entry point and expands promising neighbors first;
    - only ``ef_search`` candidates are explored, trading recall for speed.

    It is useful for learning and small demos, not as a replacement for FAISS,
    HNSWLib, ScaNN, or other production ANN libraries.
    """

    def __init__(
        self,
        metric: Metric,
        max_neighbors: int = 12,
        ef_search: int = 64,
    ) -> None:
        if max_neighbors < 1:
            raise ValueError("max_neighbors must be >= 1.")
        if ef_search < 1:
            raise ValueError("ef_search must be >= 1.")
        self.metric = metric
        self.max_neighbors = max_neighbors
        self.ef_search = ef_search
        self.records: dict[str, VectorRecord] = {}
        self.graph: dict[str, list[str]] = {}
        self.entrypoint: str | None = None

    def insert(self, record: VectorRecord) -> None:
        if not self.records:
            self.records[record.id] = record
            self.graph[record.id] = []
            self.entrypoint = record.id
            return

        neighbors = self._nearest_existing(record.vector, self.max_neighbors)
        self.records[record.id] = record
        self.graph[record.id] = [neighbor_id for neighbor_id, _ in neighbors]

        for neighbor_id, _ in neighbors:
            linked = self.graph.setdefault(neighbor_id, [])
            if record.id not in linked:
                linked.append(record.id)
            self._prune_neighbors(neighbor_id)

    def delete(self, id: str) -> None:
        self.records.pop(id, None)
        self.graph.pop(id, None)
        for neighbors in self.graph.values():
            if id in neighbors:
                neighbors.remove(id)
        if self.entrypoint == id:
            self.entrypoint = next(iter(self.records), None)

    def rebuild(self, records: dict[str, VectorRecord]) -> None:
        self.records = {}
        self.graph = {}
        self.entrypoint = None
        for record in records.values():
            self.insert(record)

    def search(
        self,
        query_vector: NDArray[np.float64],
        top_k: int,
        filters: FilterExpression | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0 or self.entrypoint is None:
            return []

        visited: set[str] = set()
        heap: list[tuple[float, str]] = []
        entry_score = score_vector(
            query_vector, self.records[self.entrypoint].vector, self.metric
        )
        heappush(heap, (-entry_score, self.entrypoint))

        candidates: list[SearchResult] = []
        max_visits = max(self.ef_search, top_k)

        while heap and len(visited) < max_visits:
            negative_score, record_id = heappop(heap)
            if record_id in visited:
                continue
            visited.add(record_id)

            record = self.records[record_id]
            score = -negative_score
            if matches_filter(record.metadata, filters):
                candidates.append(
                    SearchResult(
                        id=record.id,
                        score=score,
                        metadata=dict(record.metadata),
                    )
                )

            for neighbor_id in self.graph.get(record_id, []):
                if neighbor_id in visited:
                    continue
                neighbor = self.records[neighbor_id]
                neighbor_score = score_vector(query_vector, neighbor.vector, self.metric)
                heappush(heap, (-neighbor_score, neighbor_id))

        candidates.sort(key=lambda result: result.score, reverse=True)
        return candidates[:top_k]

    def _nearest_existing(
        self, vector: NDArray[np.float64], limit: int
    ) -> list[tuple[str, float]]:
        scored = [
            (record.id, score_vector(vector, record.vector, self.metric))
            for record in self.records.values()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def _prune_neighbors(self, record_id: str) -> None:
        record = self.records[record_id]
        neighbors = self.graph.get(record_id, [])
        scored = [
            (neighbor_id, score_vector(record.vector, self.records[neighbor_id].vector, self.metric))
            for neighbor_id in neighbors
            if neighbor_id in self.records and neighbor_id != record_id
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        self.graph[record_id] = [
            neighbor_id for neighbor_id, _ in scored[: self.max_neighbors]
        ]

    def export_state(self) -> dict[str, Any]:
        """Return serializable graph state for persistence."""

        return {
            "max_neighbors": self.max_neighbors,
            "ef_search": self.ef_search,
            "entrypoint": self.entrypoint,
            "graph": self.graph,
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """Restore graph state after records have been loaded."""

        self.max_neighbors = int(state.get("max_neighbors", self.max_neighbors))
        self.ef_search = int(state.get("ef_search", self.ef_search))
        entrypoint = state.get("entrypoint")
        self.entrypoint = entrypoint if entrypoint in self.records else next(iter(self.records), None)
        self.graph = {
            str(record_id): [
                str(neighbor_id)
                for neighbor_id in neighbors
                if str(neighbor_id) in self.records and str(neighbor_id) != str(record_id)
            ]
            for record_id, neighbors in state.get("graph", {}).items()
            if str(record_id) in self.records
        }
