"""Similarity and distance functions for dense vectors."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .exceptions import UnsupportedMetricError

Metric = Literal["cosine", "dot", "euclidean"]

SUPPORTED_METRICS: set[str] = {"cosine", "dot", "euclidean"}


def validate_metric(metric: str) -> Metric:
    """Return a typed metric name or raise if it is unsupported."""

    if metric not in SUPPORTED_METRICS:
        supported = ", ".join(sorted(SUPPORTED_METRICS))
        raise UnsupportedMetricError(
            f"Unsupported metric {metric!r}. Supported metrics: {supported}."
        )
    return metric  # type: ignore[return-value]


def cosine_similarity(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Compute cosine similarity between two vectors.

    Zero vectors receive a similarity of 0.0. This keeps search behavior
    deterministic while avoiding division-by-zero errors.
    """

    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / norm)


def dot_product(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Compute the dot product between two vectors."""

    return float(np.dot(a, b))


def euclidean_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Compute Euclidean distance between two vectors."""

    return float(np.linalg.norm(a - b))


def score_vector(a: NDArray[np.float64], b: NDArray[np.float64], metric: Metric) -> float:
    """Return a search score where larger is always better.

    For cosine similarity and dot product this is the raw metric value.
    For Euclidean distance this returns ``1 / (1 + distance)``, which maps
    smaller distances to larger scores in the range ``(0, 1]``.
    """

    if metric == "cosine":
        return cosine_similarity(a, b)
    if metric == "dot":
        return dot_product(a, b)
    if metric == "euclidean":
        distance = euclidean_distance(a, b)
        return 1.0 / (1.0 + distance)
    raise UnsupportedMetricError(f"Unsupported metric {metric!r}.")
