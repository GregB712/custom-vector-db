import numpy as np
import pytest

from custom_vector_db.exceptions import UnsupportedMetricError
from custom_vector_db.metrics import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    score_vector,
    validate_metric,
)


def test_cosine_similarity() -> None:
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_dot_product() -> None:
    assert dot_product(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 11.0


def test_euclidean_distance() -> None:
    assert euclidean_distance(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == 5.0


def test_euclidean_score_is_higher_for_closer_vectors() -> None:
    query = np.array([0.0, 0.0])
    close = score_vector(query, np.array([1.0, 0.0]), "euclidean")
    far = score_vector(query, np.array([3.0, 4.0]), "euclidean")
    assert close > far


def test_validate_metric_rejects_unknown_metric() -> None:
    with pytest.raises(UnsupportedMetricError):
        validate_metric("manhattan")
