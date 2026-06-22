import pytest

from mini_vector_db import VectorDB
from mini_vector_db.exceptions import (
    DimensionMismatchError,
    DuplicateIDError,
    MissingIDError,
)


def test_insert_and_exact_search_cosine() -> None:
    db = VectorDB(dim=3, metric="cosine")
    db.insert("a", [1, 0, 0], {"category": "article"})
    db.insert("b", [0, 1, 0], {"category": "note"})

    results = db.search([1, 0, 0], top_k=2)

    assert [result["id"] for result in results] == ["a", "b"]
    assert results[0]["metadata"] == {"category": "article"}


def test_search_with_metadata_filter() -> None:
    db = VectorDB(dim=2)
    db.insert("a", [1, 0], {"category": "article", "year": 2024})
    db.insert("b", [0.9, 0.1], {"category": "note", "year": 2024})
    db.insert("c", [0, 1], {"category": "article", "year": 2020})

    results = db.search(
        [1, 0],
        top_k=5,
        filters={"category": "article", "year": {"$gte": 2022}},
    )

    assert [result["id"] for result in results] == ["a"]


def test_duplicate_id_error() -> None:
    db = VectorDB(dim=2)
    db.insert("a", [1, 0])
    with pytest.raises(DuplicateIDError):
        db.insert("a", [0, 1])


def test_dimension_mismatch_error() -> None:
    db = VectorDB(dim=3)
    with pytest.raises(DimensionMismatchError):
        db.insert("bad", [1, 2])


def test_missing_id_error() -> None:
    db = VectorDB(dim=2)
    with pytest.raises(MissingIDError):
        db.get("missing")


def test_euclidean_search_uses_higher_score_for_closer_vectors() -> None:
    db = VectorDB(dim=2, metric="euclidean")
    db.insert("close", [0, 0])
    db.insert("far", [10, 10])

    results = db.search([1, 1], top_k=2)

    assert [result["id"] for result in results] == ["close", "far"]
    assert results[0]["score"] > results[1]["score"]


def test_approx_index_returns_reasonable_nearest_neighbor() -> None:
    db = VectorDB(dim=2, metric="cosine", index_type="approx", ef_search=10)
    db.insert("a", [1, 0])
    db.insert("b", [0.9, 0.1])
    db.insert("c", [0, 1])

    results = db.search([1, 0], top_k=1)

    assert results[0]["id"] == "a"


def test_top_k_zero_returns_empty_list() -> None:
    db = VectorDB(dim=2)
    db.insert("a", [1, 0])
    assert db.search([1, 0], top_k=0) == []
