import pytest

from custom_vector_db.exceptions import InvalidFilterError
from custom_vector_db.filters import matches_filter


def test_simple_equality_filter() -> None:
    assert matches_filter({"category": "article"}, {"category": "article"})
    assert not matches_filter({"category": "note"}, {"category": "article"})


def test_operator_filters() -> None:
    metadata = {"year": 2024, "category": "article", "views": 500}
    assert matches_filter(
        metadata,
        {"year": {"$gte": 2022}, "category": {"$eq": "article"}, "views": {"$lt": 1000}},
    )
    assert not matches_filter(metadata, {"year": {"$lt": 2020}})


def test_in_and_ne_filters() -> None:
    metadata = {"author": "alice", "category": "article"}
    assert matches_filter(metadata, {"author": {"$in": ["alice", "bob"]}})
    assert matches_filter(metadata, {"category": {"$ne": "note"}})


def test_invalid_operator() -> None:
    with pytest.raises(InvalidFilterError):
        matches_filter({"year": 2024}, {"year": {"$between": [2020, 2025]}})


def test_invalid_in_operand() -> None:
    with pytest.raises(InvalidFilterError):
        matches_filter({"author": "alice"}, {"author": {"$in": "alice"}})


def test_invalid_comparison() -> None:
    with pytest.raises(InvalidFilterError):
        matches_filter({"year": "new"}, {"year": {"$gte": 2022}})
