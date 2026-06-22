"""custom-vector-db: a small educational vector database."""

from .db import VectorDB
from .exceptions import (
    DimensionMismatchError,
    DuplicateIDError,
    InvalidFilterError,
    MissingIDError,
    StorageError,
    UnsupportedMetricError,
    VectorDBError,
)
from .index import ApproxGraphIndex, ExactIndex, SearchResult, VectorRecord

__all__ = [
    "ApproxGraphIndex",
    "DimensionMismatchError",
    "DuplicateIDError",
    "ExactIndex",
    "InvalidFilterError",
    "MissingIDError",
    "SearchResult",
    "StorageError",
    "UnsupportedMetricError",
    "VectorDB",
    "VectorDBError",
    "VectorRecord",
]
