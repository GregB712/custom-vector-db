"""Custom exceptions used by custom-vector-db."""


class VectorDBError(Exception):
    """Base exception for all vector database errors."""


class DuplicateIDError(VectorDBError):
    """Raised when inserting a record with an ID that already exists."""


class MissingIDError(VectorDBError):
    """Raised when an operation references an ID that does not exist."""


class DimensionMismatchError(VectorDBError):
    """Raised when a vector does not match the configured dimensionality."""


class UnsupportedMetricError(VectorDBError):
    """Raised when an unknown similarity metric is requested."""


class InvalidFilterError(VectorDBError):
    """Raised when a metadata filter expression is invalid."""


class StorageError(VectorDBError):
    """Raised when persistence fails."""
