"""Metadata filter evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .exceptions import InvalidFilterError

FilterExpression = Mapping[str, Any]

SUPPORTED_OPERATORS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in"}


def matches_filter(
    metadata: Mapping[str, Any], filters: FilterExpression | None
) -> bool:
    """Return whether metadata satisfies all filter clauses.

    Supported forms:

    - ``{"category": "article"}`` means equality.
    - ``{"year": {"$gte": 2022}}`` uses explicit operators.
    """

    if filters is None:
        return True
    if not isinstance(filters, Mapping):
        raise InvalidFilterError("Filters must be a dictionary.")

    for field, expression in filters.items():
        if not isinstance(field, str) or not field:
            raise InvalidFilterError("Filter field names must be non-empty strings.")

        value = metadata.get(field)
        if isinstance(expression, Mapping):
            if not expression:
                raise InvalidFilterError(f"Filter for {field!r} cannot be empty.")
            for operator, expected in expression.items():
                if operator not in SUPPORTED_OPERATORS:
                    raise InvalidFilterError(
                        f"Unsupported filter operator {operator!r} for field {field!r}."
                    )
                if not _apply_operator(value, operator, expected, field):
                    return False
        elif value != expression:
            return False

    return True


def _apply_operator(value: Any, operator: str, expected: Any, field: str) -> bool:
    if operator == "$eq":
        return value == expected
    if operator == "$ne":
        return value != expected
    if operator == "$in":
        if isinstance(expected, (str, bytes)) or not isinstance(expected, Sequence):
            raise InvalidFilterError(
                f"$in filter for {field!r} must use a non-string sequence."
            )
        return value in expected

    if value is None:
        return False

    try:
        if operator == "$gt":
            return value > expected
        if operator == "$gte":
            return value >= expected
        if operator == "$lt":
            return value < expected
        if operator == "$lte":
            return value <= expected
    except TypeError as exc:
        raise InvalidFilterError(
            f"Cannot compare metadata field {field!r} value {value!r} "
            f"with {expected!r} using {operator}."
        ) from exc

    raise InvalidFilterError(f"Unsupported filter operator {operator!r}.")
