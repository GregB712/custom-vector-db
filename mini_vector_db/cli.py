"""Command-line interface for custom-vector-db."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .db import VectorDB


def main() -> None:
    parser = argparse.ArgumentParser(prog="custom-vector-db")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create an empty database")
    create.add_argument("path")
    create.add_argument("--dim", type=int, required=True)
    create.add_argument("--metric", default="cosine", choices=["cosine", "dot", "euclidean"])
    create.add_argument("--index", default="exact", choices=["exact", "approx"])

    insert = subparsers.add_parser("insert-jsonl", help="Insert JSONL records")
    insert.add_argument("path")
    insert.add_argument("jsonl")

    search = subparsers.add_parser("search", help="Search a saved database")
    search.add_argument("path")
    search.add_argument("--vector", required=True, help="JSON array query vector")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--filters", help="JSON object metadata filters")

    args = parser.parse_args()

    if args.command == "create":
        db = VectorDB(dim=args.dim, metric=args.metric, index_type=args.index)
        db.save(args.path)
        print(f"Created database at {args.path}")
        return

    if args.command == "insert-jsonl":
        db = VectorDB.load(args.path)
        count = _insert_jsonl(db, Path(args.jsonl))
        db.save(args.path)
        print(f"Inserted {count} records into {args.path}")
        return

    if args.command == "search":
        db = VectorDB.load(args.path)
        query_vector = json.loads(args.vector)
        filters = json.loads(args.filters) if args.filters else None
        results = db.search(query_vector=query_vector, top_k=args.top_k, filters=filters)
        print(json.dumps(results, indent=2))


def _insert_jsonl(db: VectorDB, path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item: dict[str, Any] = json.loads(line)
            try:
                db.insert(
                    id=str(item["id"]),
                    vector=item["vector"],
                    metadata=dict(item.get("metadata") or {}),
                )
            except KeyError as exc:
                raise ValueError(
                    f"JSONL line {line_number} must contain id and vector fields."
                ) from exc
            count += 1
    return count


if __name__ == "__main__":
    main()
