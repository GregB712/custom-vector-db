"""Searching with metadata filters."""

from mini_vector_db import VectorDB


def main() -> None:
    db = VectorDB(dim=3)
    db.insert("doc-1", [1, 0, 0], {"category": "article", "author": "alice", "year": 2024})
    db.insert("doc-2", [0, 1, 0], {"category": "article", "author": "bob", "year": 2021})
    db.insert("doc-3", [0.9, 0.1, 0], {"category": "note", "author": "alice", "year": 2023})

    results = db.search(
        query_vector=[1, 0, 0],
        top_k=5,
        filters={"category": {"$eq": "article"}, "year": {"$gte": 2022}},
    )
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
