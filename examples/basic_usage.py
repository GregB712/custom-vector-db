"""Basic custom-vector-db usage."""

from mini_vector_db import VectorDB


def main() -> None:
    db = VectorDB(dim=3, metric="cosine")
    db.insert(
        id="doc-1",
        vector=[0.1, 0.2, 0.3],
        metadata={"title": "Intro to embeddings", "category": "article", "year": 2024},
    )
    db.insert(
        id="doc-2",
        vector=[0.9, 0.1, 0.1],
        metadata={"title": "Vector search notes", "category": "note", "year": 2023},
    )

    results = db.search(query_vector=[0.1, 0.2, 0.25], top_k=2)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
