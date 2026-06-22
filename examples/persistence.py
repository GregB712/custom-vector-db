"""Saving and loading a database."""

from pathlib import Path
from tempfile import TemporaryDirectory

from custom_vector_db import VectorDB


def main() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "my_db"

        db = VectorDB(dim=3, metric="euclidean", index_type="approx")
        db.insert("doc-1", [0.0, 0.0, 0.0], {"category": "origin"})
        db.insert("doc-2", [1.0, 1.0, 1.0], {"category": "corner"})
        db.save(path)

        loaded = VectorDB.load(path)
        print(loaded.search([0.1, 0.1, 0.1], top_k=1))


if __name__ == "__main__":
    main()
