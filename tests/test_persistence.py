from pathlib import Path

from custom_vector_db import VectorDB


def test_save_and_load_exact_database(tmp_path: Path) -> None:
    path = tmp_path / "db"
    db = VectorDB(dim=3, metric="cosine")
    db.insert("doc-1", [1, 0, 0], {"category": "article"})
    db.save(path)

    loaded = VectorDB.load(path)
    results = loaded.search([1, 0, 0], top_k=1)

    assert len(loaded) == 1
    assert results[0]["id"] == "doc-1"
    assert results[0]["metadata"] == {"category": "article"}


def test_save_and_load_approx_database(tmp_path: Path) -> None:
    path = tmp_path / "db"
    db = VectorDB(dim=2, metric="cosine", index_type="approx")
    db.insert("a", [1, 0])
    db.insert("b", [0, 1])
    db.save(path)

    loaded = VectorDB.load(path)
    results = loaded.search([1, 0], top_k=1)

    assert results[0]["id"] == "a"


def test_insert_after_loading_empty_approx_database(tmp_path: Path) -> None:
    path = tmp_path / "db"
    db = VectorDB(dim=2, metric="cosine", index_type="approx")
    db.save(path)

    loaded = VectorDB.load(path)
    loaded.insert("a", [1, 0], {"category": "article"})
    loaded.insert("b", [0, 1], {"category": "note"})
    loaded.save(path)

    reloaded = VectorDB.load(path)
    results = reloaded.search([1, 0], top_k=1, filters={"category": "article"})

    assert results[0]["id"] == "a"
