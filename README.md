# custom-vector-db

`custom-vector-db` is a small, readable vector database built from scratch in
Python. It stores dense vectors with string IDs and metadata, supports exact and
educational approximate similarity search, and persists databases to disk as
inspectable JSON.

The project is designed for learning. The code favors clear module boundaries,
typed APIs, and straightforward algorithms over production-level indexing
complexity.

## Features

- Insert vectors with unique string IDs and optional metadata.
- Search by cosine similarity, dot product, or Euclidean distance.
- Exact brute-force search for correctness and a baseline.
- Simple graph-based approximate search inspired by HNSW.
- Metadata filters with equality and comparison operators.
- JSON persistence through a small storage backend interface.
- Command-line interface for creating, loading, inserting, and searching.
- Pytest test suite and runnable examples.
- Benchmark script for exact vs approximate search.

## Installation

```bash
python -m pip install -e ".[dev]"
```

For runtime use only:

```bash
python -m pip install -e .
```

## Basic Usage

```python
from custom_vector_db import VectorDB

db = VectorDB(dim=3, metric="cosine")

db.insert(
    id="doc-1",
    vector=[0.1, 0.2, 0.3],
    metadata={
        "title": "Intro to embeddings",
        "category": "article",
        "year": 2024,
    },
)

results = db.search(
    query_vector=[0.2, 0.1, 0.4],
    top_k=5,
    filters={"category": "article"},
)

db.save("my_db")
db2 = VectorDB.load("my_db")
```

Search results are dictionaries:

```python
[
    {
        "id": "doc-1",
        "score": 0.9258,
        "metadata": {
            "title": "Intro to embeddings",
            "category": "article",
            "year": 2024,
        },
    }
]
```

## Metadata Filtering

Simple filters use equality:

```python
db.search([0.2, 0.1, 0.4], filters={"category": "article"})
```

Operator filters support `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, and `$in`:

```python
db.search(
    [0.2, 0.1, 0.4],
    filters={
        "year": {"$gte": 2022},
        "category": {"$eq": "article"},
        "views": {"$lt": 10000},
    },
)
```

All filter clauses must match. Missing fields fail comparison operators and
equality checks, except `$ne`, where a missing field is considered not equal.

## Similarity Metrics

- `cosine`: compares vector direction. Higher is better.
- `dot`: raw dot product. Higher is better.
- `euclidean`: physical distance in vector space. The public search score is
  normalized as `1 / (1 + distance)`, so higher is still better.

This keeps result sorting consistent across all metrics.

## Exact vs Approximate Search

`ExactIndex` scores every matching vector and returns the true top-k results. It
is simple and correct, but search time grows linearly with the number of stored
vectors.

`ApproxGraphIndex` builds a one-layer neighbor graph:

1. Each vector is a node.
2. New nodes connect to their closest existing neighbors.
3. Search starts from one entry point.
4. The index expands promising graph neighbors until `ef_search` candidates have
   been visited.

This can reduce query work, but it may miss true nearest neighbors. It is meant
to demonstrate the core idea behind navigable graph indexes, not compete with
production ANN libraries.

Use it with:

```python
db = VectorDB(dim=384, metric="cosine", index_type="approx")
```

## Persistence

Databases are saved to a directory containing `manifest.json`:

```python
db.save("my_db")
loaded = VectorDB.load("my_db")
```

The storage layer is intentionally small. `LocalStorage` implements JSON
persistence, and the `StorageBackend` protocol makes it straightforward to add
other backends later.

## CLI

Create a database:

```bash
custom-vector-db create my_db --dim 3 --metric cosine --index exact
```

Insert records from JSONL:

```bash
custom-vector-db insert-jsonl my_db examples/sample_records.jsonl
```

Each JSONL line should look like:

```json
{"id": "doc-1", "vector": [0.1, 0.2, 0.3], "metadata": {"category": "article"}}
```

Search from the terminal:

```bash
custom-vector-db search my_db --vector '[0.1, 0.2, 0.3]' --top-k 5 --filters '{"category":"article"}'
```

## Benchmarks

Run synthetic benchmarks:

```bash
python benchmarks/benchmark.py --sizes 1000 10000 50000 --dim 128 --queries 50 --top-k 10
```

The script reports:

- average exact query latency;
- average approximate query latency;
- recall@k, comparing approximate results against exact results.

The 50k-vector run is intentionally heavier because the pure-Python graph
construction does exact neighbor selection during insertion. Use smaller sizes
while iterating.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
python examples/basic_usage.py
```

## Limitations

- Vectors are kept in memory.
- JSON persistence is inspectable but not compact for large datasets.
- Approximate graph insertion uses brute-force neighbor selection.
- Metadata filtering is applied during search, not backed by a metadata index.
- There is no concurrency control or transactional write path.

## Future Improvements

- Add binary or memory-mapped storage for larger datasets.
- Add metadata indexes for faster filtered search.
- Improve graph construction with randomized entry points and multiple layers.
- Add update/upsert APIs.
- Add batch insertion.
- Add typed benchmark output such as CSV or JSON.
- Add optional vector normalization for cosine-heavy workloads.
