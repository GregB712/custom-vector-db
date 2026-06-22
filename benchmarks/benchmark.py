"""Benchmark exact and approximate search on synthetic datasets."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from custom_vector_db import VectorDB


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 10000, 50000])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    for size in args.sizes:
        vectors = rng.normal(size=(size, args.dim))
        query_vectors = rng.normal(size=(args.queries, args.dim))

        exact = VectorDB(dim=args.dim, metric="cosine", index_type="exact")
        approx = VectorDB(
            dim=args.dim,
            metric="cosine",
            index_type="approx",
            max_neighbors=12,
            ef_search=128,
        )

        for i, vector in enumerate(vectors):
            metadata = {"bucket": int(i % 10)}
            exact.insert(f"vec-{i}", vector, metadata)
            approx.insert(f"vec-{i}", vector, metadata)

        exact_latency, exact_results = _measure(exact, query_vectors, args.top_k)
        approx_latency, approx_results = _measure(approx, query_vectors, args.top_k)
        recall = _recall_at_k(exact_results, approx_results)

        print(
            f"n={size:>6} | "
            f"exact={exact_latency * 1000:>8.2f} ms/query | "
            f"approx={approx_latency * 1000:>8.2f} ms/query | "
            f"recall@{args.top_k}={recall:.3f}"
        )


def _measure(
    db: VectorDB, queries: np.ndarray, top_k: int
) -> tuple[float, list[list[str]]]:
    latencies: list[float] = []
    results: list[list[str]] = []
    for query in queries:
        start = time.perf_counter()
        hits = db.search(query, top_k=top_k)
        latencies.append(time.perf_counter() - start)
        results.append([hit["id"] for hit in hits])
    return statistics.mean(latencies), results


def _recall_at_k(exact_results: list[list[str]], approx_results: list[list[str]]) -> float:
    recalls: list[float] = []
    for exact, approx in zip(exact_results, approx_results, strict=True):
        if not exact:
            continue
        recalls.append(len(set(exact) & set(approx)) / len(exact))
    return statistics.mean(recalls) if recalls else 0.0


if __name__ == "__main__":
    main()
