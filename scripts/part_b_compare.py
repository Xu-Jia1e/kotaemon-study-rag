"""Compare vector and existing hybrid retrieval on three PDF page targets."""

import argparse
import contextlib
import json
import sys
import time
from types import SimpleNamespace
from pathlib import Path

from part_b_retrieval import query

CASES = [
    {"id": "quadratic-example", "question": "按二次探测依次把 89、18、49、58、79 插入大小为 10 的空哈希表，哪些槽位有值？", "expected_page": "25"},
    {"id": "quadratic-offsets", "question": "What are the first four offsets used in quadratic probing?", "expected_page": "24"},
    {"id": "double-hashing", "question": "How does double hashing resolve collisions using a second hash function?", "expected_page": "30"},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    all_cases = []
    for case in CASES:
        item = dict(case)
        item["modes"] = {}
        for mode in ("vector", "hybrid"):
            config = SimpleNamespace(index_dir=args.index_dir, env_file=args.env_file,
                                     file="11-hash2.pdf", question=case["question"],
                                     top_k=args.top_k, mode=mode)
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                result = query(config)
            elapsed = time.perf_counter() - started
            pages = [r["page"] for r in result["results"]]
            ids = [r["id"] for r in result["results"]]
            item["modes"][mode] = {
                "pages": pages,
                "expected_rank": pages.index(case["expected_page"]) + 1 if case["expected_page"] in pages else None,
                "duplicate_ids": len(ids) - len(set(ids)),
                "elapsed_seconds": round(elapsed, 3),
                "results": result["results"],
            }
        all_cases.append(item)
    print(json.dumps({"top_k": args.top_k, "file": "11-hash2.pdf", "cases": all_cases},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
