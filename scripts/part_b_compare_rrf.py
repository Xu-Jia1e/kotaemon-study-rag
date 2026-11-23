"""Compare vector, original hybrid, and RRF on fixed page-level cases."""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from part_b_retrieval import query

# Fixed before the experiment; each target page contains the requested evidence.
CASES = [
    {"id": "quadratic-example", "file": "11-hash2.pdf", "question": "按二次探测依次把 89、18、49、58、79 插入大小为 10 的空哈希表，哪些槽位有值？", "expected_page": "25"},
    {"id": "quadratic-offsets", "file": "11-hash2.pdf", "question": "What are the first four offsets used in quadratic probing?", "expected_page": "24"},
    {"id": "double-hashing", "file": "11-hash2.pdf", "question": "How does double hashing resolve collisions using a second hash function?", "expected_page": "30"},
    {"id": "lazy-deletion", "file": "11-hash2.pdf", "question": "Why does open addressing require lazy deletion rather than simply removing a key?", "expected_page": "18"},
    {"id": "ustc-login", "file": "USTC-ClassIn.pdf", "question": "瀚海教学网如何登录并激活待激活课程？", "expected_page": "3"},
    {"id": "ustc-formats", "file": "USTC-ClassIn.pdf", "question": "ClassIn 云盘支持哪些课件文件格式？", "expected_page": "8"},
    {"id": "ustc-sharing", "file": "USTC-ClassIn.pdf", "question": "老师如何通过教学工具箱共享电脑屏幕？", "expected_page": "10"},
    {"id": "ustc-groups", "file": "USTC-ClassIn.pdf", "question": "课堂最多可以分多少小组，教师能否旁听小组讨论？", "expected_page": "14"},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    cases = []
    for case in CASES:
        item = dict(case)
        item["modes"] = {}
        for mode in ("vector", "hybrid", "rrf"):
            config = SimpleNamespace(index_dir=args.index_dir, env_file=args.env_file,
                                     file=case["file"], question=case["question"],
                                     top_k=args.top_k, mode=mode)
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                result = query(config)
            pages = [r["page"] for r in result["results"]]
            ids = [r["id"] for r in result["results"]]
            item["modes"][mode] = {
                "pages": pages,
                "expected_rank": pages.index(case["expected_page"]) + 1 if case["expected_page"] in pages else None,
                "duplicate_ids": len(ids) - len(set(ids)),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "results": result["results"],
                "rrf": result.get("rrf"),
            }
        cases.append(item)
    summary = {}
    for mode in ("vector", "hybrid", "rrf"):
        ranks = [case["modes"][mode]["expected_rank"] for case in cases]
        summary[mode] = {"hits": sum(r is not None for r in ranks), "total": len(ranks),
                         "top1_hits": sum(r == 1 for r in ranks),
                         "mrr": round(sum(1 / r for r in ranks if r is not None) / len(ranks), 4),
                         "duplicate_ids": sum(case["modes"][mode]["duplicate_ids"] for case in cases)}
    print(json.dumps({"top_k": args.top_k, "cases": cases, "summary": summary},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
