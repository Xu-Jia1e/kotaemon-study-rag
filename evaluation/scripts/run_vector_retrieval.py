"""Real (non-simulated) vector retrieval over A's index, recorded for D's metrics.

Runs: question -> query embedding (Bailian) -> B ``VectorRetrieval`` -> Top-K, for a
small case selection, and writes the raw results plus Hit@K / MRR / Recall@K.

This script calls the real embedding API once per question. It never prints the
key; ``--env-file`` points at the local, git-ignored ``.env``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = REPO_ROOT / "evaluation" / "datasets" / "qa_cases_v2.jsonl"
DEFAULT_OUT = Path(r"E:/dsh_workplace/.eval-data/retrieval-real")


def _load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--case-ids",
        default="v2-d01,v2-s01,v2-l01,v2-u01",
        help="comma separated case ids",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tag", default="1024-256-vector-k5")
    args = parser.parse_args(argv)

    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    from evaluation.adapters.b_a_index import AIndexVectorAdapter
    from evaluation.metrics.retrieval import evaluate_retrieval

    all_cases = _load_cases(args.cases)
    by_id = {case["id"]: case for case in all_cases}
    wanted = [item.strip() for item in args.case_ids.split(",") if item.strip()]
    missing = [case_id for case_id in wanted if case_id not in by_id]
    if missing:
        raise SystemExit(f"unknown case ids: {missing}")
    selected = [by_id[case_id] for case_id in wanted]

    started = time.perf_counter()
    adapter = AIndexVectorAdapter(args.index_dir, env_file=args.env_file)
    rankings: dict[str, list[dict]] = {}
    per_case: list[dict] = []
    for case in selected:
        items = adapter.retrieve(str(case["question"]), top_k=args.top_k, case=case)
        records = [adapter.to_record(item) for item in items]
        rankings[case["id"]] = records
        per_case.append(
            {
                "case_id": case["id"],
                "category": case.get("category"),
                "question": case["question"],
                "top_k": args.top_k,
                "latency_s": adapter.last_latency_s,
                "results": records,
            }
        )

    report = evaluate_retrieval(selected, rankings, k=args.top_k)
    payload = {
        "simulated": False,
        "mode": "vector",
        "index_dir": str(Path(args.index_dir).resolve()),
        "chunking": {
            "chunk_size": adapter.bundle.manifest["chunk_size"],
            "chunk_overlap": adapter.bundle.manifest["chunk_overlap"],
        },
        "embedding": {
            "model": adapter.bundle.manifest["embedding"]["model"],
            "dimensions": adapter.bundle.manifest["embedding"]["dimensions"],
        },
        "top_k": args.top_k,
        "dataset": {
            "path": str(args.cases.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        },
        "cases": per_case,
        "metrics": report.to_dict(),
        "elapsed_s": round(time.perf_counter() - started, 2),
    }

    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for case in per_case:
        top = case["results"][0] if case["results"] else None
        head = (
            f"{top['file']} p{top['locator']['value']} score={top['score']}"
            if top
            else "no results"
        )
        print(
            f"{case['case_id']:7s} {case['category']:12s} "
            f"latency={case['latency_s']}s top1={head}"
        )
    print(
        f"Hit@{args.top_k}={report.hit_rate:.4f} MRR@{args.top_k}={report.mrr:.4f} "
        f"Recall@{args.top_k}={report.evidence_recall:.4f} -> {out_path}"
    )
    adapter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
