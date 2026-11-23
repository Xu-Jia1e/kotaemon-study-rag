"""Run the retrieval metrics over a dataset and an adapter.

Offline by default: ``--adapter fixture`` replays a rankings fixture. The
``part-b`` adapter shells out to Part B's CLI, which requires a built index and a
DashScope key in a local ``.env`` (never committed); until A's main index is
queryable, that path stays the experimental one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.adapters import (
    FixtureRetrievalAdapter,
    PartBRetrievalAdapter,
)
from evaluation.adapters.base import RetrievalAdapter
from evaluation.metrics import evaluate_retrieval
from evaluation.scripts.validate_dataset import load_cases

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = REPO_ROOT / "evaluation" / "datasets" / "qa_cases_v2.jsonl"
DEFAULT_OUT = REPO_ROOT / "evaluation" / "results" / "retrieval_report.json"


def build_adapter(args: argparse.Namespace) -> RetrievalAdapter:
    if args.adapter == "fixture":
        if not args.stub_fixture:
            raise SystemExit("--adapter fixture requires --stub-fixture")
        return FixtureRetrievalAdapter.from_file(args.stub_fixture)
    if args.adapter == "part-b":
        if not args.index_dir:
            raise SystemExit("--adapter part-b requires --index-dir")
        return PartBRetrievalAdapter(
            index_dir=args.index_dir,
            env_file=args.env_file,
            mode=args.mode,
            repo_root=REPO_ROOT,
            python=args.python,
        )
    raise SystemExit(f"unknown adapter {args.adapter!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--adapter", choices=["fixture", "part-b"], default="fixture")
    parser.add_argument("--stub-fixture", type=Path)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--mode", choices=["vector", "hybrid", "rrf"], default="vector")
    parser.add_argument("--python", help="interpreter for the part-b subprocess")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--include-texts", action="store_true")
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    adapter = build_adapter(args)

    rankings: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        if case.get("question_type") != "answerable":
            continue
        items = adapter.retrieve(
            str(case["question"]), top_k=args.k, case=case
        )
        rankings[str(case["id"])] = [
            item.to_dict() if args.include_texts else {**item.to_dict(), "text": ""}
            for item in items
        ]

    report = evaluate_retrieval(cases, rankings, k=args.k)
    payload = {
        "adapter": adapter.name,
        "config": {
            "k": args.k,
            "mode": args.mode if args.adapter == "part-b" else None,
            "cases": str(args.cases),
        },
        "metrics": report.to_dict(),
        "rankings": rankings,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"adapter={adapter.name} k={args.k} cases={report.cases_scored} "
        f"Hit@{args.k}={report.hit_rate:.4f} MRR@{args.k}={report.mrr:.4f} "
        f"recall@{args.k}={report.evidence_recall:.4f} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
