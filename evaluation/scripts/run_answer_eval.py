"""Aggregate answer, refusal and citation metrics from recorded predictions.

Predictions come from Part C's pipeline (written to JSONL by whoever runs it) or
from a fixture. This script never calls a model: it only joins
``predictions + human labels -> metrics``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.metrics import (
    answer_correctness,
    evaluate_citations,
    evaluate_refusal,
)
from evaluation.scripts.validate_dataset import load_cases

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = REPO_ROOT / "evaluation" / "datasets" / "qa_cases.jsonl"
DEFAULT_OUT = REPO_ROOT / "evaluation" / "results" / "answer_report.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON ({exc})") from exc
    return records


def index_by_case(
    records: list[dict[str, Any]], key: str = "case_id"
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in records:
        case_id = str(record.get(key, ""))
        if case_id in result:
            raise SystemExit(f"duplicate {key} {case_id!r}")
        result[case_id] = record
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="human answer labels (JSONL)")
    parser.add_argument(
        "--citation-labels", type=Path, help="per-citation support labels"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    predictions = index_by_case(load_jsonl(args.predictions))
    labels = load_jsonl(args.labels) if args.labels else []
    citation_labels = load_jsonl(args.citation_labels) if args.citation_labels else []

    payload = {
        "config": {
            "cases": str(args.cases),
            "predictions": str(args.predictions),
            "labels": str(args.labels) if args.labels else None,
            "citation_labels": (
                str(args.citation_labels) if args.citation_labels else None
            ),
        },
        "answer_correctness": answer_correctness(labels),
        "refusal": evaluate_refusal(cases, predictions),
        "citations": evaluate_citations(cases, predictions, citation_labels),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    correctness = payload["answer_correctness"]
    refusal = payload["refusal"]
    citations = payload["citations"]
    print(
        "answer accuracy="
        f"{correctness['accuracy']} (judged {correctness['cases_judged']}), "
        f"refusal_accuracy={refusal['refusal_accuracy']}, "
        f"false_refusal_rate={refusal['false_refusal_rate']}, "
        f"citation_validity={citations['citation_validity_rate']}, "
        f"citation_support={citations['citation_support_rate']} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
