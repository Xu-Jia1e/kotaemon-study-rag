"""Experiment driver: matrix status, and a clearly-labelled simulated run.

Two modes:

* ``--matrix``   print/save the availability of every cell. Blocked cells are
  reported with their reason and **no metrics are produced** for them.
* ``--simulated-fixture DIR``  run the full record/report path over synthetic
  fixture data. Output is stamped ``simulated: true`` and the markdown carries a
  SIMULATED banner.

There is deliberately no mode that silently "runs" a blocked cell: while the
index and embedding key are missing, a real result cannot exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from evaluation import matrix as matrix_module
from evaluation import reporting
from evaluation.matrix import ExperimentCell, build_matrix, dataset_manifest, summarise
from evaluation.reporting import build_case_records

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "evaluation" / "results"


def _write_matrix_status(cells: list[ExperimentCell], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_manifest(),
        "summary": summarise(cells),
        "cells": [cell.to_dict() for cell in cells],
        "note": (
            "No cell is ready yet. Missing: a queryable index for both chunking "
            "profiles and a retrieval path that does not need a real embedding API."
        ),
    }
    path = out_dir / "matrix_status.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = out_dir / "matrix_status.md"
    lines = [
        "# 实验矩阵状态",
        "",
        f"- dataset: `{payload['dataset']['path']}` "
        f"({payload['dataset']['cases']} cases, "
        f"sha256 `{payload['dataset']['sha256'][:12]}…`)",
        f"- cells: {payload['summary']['cells_total']}, "
        f"ready: {payload['summary']['by_status'].get('ready', 0)}, "
        f"blocked: {payload['summary']['by_status'].get('blocked', 0)}",
        "",
        matrix_module.status_table(cells),
        "",
        "> 阻塞项不产出任何指标；本文件不是实验结果。",
    ]
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_fixture(fixture_dir: Path) -> tuple[dict[str, Any], dict, dict, dict, dict]:
    def read(name: str) -> Any:
        return json.loads((fixture_dir / name).read_text(encoding="utf-8"))

    meta = read("meta.json")
    if not meta.get("simulated"):
        raise SystemExit(
            f"{fixture_dir} is not marked simulated; refusing to label it as such"
        )
    return (
        meta,
        read("rankings.json"),
        read("predictions.json"),
        read("answer_labels.json"),
        read("citation_labels.json"),
    )


def run_simulated(
    dataset: Path,
    fixture_dir: Path,
    out_dir: Path,
    top_k: int,
) -> dict[str, Any]:
    meta, rankings, predictions, answer_labels, citation_labels = _load_fixture(
        fixture_dir
    )
    cases = reporting.load_jsonl(dataset)
    cell = {
        "chunking_strategy": "SIMULATED-1024-256",
        "retrieval_method": "SIMULATED-vector",
        "top_k": top_k,
        "reranking": "off",
        "simulated": True,
        "dataset_path": str(dataset.relative_to(REPO_ROOT)).replace("\\", "/"),
        "dataset_sha256": meta.get("dataset_sha256", ""),
    }
    records = build_case_records(
        cases,
        rankings,
        predictions,
        answer_labels,
        citation_labels,
        chunking_strategy=cell["chunking_strategy"],
        retrieval_method=cell["retrieval_method"],
        top_k=top_k,
        reranking=cell["reranking"],
        simulated=True,
    )
    report = reporting.aggregate(records)
    paths = reporting.write_outputs(cell, records, report, out_dir)
    return {"cell": cell, "report": report, "paths": paths, "records": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=matrix_module.DATASET_PATH
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--matrix", action="store_true", help="write the matrix status report"
    )
    parser.add_argument(
        "--simulated-fixture",
        type=Path,
        help="directory produced by make_simulated_fixture (pipeline check only)",
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)

    if not args.matrix and not args.simulated_fixture:
        parser.error("choose --matrix and/or --simulated-fixture")

    cells = build_matrix()
    if args.matrix:
        status_path = _write_matrix_status(cells, args.out_dir)
        summary = summarise(cells)
        print(
            f"matrix: {summary['cells_total']} cells, "
            f"ready={summary['by_status'].get('ready', 0)}, "
            f"blocked={summary['by_status'].get('blocked', 0)} -> {status_path}"
        )
        for name, count in summary["blocking_dimensions"].items():
            print(f"  blocked {name} in {count} cell(s)")

    if args.simulated_fixture:
        result = run_simulated(
            args.dataset, args.simulated_fixture, args.out_dir / f"simulated-k{args.k}",
            args.k,
        )
        overall = result["report"]["overall"]
        print(
            f"SIMULATED (not results): k={args.k} cases={overall['cases']} "
            f"Hit@K={overall['hit_at_k']} MRR={overall['mrr_at_k']} "
            f"Recall@K={overall['recall_at_k']}"
        )
        print(
            "  answer_correctness="
            f"{overall['answer_correctness']} "
            f"citation_validity={overall['citation_validity']} "
            f"citation_support={overall['citation_support']} "
            f"refusal_accuracy={overall['refusal_accuracy']}"
        )
        print(f"  -> {result['paths']['report_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
