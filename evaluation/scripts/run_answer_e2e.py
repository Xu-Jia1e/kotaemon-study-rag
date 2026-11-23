"""Real end-to-end answer run over B's saved retrieval results (4 cases).

Chain: B ``retrieval.json`` -> C ``PrepareEvidencePipeline`` -> DeepSeek answer +
citation validation -> D records/metrics.

Fields that require human judgement (answer correctness, citation support) are
left ``not_judged``; only automatic fields are computed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = REPO_ROOT / "evaluation" / "datasets" / "qa_cases_v2.jsonl"
DEFAULT_RETRIEVAL = Path(
    r"E:/dsh_workplace/.eval-data/retrieval-real/1024-256-vector-k5/retrieval.json"
)
DEFAULT_OUT = Path(r"E:/dsh_workplace/.eval-data/answer-e2e")


def _load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lang", default="Chinese")
    args = parser.parse_args(argv)

    from evaluation.adapters.c_answer import EvidenceAnswerer
    from evaluation.reporting import aggregate, build_case_records, write_outputs

    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    if retrieval.get("simulated"):
        raise SystemExit("refusing to treat simulated retrieval as real data")
    retrieval_cases = {case["case_id"]: case for case in retrieval["cases"]}

    dataset = {case["id"]: case for case in _load_cases(args.cases)}
    selected = [dataset[case_id] for case_id in retrieval_cases if case_id in dataset]
    if len(selected) != len(retrieval_cases):
        raise SystemExit("retrieval file references cases outside the dataset")

    answerer = EvidenceAnswerer(args.env_file, lang=args.lang)
    started = time.perf_counter()
    predictions: dict[str, dict] = {}
    details: dict[str, dict] = {}
    for case in selected:
        case_id = case["id"]
        items = retrieval_cases[case_id]["results"]
        result = answerer.answer(str(case["question"]), items)
        predictions[case_id] = {
            "text": result["answer_text"],
            "evidence_index": result["evidence_index"],
        }
        details[case_id] = result
        print(
            f"{case_id:7s} latency={result['latency_s']:6.2f}s "
            f"citation_status={result['citation_status']} "
            f"chars={len(result['answer_text'])}"
        )

    rankings = {
        case_id: [
            {
                "chunk_id": item.get("chunk_id"),
                "file": item.get("file"),
                "locator": item.get("locator"),
                "score": item.get("score"),
                "text": item.get("text", ""),
            }
            for item in retrieval_cases[case_id]["results"]
        ]
        for case_id in retrieval_cases
    }

    records = build_case_records(
        selected,
        rankings,
        predictions,
        {},  # answer_correct: human judgement, deliberately absent
        {},  # citation_supported: human judgement, deliberately absent
        chunking_strategy=f"{retrieval['chunking']['chunk_size']}-"
        f"{retrieval['chunking']['chunk_overlap']}",
        retrieval_method=retrieval["mode"],
        top_k=retrieval["top_k"],
        reranking="off",
        simulated=False,
    )
    report = aggregate(records)

    out_dir = args.out_dir / Path(args.retrieval).parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answers.json").write_text(
        json.dumps(
            {
                "simulated": False,
                "llm": {"provider": "deepseek", "model": "deepseek-flash"},
                "retrieval_file": str(args.retrieval),
                "cases": details,
                "elapsed_s": round(time.perf_counter() - started, 2),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (out_dir / "predictions.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for case_id, prediction in predictions.items():
            line = json.dumps(
                {"case_id": case_id, **prediction}, ensure_ascii=False
            )
            handle.write(line + "\n")

    write_outputs(
        {
            "chunking_strategy": f"{retrieval['chunking']['chunk_size']}-"
            f"{retrieval['chunking']['chunk_overlap']}",
            "retrieval_method": retrieval["mode"],
            "top_k": retrieval["top_k"],
            "reranking": "off",
            "simulated": False,
            "dataset_path": retrieval["dataset"]["path"],
            "dataset_sha256": retrieval["dataset"]["sha256"],
        },
        records,
        report,
        out_dir,
    )

    overall = report["overall"]
    print(
        f"\ncases={overall['cases']} "
        f"citation_validity={overall['citation_validity']} "
        f"refusal_accuracy={overall['refusal_accuracy']} "
        f"answer_correctness={overall['answer_correctness']} "
        f"citation_support={overall['citation_support']}"
    )
    print(f"-> {out_dir / 'answers.json'}")
    print(f"-> {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
