"""Generate a deterministic synthetic fixture for pipeline validation.

The output is **simulated data**, never experimental results: it exists so the
matrix -> records -> report path can be exercised without an index, an embedding
key or a language model. Every generated file carries ``"simulated": true`` and
the run script stamps the reports accordingly.

Rankings deliberately place the gold evidence at ranks 1..5 only for *some*
cases (so Hit@3 / Hit@5 / Hit@10 differ), and citations deliberately include
out-of-range markers for every fourth case, so both metric paths are exercised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "evaluation" / "datasets" / "qa_cases_v2.jsonl"
ABSTAIN = "Insufficient evidence to answer from the provided materials."


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _wrong_locator(gold_locator: dict[str, Any], offset: int) -> dict[str, Any]:
    kind = gold_locator.get("kind", "page")
    value = gold_locator.get("value")
    if isinstance(value, int):
        return {"kind": kind, "value": value + 100 + offset}
    return {"kind": kind, "value": f"simulated-{offset}"}


def _item(
    case_id: str,
    index: int,
    file: str,
    locator: dict[str, Any],
    text: str,
) -> dict[str, Any]:
    return {
        "chunk_id": f"sim-{case_id}-{index}",
        "file": file,
        "locator": locator,
        "text": text,
        "score": round(0.9 - 0.01 * index, 4),
    }


def build(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    predictions: dict[str, dict[str, Any]] = {}
    answer_labels: dict[str, str] = {}
    citation_labels: dict[str, list[dict[str, Any]]] = {}

    for index, case in enumerate(cases):
        case_id = str(case["id"])
        file = (case.get("expected_files") or ["unknown"])[0]
        gold = (case.get("gold_evidence") or [None])[0]
        gold_locator = (gold or {}).get("locator") or {"kind": "page", "value": 1}
        gold_rank = (index % 5) + 1

        items: list[dict[str, Any]] = []
        for slot in range(1, 11):
            if gold is not None and slot == gold_rank:
                items.append(
                    _item(
                        case_id,
                        slot,
                        file,
                        dict(gold_locator),
                        "[SIMULATED] gold evidence placeholder",
                    )
                )
            else:
                items.append(
                    _item(
                        case_id,
                        slot,
                        file,
                        _wrong_locator(gold_locator, slot),
                        f"[SIMULATED] distractor chunk {slot}",
                    )
                )
        rankings[case_id] = items

        evidence_index = [
            {
                "index": position,
                "doc_id": item["chunk_id"],
                "file_name": item["file"],
                "page_label": item["locator"].get("value"),
                "section": None,
                "type": "text",
            }
            for position, item in enumerate(items[:5], start=1)
        ]

        if case.get("question_type") == "unanswerable":
            if index % 2 == 0:
                text = ABSTAIN
            else:
                text = "根据资料，答案是 42【1】。"
            predictions[case_id] = {"text": text, "evidence_index": evidence_index}
            answer_labels[case_id] = "not_judged"
            citation_labels[case_id] = []
            continue

        marker = "【9】" if index % 4 == 0 else "【1】"
        predictions[case_id] = {
            "text": f"{case.get('reference_answer', '')}{marker}",
            "evidence_index": evidence_index,
        }
        if gold_rank <= 1:
            answer_labels[case_id] = "correct"
        elif gold_rank <= 3:
            answer_labels[case_id] = "partial"
        else:
            answer_labels[case_id] = "incorrect"
        cited_index = 9 if index % 4 == 0 else 1
        citation_labels[case_id] = [
            {"citation_index": cited_index, "supported": index % 2 == 0}
        ]

    return {
        "meta": {
            "simulated": True,
            "generated_by": "evaluation.scripts.make_simulated_fixture",
            "dataset_sha256": hashlib.sha256(
                DEFAULT_DATASET.read_bytes()
            ).hexdigest(),
            "note": (
                "Synthetic fixture for pipeline validation only. "
                "Not experimental results; never cite these numbers."
            ),
        },
        "rankings": rankings,
        "predictions": predictions,
        "answer_labels": answer_labels,
        "citation_labels": citation_labels,
    }


def write(payload: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in ("meta", "rankings", "predictions", "answer_labels", "citation_labels"):
        path = out_dir / f"{name}.json"
        path.write_text(
            json.dumps(payload[name], ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        paths[name] = path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    cases = load_cases(args.dataset)
    payload = build(cases)
    paths = write(payload, args.out_dir)
    print(
        f"wrote simulated fixture for {len(cases)} cases to {args.out_dir} "
        f"({', '.join(sorted(paths))}) — SIMULATED, not results"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
