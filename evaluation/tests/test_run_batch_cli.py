from __future__ import annotations

import json
from pathlib import Path

from evaluation.batch import config_fingerprint
from evaluation.scripts import run_batch_e2e

DATASET = (
    '{"id": "c1", "category": "direct", "question_type": "answerable", '
    '"question": "q1", "reference_answer": "a1", "expected_files": ["f.pdf"], '
    '"gold_evidence": [{"file": "f.pdf", "locator": {"kind": "page", "value": 1}, '
    '"group": "g", "quote": "q"}]}\n'
)

RETRIEVAL = {
    "simulated": True,
    "index_dir": None,
    "chunking": {"chunk_size": 1024, "chunk_overlap": 256},
    "embedding": {"model": "text-embedding-v4", "dimensions": 2048},
    "mode": "vector",
    "top_k": 5,
    "cases": [
        {
            "case_id": "c1",
            "results": [
                {
                    "chunk_id": "x1",
                    "file": "f.pdf",
                    "locator": {"kind": "page", "value": 1},
                    "score": 0.9,
                    "text": "q",
                }
            ],
        }
    ],
}


def test_rebuild_only_recovers_answers_from_per_case(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(DATASET, encoding="utf-8")
    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(json.dumps(RETRIEVAL), encoding="utf-8")
    out = tmp_path / "out"

    base = [
        "--cases", str(cases),
        "--retrieval", str(retrieval),
        "--out-dir", str(out),
        "--rebuild-only",
    ]
    assert run_batch_e2e.main(base) == 0  # empty run: no records yet

    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    fingerprint = manifest["config_fingerprint"]
    record = {
        "case_id": "c1",
        "status": "ok",
        "simulated": True,
        "config_fingerprint": fingerprint,
        "attempts": 1,
        "category": "direct",
        "question_type": "answerable",
        "question": "q1",
        "answer_text": "a1【1】",
        "evidence_index": [
            {"index": 1, "doc_id": "x1", "file_name": "f.pdf", "page_label": 1}
        ],
        "citation_status": "ok",
        "citation_validation": {"citation_valid": True, "cited_indices": [1]},
        "latency_s": 0.1,
    }
    with (out / "per_case.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    assert run_batch_e2e.main(base) == 0
    answers = json.loads((out / "answers.json").read_text(encoding="utf-8"))
    assert answers["simulated"] is True
    assert set(answers["cases"]) == {"c1"}
    summary = json.loads((out / "report_auto.json").read_text(encoding="utf-8"))
    assert summary["counts"]["ok"] == 1
    assert summary["statuses"]["real_or_simulated"] == "simulated"


def test_manifest_records_hashes_and_no_secrets(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(DATASET, encoding="utf-8")
    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(json.dumps(RETRIEVAL), encoding="utf-8")
    out = tmp_path / "out"
    run_batch_e2e.main(
        [
            "--cases", str(cases),
            "--retrieval", str(retrieval),
            "--out-dir", str(out),
            "--rebuild-only",
        ]
    )
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["secrets_recorded"] is False
    assert manifest["dataset"]["sha256"]
    assert manifest["retrieval"]["sha256"]
    assert "sk-" not in json.dumps(manifest)
    assert manifest["config_fingerprint"] == config_fingerprint(manifest["config"])
