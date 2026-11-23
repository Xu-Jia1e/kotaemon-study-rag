from __future__ import annotations

import json
from pathlib import Path

from evaluation.scripts import run_answer_eval, run_retrieval_eval

FIXTURES = Path(__file__).resolve().parent / "fixtures"
# These smoke tests use the deprecated v1 dataset on purpose: the fixtures are
# keyed to its sc-* ids, and v2's real evidence must not be mixed with stubs.
V1_CASES = Path(__file__).resolve().parents[1] / "datasets" / "qa_cases.jsonl"


def test_retrieval_eval_runs_end_to_end_with_fixture(tmp_path):
    out = tmp_path / "retrieval.json"
    code = run_retrieval_eval.main(
        [
            "--cases",
            str(V1_CASES),
            "--adapter",
            "fixture",
            "--stub-fixture",
            str(FIXTURES / "rankings_stub.json"),
            "--k",
            "5",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    assert metrics["cases_scored"] == 6
    # Only sc-01 (rank 2) and sc-05 (both groups) are covered by the fixture.
    assert metrics["hit_at_k"] == round(2 / 6, 6)
    assert metrics["per_case"][0]["case_id"] == "sc-01"
    assert metrics["per_case"][0]["first_rank"] == 2
    sc05 = next(row for row in metrics["per_case"] if row["case_id"] == "sc-05")
    assert sc05["evidence_recall"] == 1.0


def test_answer_eval_aggregates_predictions_and_labels(tmp_path):
    out = tmp_path / "answer.json"
    code = run_answer_eval.main(
        [
            "--cases",
            str(V1_CASES),
            "--predictions",
            str(FIXTURES / "predictions_stub.jsonl"),
            "--labels",
            str(FIXTURES / "labels_stub.jsonl"),
            "--citation-labels",
            str(FIXTURES / "citation_labels_stub.jsonl"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["answer_correctness"]["cases_judged"] == 2
    assert payload["answer_correctness"]["accuracy"] == 0.5
    # sc-05 is an answerable case answered with an abstention -> false refusal.
    # Only sc-01 and sc-05 have predictions, so the rate is 1 of 2 answerable.
    assert payload["refusal"]["answerable_total"] == 2
    assert payload["refusal"]["false_refusal_rate"] == 0.5
    assert payload["refusal"]["refusal_accuracy"] is None
    assert payload["refusal"]["missing_predictions"] == [
        "sc-02",
        "sc-03",
        "sc-04",
        "sc-06",
    ]
    assert payload["citations"]["citation_validity_rate"] == 1.0
    assert payload["citations"]["citation_support_rate"] == 0.5
