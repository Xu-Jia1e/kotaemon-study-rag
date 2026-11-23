from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.matrix import DATASET_PATH
from evaluation.scripts import make_simulated_fixture, run_experiment


def _generate(tmp_path: Path) -> Path:
    fixture = tmp_path / "fixture"
    code = make_simulated_fixture.main(["--out-dir", str(fixture)])
    assert code == 0
    return fixture


def test_simulated_run_produces_all_artefacts(tmp_path):
    fixture = _generate(tmp_path)
    out = tmp_path / "out"
    code = run_experiment.main(
        [
            "--matrix",
            "--simulated-fixture",
            str(fixture),
            "--out-dir",
            str(out),
            "--k",
            "3",
        ]
    )
    assert code == 0

    run_dir = out / "simulated-k3"
    per_case = run_dir / "per_case.jsonl"
    report_json = run_dir / "report.json"
    report_md = run_dir / "report.md"
    for path in (per_case, report_json, report_md):
        assert path.is_file(), path

    records = [
        json.loads(line) for line in per_case.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 24
    assert all(record["simulated"] is True for record in records)
    assert all(
        record["chunking_strategy"].startswith("SIMULATED") for record in records
    )
    assert all(record["top_k"] == 3 for record in records)
    required = {
        "case_id",
        "chunking_strategy",
        "retrieval_method",
        "top_k",
        "retrieved_evidence",
        "hit_at_k",
        "reciprocal_rank",
        "generated_answer",
        "answer_correct",
        "citation_valid",
        "citation_supported",
        "refusal_correct",
    }
    assert required <= set(records[0])

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["report"]["simulated"] is True
    assert payload["cell"]["dataset_sha256"]
    assert payload["report"]["overall"]["cases"] == 24

    markdown = report_md.read_text(encoding="utf-8")
    assert "SIMULATED" in markdown
    assert "NOT experimental results" in markdown
    for label in (
        "Hit@K",
        "MRR",
        "Recall@K",
        "Answer Correctness",
        "Citation Validity",
        "Citation Support",
        "Refusal Accuracy",
    ):
        assert label in markdown


def test_matrix_status_has_no_ready_cell(tmp_path):
    fixture = _generate(tmp_path)
    out = tmp_path / "out"
    run_experiment.main(
        ["--matrix", "--simulated-fixture", str(fixture), "--out-dir", str(out)]
    )
    status = json.loads((out / "matrix_status.json").read_text(encoding="utf-8"))
    assert status["summary"]["cells_total"] == 18
    assert status["summary"]["by_status"] == {"blocked": 18}
    assert status["summary"]["ready_cell_ids"] == []
    assert all(cell["status"] == "blocked" for cell in status["cells"])
    markdown = (out / "matrix_status.md").read_text(encoding="utf-8")
    assert "不是实验结果" in markdown


def test_fixture_rejects_unmarked_data(tmp_path):
    fixture = tmp_path / "fake"
    fixture.mkdir()
    (fixture / "meta.json").write_text(
        json.dumps({"simulated": False}), encoding="utf-8"
    )
    for name in ("rankings", "predictions", "answer_labels", "citation_labels"):
        (fixture / f"{name}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_experiment.run_simulated(
            DATASET_PATH,
            fixture,
            tmp_path / "out",
            5,
        )


def test_hit_rate_changes_with_k(tmp_path):
    fixture = _generate(tmp_path)
    hits = {}
    for k in (3, 5):
        result = run_experiment.run_simulated(
            DATASET_PATH,
            fixture,
            tmp_path / f"k{k}",
            k,
        )
        hits[k] = result["report"]["overall"]["hit_at_k"]
    # Gold evidence is placed at ranks 1..5, so k=5 can only do better.
    assert hits[5] >= hits[3]
    assert 0.0 < hits[3] <= 1.0
