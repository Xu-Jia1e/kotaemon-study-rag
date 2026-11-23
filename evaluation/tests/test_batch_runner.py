from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.batch import (
    BatchRunner,
    build_manifest,
    config_fingerprint,
    retry_call,
)

CASES = [
    {
        "id": "c1",
        "category": "direct",
        "question_type": "answerable",
        "question": "q1",
        "reference_answer": "a1",
        "expected_files": ["f.pdf"],
        "gold_evidence": [
            {
                "file": "f.pdf",
                "locator": {"kind": "page", "value": 1},
                "group": "g",
                "quote": "evidence one",
            }
        ],
    },
    {
        "id": "c2",
        "category": "detail",
        "question_type": "answerable",
        "question": "q2",
        "reference_answer": "a2",
        "expected_files": ["f.pdf"],
        "gold_evidence": [
            {
                "file": "f.pdf",
                "locator": {"kind": "page", "value": 2},
                "group": "g",
                "quote": "evidence two",
            }
        ],
    },
    {
        "id": "c3",
        "category": "unanswerable",
        "question_type": "unanswerable",
        "question": "q3",
        "reference_answer": "unknown",
        "expected_files": ["f.pdf"],
        "gold_evidence": [],
    },
]

RETRIEVAL = {
    "simulated": False,
    "index_dir": "index",
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
                    "text": "evidence one",
                }
            ],
        },
        {
            "case_id": "c2",
            "results": [
                {
                    "chunk_id": "x2",
                    "file": "f.pdf",
                    "locator": {"kind": "page", "value": 2},
                    "score": 0.8,
                    "text": "evidence two",
                }
            ],
        },
        {"case_id": "c3", "results": []},
    ],
}

CONFIG = {
    "dataset_sha256": "d" * 16,
    "retrieval_sha256": "r" * 16,
    "index_manifest_sha256": "i" * 16,
    "chunking": "1024-256",
    "top_k": 5,
    "retrieval_method": "vector",
    "simulated": False,
}


class TransientError(Exception):
    status_code = 429


class ScriptedAnswerer:
    """Per-question script: 'ok', an Exception instance, or a callable."""

    def __init__(self, scripts: dict[str, list]) -> None:
        self.scripts = {key: list(value) for key, value in scripts.items()}
        self.calls: list[str] = []

    def __call__(self, question: str, items):
        self.calls.append(question)
        script = self.scripts.get(question, [])
        action = script.pop(0) if script else "ok"
        if isinstance(action, BaseException):
            raise action
        if callable(action):
            return action()
        return _ok_result(question)


def _ok_result(question: str) -> dict:
    case_id = {"q1": "c1", "q2": "c2", "q3": "c3"}[question]
    return {
        "answer_text": f"answer-{case_id}【1】",
        "evidence_index": [
            {"index": 1, "doc_id": f"x{case_id}", "file_name": "f.pdf", "page_label": 1}
        ],
        "citation_status": "ok",
        "citation_validation": {"citation_valid": True, "cited_indices": [1]},
        "latency_s": 0.01,
    }


def _runner(
    tmp_path: Path, answerer, *, config=None, simulated=False, **kwargs
) -> BatchRunner:
    return BatchRunner(
        cases=CASES,
        retrieval_cases={case["case_id"]: case for case in RETRIEVAL["cases"]},
        answer_fn=answerer,
        out_dir=tmp_path / "out",
        config=config or CONFIG,
        simulated=simulated,
        sleep=kwargs.pop("sleep", lambda seconds: None),
        log=lambda message: None,
        **kwargs,
    )


def _latest(runner: BatchRunner) -> dict[str, dict]:
    return runner.latest_records()


def test_failure_isolation_and_rebuild(tmp_path):
    answerer = ScriptedAnswerer({"q2": [ValueError("boom")]})
    runner = _runner(tmp_path, answerer)
    stats = runner.run()
    assert stats == {"total": 3, "ok": 2, "failed": 1, "skipped": 0}

    assert len(runner.read_records()) == 3
    rebuilt = runner.rebuild()
    assert rebuilt == {"ok": 2, "failed": 1, "missing": 0, "unscored": 2}

    answers = json.loads((runner.out_dir / "answers.json").read_text(encoding="utf-8"))
    assert set(answers["cases"]) == {"c1", "c3"}
    failures = json.loads(
        (runner.out_dir / "failures.json").read_text(encoding="utf-8")
    )
    assert failures["cases"]["c2"]["error"]["type"] == "ValueError"
    assert failures["cases"]["c2"]["error"]["retryable"] is False
    predictions = [
        json.loads(line)
        for line in (runner.out_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(predictions) == 2


def test_transient_retry_then_success(tmp_path):
    delays: list[float] = []
    answerer = ScriptedAnswerer(
        {"q2": [TransientError("429"), TransientError("429"), "ok"]}
    )
    runner = _runner(tmp_path, answerer, max_retries=2, backoff_s=0.5)
    runner.sleep = delays.append
    runner.run()

    record = _latest(runner)["c2"]
    assert record["status"] == "ok"
    assert record["attempts"] == 3
    assert delays == [0.5, 1.0]


def test_retry_exhaustion_marks_failed(tmp_path):
    answerer = ScriptedAnswerer({"q2": [TransientError("429")] * 3})
    runner = _runner(tmp_path, answerer, max_retries=2)
    runner.run()
    record = _latest(runner)["c2"]
    assert record["status"] == "failed"
    assert record["attempts"] == 3
    assert record["error"]["retryable"] is True
    assert record["error"]["message"] == "429"


def test_resume_skips_complete_and_retries_failed(tmp_path):
    first = ScriptedAnswerer({"q2": [ValueError("boom")]})
    runner = _runner(tmp_path, first)
    runner.run()
    assert len(first.calls) == 3

    second = ScriptedAnswerer({})
    resumed = _runner(tmp_path, second)
    stats = resumed.run(resume=True)
    assert stats["skipped"] == 2
    assert second.calls == ["q2"]
    assert all(record["status"] == "ok" for record in _latest(resumed).values())
    assert len(resumed.read_records()) == 4


def test_resume_with_changed_config_reruns_everything(tmp_path):
    first = ScriptedAnswerer({})
    runner = _runner(tmp_path, first)
    runner.run()

    changed = dict(CONFIG, top_k=10)
    second = ScriptedAnswerer({})
    resumed = _runner(tmp_path, second, config=changed)
    stats = resumed.run(resume=True)
    assert stats["skipped"] == 0
    assert sorted(second.calls) == ["q1", "q2", "q3"]


def test_interruption_keeps_finished_cases(tmp_path):
    answerer = ScriptedAnswerer({"q2": [KeyboardInterrupt()]})
    runner = _runner(tmp_path, answerer)
    with pytest.raises(KeyboardInterrupt):
        runner.run()

    records = runner.read_records()
    assert [record["case_id"] for record in records] == ["c1"]
    rebuilt = runner.rebuild()
    assert rebuilt["ok"] == 1
    assert rebuilt["missing"] == 2
    answers = json.loads((runner.out_dir / "answers.json").read_text(encoding="utf-8"))
    assert set(answers["cases"]) == {"c1"}


def test_manifest_never_contains_secrets(tmp_path):
    manifest = build_manifest(
        repo_root=tmp_path,
        config={**CONFIG, "api_key": "SECRET-VALUE", "DEEPSEEK_API_KEY": "SECRET"},
        code_files=[__file__],
        dataset_path=__file__,
        index_manifest_path=None,
        retrieval_path=__file__,
        simulated=False,
        status="finished",
        started_at="2026-01-01T00:00:00+08:00",
        finished_at="2026-01-01T00:01:00+08:00",
        counts={"ok": 3},
    )
    text = json.dumps(manifest, ensure_ascii=False)
    assert "SECRET" not in text
    assert manifest["secrets_recorded"] is False
    assert manifest["config_fingerprint"] == config_fingerprint(CONFIG)
    assert manifest["dataset"]["sha256"]
    assert manifest["code"][Path(__file__).name]


def test_unscored_metrics_are_none_not_zero(tmp_path):
    runner = _runner(tmp_path, ScriptedAnswerer({}))
    runner.run()
    runner.rebuild()
    summary = json.loads(
        (runner.out_dir / "report_auto.json").read_text(encoding="utf-8")
    )
    assert summary["counts"]["unscored"] == 3
    assert summary["counts"]["scored"] == 0
    assert summary["report"]["overall"]["answer_correctness"] is None
    assert summary["report"]["overall"]["citation_support"] is None


def test_simulated_flag_propagates(tmp_path):
    runner = _runner(tmp_path, ScriptedAnswerer({}), simulated=True)
    runner.run()
    runner.rebuild()
    answers = json.loads((runner.out_dir / "answers.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (runner.out_dir / "report_auto.json").read_text(encoding="utf-8")
    )
    assert answers["simulated"] is True
    assert summary["simulated"] is True
    assert summary["statuses"]["real_or_simulated"] == "simulated"


def test_retry_call_retries_transient_then_returns():
    delays: list[float] = []
    state = {"calls": 0}

    def action():
        state["calls"] += 1
        if state["calls"] < 3:
            raise TransientError("429")
        return "ok"

    value, attempts = retry_call(
        action, max_retries=2, backoff_s=0.5, sleep=delays.append
    )
    assert value == "ok"
    assert attempts == 3
    assert delays == [0.5, 1.0]


def test_retry_call_does_not_retry_fatal_error():
    state = {"calls": 0}

    def action():
        state["calls"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        retry_call(action, max_retries=3)
    assert state["calls"] == 1


def test_retry_call_exhausts_transient_retries():
    state = {"calls": 0}

    def action():
        state["calls"] += 1
        raise TransientError("503")

    with pytest.raises(TransientError):
        retry_call(action, max_retries=1, sleep=lambda seconds: None)
    assert state["calls"] == 2
