from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.scripts import run_batch_retrieval

DATASET = [
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
                "quote": "q",
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
                "quote": "q",
            }
        ],
    },
]

INDEX_MANIFEST = {
    "chunk_size": 1024,
    "chunk_overlap": 256,
    "embedding": {"model": "text-embedding-v4", "dimensions": 2048},
    "vector_count": 2,
}


class Item:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id

    def to_dict(self) -> dict:
        return {
            "chunk_id": f"x-{self.case_id}",
            "file": "f.pdf",
            "locator": {"kind": "page", "value": 1},
            "score": 0.9,
            "text": "q",
        }


class FakeAdapter:
    calls: list[str] = []
    fail_on_second_call = False

    def __init__(self, index_dir, env_file=None):
        self.index_dir = index_dir

    def retrieve(self, question, top_k=5, case=None):
        type(self).calls.append(question)
        if type(self).fail_on_second_call and len(type(self).calls) == 2:
            raise KeyboardInterrupt()
        return [Item(case["id"] if case else "unknown")]


@pytest.fixture()
def workspace(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in DATASET),
        encoding="utf-8",
    )
    index = tmp_path / "index"
    index.mkdir()
    (index / "index.json").write_text(
        json.dumps(INDEX_MANIFEST), encoding="utf-8"
    )
    return cases, index, tmp_path / "out"


def _argv(cases, index, out, *extra):
    return [
        "--cases", str(cases),
        "--index-dir", str(index),
        "--out-dir", str(out),
        "--tag", "t",
        *extra,
    ]


def test_interrupt_marks_manifest_interrupted(workspace, monkeypatch):
    cases, index, out = workspace
    FakeAdapter.calls = []
    FakeAdapter.fail_on_second_call = True
    monkeypatch.setattr(
        "evaluation.adapters.b_a_index.AIndexVectorAdapter", FakeAdapter
    )
    with pytest.raises(KeyboardInterrupt):
        run_batch_retrieval.main(_argv(cases, index, out))

    manifest = json.loads((out / "t" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    summary = json.loads((out / "t" / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["ok"] == 1
    assert summary["counts"]["missing"] == 1


def test_resume_after_interrupt_runs_only_missing(workspace, monkeypatch):
    cases, index, out = workspace
    FakeAdapter.calls = []
    FakeAdapter.fail_on_second_call = True
    monkeypatch.setattr(
        "evaluation.adapters.b_a_index.AIndexVectorAdapter", FakeAdapter
    )
    with pytest.raises(KeyboardInterrupt):
        run_batch_retrieval.main(_argv(cases, index, out))

    FakeAdapter.fail_on_second_call = False
    FakeAdapter.calls = []
    run_batch_retrieval.main(_argv(cases, index, out, "--resume"))
    assert FakeAdapter.calls == ["q2"]
    summary = json.loads((out / "t" / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"] == {"total": 2, "ok": 2, "failed": 0, "missing": 0}
    assert summary["run"]["skipped"] == 1


def test_rebuild_only_uses_no_adapter(workspace, monkeypatch):
    cases, index, out = workspace
    FakeAdapter.calls = []
    FakeAdapter.fail_on_second_call = False
    monkeypatch.setattr(
        "evaluation.adapters.b_a_index.AIndexVectorAdapter", FakeAdapter
    )
    run_batch_retrieval.main(_argv(cases, index, out))

    FakeAdapter.calls = []
    run_batch_retrieval.main(_argv(cases, index, out, "--rebuild-only"))
    assert FakeAdapter.calls == []
    payload = json.loads((out / "t" / "retrieval.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["cases_scored"] == 2
    assert payload["simulated"] is False
