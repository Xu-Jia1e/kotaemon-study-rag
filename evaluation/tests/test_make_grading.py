from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.scripts import make_grading

CASES = [
    {
        "id": "c1",
        "category": "direct",
        "question_type": "answerable",
        "question": "问题一",
        "reference_answer": "答案一",
        "expected_files": ["f.pdf"],
        "gold_evidence": [
            {
                "file": "f.pdf",
                "locator": {"kind": "page", "value": 1},
                "group": "g",
                "quote": "quote one",
            }
        ],
    },
    {
        "id": "c2",
        "category": "unanswerable",
        "question_type": "unanswerable",
        "question": "问题二",
        "reference_answer": "无法确定",
        "expected_files": ["f.pdf"],
        "gold_evidence": [],
    },
]

RETRIEVAL = {
    "simulated": False,
    "chunking": {"chunk_size": 1024, "chunk_overlap": 256},
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
                    "text": "quote one",
                }
            ],
        },
        {"case_id": "c2", "results": []},
    ],
}

ANSWERS = {
    "simulated": False,
    "cases": {
        "c1": {
            "answer_text": "答案一【1】",
            "evidence_index": [
                {"index": 1, "doc_id": "x1", "file_name": "f.pdf", "page_label": 1}
            ],
            "citation_status": "ok",
            "citation_validation": {"citation_valid": True, "cited_indices": [1]},
            "latency_s": 1.0,
            "attempts": 1,
        },
        "c2": {
            "answer_text": (
                "Insufficient evidence to answer from the provided materials."
            ),
            "evidence_index": [],
            "citation_status": "abstained",
            "citation_validation": None,
            "latency_s": 0.5,
            "attempts": 1,
        },
    },
}


def _fixture(tmp_path: Path, *, drop: str | None = None) -> tuple[Path, Path, Path]:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in CASES),
        encoding="utf-8",
    )
    retrieval_path = tmp_path / "retrieval.json"
    retrieval_path.write_text(
        json.dumps(RETRIEVAL, ensure_ascii=False), encoding="utf-8"
    )
    answers = json.loads(json.dumps(ANSWERS))
    if drop:
        answers["cases"].pop(drop, None)
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
    return cases_path, retrieval_path, answers_path


def _run(
    tmp_path: Path, cases: Path, retrieval: Path, answers: Path, out: Path, **extra
):
    argv = [
        "--answers",
        str(answers),
        "--retrieval",
        str(retrieval),
        "--cases",
        str(cases),
        "--out",
        str(out),
    ]
    argv.extend(extra.pop("flags", []))
    return make_grading.main(argv)


def test_generates_all_cases(tmp_path):
    cases, retrieval, answers = _fixture(tmp_path)
    out = tmp_path / "grading.md"
    assert _run(tmp_path, cases, retrieval, answers, out) == 0
    text = out.read_text(encoding="utf-8")
    assert "## c1（直接事实题 / 有答案）" in text
    assert "## c2（无答案题）" in text
    assert "不适用（无引用）" in text
    assert "☐ 全部支持" in text
    assert text.count("评分汇总") == 1


def test_missing_case_is_marked_not_fabricated(tmp_path):
    cases, retrieval, answers = _fixture(tmp_path, drop="c2")
    out = tmp_path / "grading.md"
    _run(tmp_path, cases, retrieval, answers, out)
    text = out.read_text(encoding="utf-8")
    assert "## c2（未作答）" in text
    assert "本次不生成评分项" in text


def test_refuses_to_overwrite_without_force(tmp_path):
    cases, retrieval, answers = _fixture(tmp_path)
    out = tmp_path / "grading.md"
    out.write_text("existing", encoding="utf-8")
    with pytest.raises(SystemExit):
        _run(tmp_path, cases, retrieval, answers, out)
    assert out.read_text(encoding="utf-8") == "existing"

    _run(tmp_path, cases, retrieval, answers, out, flags=["--force"])
    assert "评分汇总" in out.read_text(encoding="utf-8")


def test_pilot_grading_file_untouched(tmp_path):
    pilot = Path(
        r"E:/dsh_workplace/.eval-data/answer-e2e/1024-256-vector-k5/grading.md"
    )
    before = hashlib.sha256(pilot.read_bytes()).hexdigest() if pilot.is_file() else None

    cases, retrieval, answers = _fixture(tmp_path)
    _run(tmp_path, cases, retrieval, answers, tmp_path / "grading.md")

    after = hashlib.sha256(pilot.read_bytes()).hexdigest() if pilot.is_file() else None
    assert before == after


def test_case_ids_subset(tmp_path):
    cases, retrieval, answers = _fixture(tmp_path)
    out = tmp_path / "grading.md"
    _run(tmp_path, cases, retrieval, answers, out, flags=["--case-ids", "c1"])
    text = out.read_text(encoding="utf-8")
    assert "## c1" in text
    assert "## c2" not in text
