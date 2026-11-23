from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.scripts.validate_dataset import load_cases, validate_cases

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "qa_cases_v2.jsonl"
SAMPLES = Path(__file__).resolve().parents[2] / "downloads" / "real-samples"


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    return load_cases(DATASET)


def test_dataset_has_30_cases_and_requested_distribution(cases):
    assert len(cases) == 30
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    assert counts == {"direct": 12, "synthesis": 8, "detail": 5, "unanswerable": 5}


def test_ids_are_unique_and_prefixed(cases):
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    assert all(case_id.startswith("v2-") for case_id in ids)


def test_dataset_passes_schema_validation(cases):
    errors, warnings = validate_cases(cases)
    assert errors == []
    assert warnings == []


def test_answerable_cases_have_verified_quotes(cases):
    for case in cases:
        if case["question_type"] != "answerable":
            continue
        assert case["gold_evidence"], case["id"]
        assert case["evidence_status"] == "quote_verified"
        for unit in case["gold_evidence"]:
            assert unit["quote"], f"{case['id']} has an empty quote"
            assert unit["quote_status"] == "verified", case["id"]
            assert unit["file"] in case["expected_files"], case["id"]


def test_synthesis_cases_require_multiple_evidence_groups(cases):
    for case in cases:
        if case["category"] != "synthesis":
            continue
        groups = {unit["group"] for unit in case["gold_evidence"]}
        assert len(groups) >= 2, f"{case['id']} has only one evidence group"
        assert len(case["gold_evidence"]) >= 2, case["id"]


def test_unanswerable_cases_document_their_scope(cases):
    for case in cases:
        if case["question_type"] != "unanswerable":
            continue
        assert case["gold_evidence"] == []
        scope = case["checked_scope"]
        assert len(scope["files"]) == 4
        assert scope["method"]
        assert any(
            scope.get(key) for key in ("absent_phrases", "max_mentions", "blank_fields")
        ), case["id"]


def test_source_field_points_at_a_locator(cases):
    for case in cases:
        if case["question_type"] != "answerable":
            continue
        assert "=" in case["source"], case["id"]


@pytest.mark.skipif(
    not SAMPLES.is_dir(), reason="source documents not downloaded (offline)"
)
def test_quotes_really_appear_in_the_documents(cases):
    from evaluation.scripts.verify_evidence import build_index, verify_case

    index, missing = build_index(SAMPLES)
    assert not missing, f"documents missing: {missing}"
    problems: list[str] = []
    for case in cases:
        if case["question_type"] == "answerable":
            problems.extend(verify_case(case, index))
    assert problems == []


@pytest.mark.skipif(
    not SAMPLES.is_dir(), reason="source documents not downloaded (offline)"
)
def test_unanswerable_absence_checks_hold(cases):
    from evaluation.scripts.verify_evidence import build_index, verify_unanswerable

    index, _ = build_index(SAMPLES)
    problems: list[str] = []
    for case in cases:
        if case["question_type"] == "unanswerable":
            problems.extend(verify_unanswerable(case, index))
    assert problems == []


def test_dataset_is_valid_jsonl_with_newline_termination():
    raw = DATASET.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    for line in raw.splitlines():
        assert json.loads(line)["id"]
