from __future__ import annotations

from evaluation.metrics.citation import (
    evaluate_citations,
    validate_answer_citations,
)

EVIDENCE = [
    {"index": 1, "doc_id": "d1", "file_name": "a.pdf", "page_label": 25},
    {"index": 2, "doc_id": "d2", "file_name": "a.pdf", "page_label": 26},
]


def test_valid_single_and_multi_citations():
    assert validate_answer_citations("答案【1】", EVIDENCE)["citation_valid"] is True
    multi = validate_answer_citations("答案【1,2】", EVIDENCE)
    assert multi["cited_indices"] == [1, 2]
    assert multi["invalid_citation_ids"] == []


def test_range_expansion_and_out_of_range_ids():
    ranged = validate_answer_citations("答案【1-3】", EVIDENCE)
    assert ranged["cited_indices"] == [1, 2]
    assert ranged["invalid_citation_ids"] == [3]
    assert ranged["citation_valid"] is False


def test_answer_without_markers_is_invalid():
    result = validate_answer_citations("没有引用的答案", EVIDENCE)
    assert result["citation_valid"] is False
    assert result["unsupported_citations"] == [1, 2]


def test_evaluate_citations_excludes_abstentions_and_missing_evidence():
    cases = [
        {"id": "a", "question_type": "answerable"},
        {"id": "b", "question_type": "unanswerable"},
        {"id": "c", "question_type": "answerable"},
    ]
    predictions = {
        "a": {"text": "答案【1】", "evidence_index": EVIDENCE},
        "b": {
            "text": "Insufficient evidence to answer from the provided materials.",
            "evidence_index": [],
        },
        "c": {"text": "答案没有证据编号", "evidence_index": []},
    }
    labels = [{"case_id": "a", "citation_index": 1, "supported": True}]
    report = evaluate_citations(cases, predictions, labels)
    assert report["cases_scored"] == 1
    assert report["excluded_abstentions"] == 1
    assert report["answers_without_evidence"] == 1
    assert report["citation_validity_rate"] == 1.0
    assert report["citation_support_rate"] == 1.0
    assert report["invalid_citation_id_count"] == 0


def test_invalid_citation_lowers_validity_and_support():
    cases = [{"id": "a", "question_type": "answerable"}]
    predictions = {"a": {"text": "答案【9】", "evidence_index": EVIDENCE}}
    labels = [
        {"case_id": "a", "citation_index": 9, "supported": False},
        {"case_id": "a", "citation_index": 1, "supported": True},
    ]
    report = evaluate_citations(cases, predictions, labels)
    assert report["citation_validity_rate"] == 0.0
    assert report["invalid_citation_id_count"] == 1
    assert report["citation_support_rate"] == 0.5
    assert report["unsupported_citations"] == ["a#9"]


def test_support_rate_is_none_without_labels():
    cases = [{"id": "a", "question_type": "answerable"}]
    predictions = {"a": {"text": "答案【1】", "evidence_index": EVIDENCE}}
    report = evaluate_citations(cases, predictions)
    assert report["citation_support_rate"] is None
