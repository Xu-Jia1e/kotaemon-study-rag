from __future__ import annotations

import pytest

from evaluation.metrics.citation import evaluate_citations
from evaluation.reporting import aggregate, build_case_records

CASES = [
    {
        "id": "a",
        "category": "direct",
        "question_type": "answerable",
        "gold_evidence": [
            {"file": "f.pdf", "locator": {"kind": "page", "value": 1}, "group": "g"}
        ],
    }
]
RANKINGS = {
    "a": [{"chunk_id": "d1", "file": "f.pdf", "locator": {"kind": "page", "value": 1}}]
}
PREDICTIONS = {
    "a": {
        "text": "答案【1】",
        "evidence_index": [
            {"index": 1, "doc_id": "d1", "file_name": "f.pdf", "page_label": 1}
        ],
    }
}


def make_labels(supported: int, unsupported: int, uncertain: int) -> list[dict]:
    values = (
        ["supported"] * supported
        + ["unsupported"] * unsupported
        + ["uncertain"] * uncertain
    )
    return [
        {"case_id": "a", "citation_index": index, "supported": value}
        for index, value in enumerate(values, start=1)
    ]


def both_rates(labels: list[dict]) -> tuple[dict, dict]:
    citations = evaluate_citations(CASES, PREDICTIONS, labels)
    records = build_case_records(
        CASES,
        RANKINGS,
        PREDICTIONS,
        {},
        {"a": labels},
        chunking_strategy="512-128",
        retrieval_method="vector",
        top_k=5,
    )
    return citations, aggregate(records)["overall"]


def test_uncertain_is_excluded_from_denominator():
    citations, overall = both_rates(make_labels(46, 0, 4))
    assert citations["citation_support_rate"] == 1.0
    assert citations["supported_count"] == 46
    assert citations["unsupported_count"] == 0
    assert citations["uncertain_count"] == 4
    assert citations["assessable_count"] == 46
    assert citations["total_citations"] == 50
    assert overall["citation_support_rate"] == 1.0
    assert overall["assessable_count"] == 46
    assert overall["total_citations"] == 50


def test_unsupported_is_counted_only_in_denominator():
    citations, overall = both_rates(make_labels(46, 2, 4))
    assert citations["citation_support_rate"] == round(46 / 48, 6)
    assert citations["supported_count"] == 46
    assert citations["unsupported_count"] == 2
    assert citations["uncertain_count"] == 4
    assert citations["assessable_count"] == 48
    assert citations["total_citations"] == 52
    assert overall["citation_support_rate"] == round(46 / 48, 6)


def test_all_uncertain_gives_none_not_zero_or_one():
    citations, overall = both_rates(make_labels(0, 0, 4))
    assert citations["citation_support_rate"] is None
    assert citations["assessable_count"] == 0
    assert citations["total_citations"] == 4
    assert overall["citation_support_rate"] is None
    assert overall["assessable_count"] == 0


def test_legacy_boolean_labels_keep_same_rate():
    labels = [
        {"case_id": "a", "citation_index": index, "supported": index <= 49}
        for index in range(1, 52)
    ]
    citations, overall = both_rates(labels)
    assert citations["citation_support_rate"] == 0.960784
    assert citations["supported_count"] == 49
    assert citations["unsupported_count"] == 2
    assert citations["uncertain_count"] == 0
    assert overall["citation_support_rate"] == 0.960784


def test_evaluate_and_reporting_agree_on_tristate():
    labels = make_labels(46, 2, 4)
    citations, overall = both_rates(labels)
    assert citations["citation_support_rate"] == overall["citation_support_rate"]
    assert citations["supported_count"] == overall["supported_count"]
    assert citations["unsupported_count"] == overall["unsupported_count"]
    assert citations["uncertain_count"] == overall["uncertain_count"]


def test_invalid_support_label_raises_clear_error():
    labels = [{"case_id": "a", "citation_index": 1, "supported": "maybe"}]
    with pytest.raises(ValueError, match="invalid citation support label"):
        evaluate_citations(CASES, PREDICTIONS, labels)
    with pytest.raises(ValueError, match="invalid citation support label"):
        build_case_records(
            CASES,
            RANKINGS,
            PREDICTIONS,
            {},
            {"a": labels},
            chunking_strategy="512-128",
            retrieval_method="vector",
            top_k=5,
        )
