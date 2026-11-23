from __future__ import annotations

from evaluation.metrics.content import content_report, match_gold_units

CASES = [
    {
        "id": "a1",
        "question_type": "answerable",
        "gold_evidence": [
            {"file": "f.pdf", "locator": {"kind": "page", "value": 1}, "group": "g1", "quote": "alpha beta"}
        ],
    },
    {
        "id": "a2",
        "question_type": "answerable",
        "gold_evidence": [
            {"file": "f.pdf", "locator": {"kind": "table_row", "value": "row"}, "group": "g1", "quote": "gamma delta"},
            {"file": "f.pdf", "locator": {"kind": "page", "value": 3}, "group": "g2", "quote": "missing quote"},
        ],
    },
    {
        "id": "u1",
        "question_type": "unanswerable",
        "gold_evidence": [],
    },
]

RANKINGS = {
    "a1": [
        {"chunk_id": "c1", "file": "other.pptx", "locator": {"kind": "slide", "value": 9}, "text": "Alpha   Beta appears here"}
    ],
    "a2": [
        {"chunk_id": "c2", "file": "f.pdf", "locator": {"kind": "page", "value": 7}, "text": "gamma delta ..."},
        {"chunk_id": "c3", "file": "f.pdf", "locator": {"kind": "page", "value": 8}, "text": "unrelated"},
    ],
}


def test_cross_file_content_match_is_allowed():
    units = match_gold_units(CASES[0]["gold_evidence"], RANKINGS["a1"], 5)
    assert units[0]["matched"] is True
    assert units[0]["matched_chunk"]["file"] == "other.pptx"


def test_partial_gold_coverage():
    units = match_gold_units(CASES[1]["gold_evidence"], RANKINGS["a2"], 5)
    assert units[0]["matched"] is True
    assert units[1]["matched"] is False


def test_report_skips_unanswerable_and_aggregates():
    report = content_report(CASES, RANKINGS, k=5)
    assert report["cases_scored"] == 2
    assert report["content_hit_at_k"] == 1.0
    assert report["content_recall_at_k"] == round((1.0 + 0.5) / 2, 6)
    assert {row["case_id"] for row in report["per_case"]} == {"a1", "a2"}
