from __future__ import annotations

from evaluation.reporting import aggregate, build_case_records, render_markdown

CASES = [
    {
        "id": "a1",
        "category": "direct",
        "question_type": "answerable",
        "gold_evidence": [
            {"file": "f.pdf", "locator": {"kind": "page", "value": 5}, "group": "g1"}
        ],
    },
    {
        "id": "a2",
        "category": "synthesis",
        "question_type": "answerable",
        "gold_evidence": [
            {"file": "f.pdf", "locator": {"kind": "page", "value": 6}, "group": "g1"},
            {"file": "f.pdf", "locator": {"kind": "page", "value": 7}, "group": "g2"},
        ],
    },
    {
        "id": "u1",
        "category": "unanswerable",
        "question_type": "unanswerable",
        "gold_evidence": [],
    },
]

RANKINGS = {
    "a1": [
        {"chunk_id": "w1", "file": "f.pdf", "locator": {"kind": "page", "value": 99}},
        {"chunk_id": "g1", "file": "f.pdf", "locator": {"kind": "page", "value": 5}},
    ],
    "a2": [
        {"chunk_id": "g2", "file": "f.pdf", "locator": {"kind": "page", "value": 6}},
    ],
    "u1": [],
}

EVIDENCE_INDEX = [
    {"index": 1, "doc_id": "g1", "file_name": "f.pdf", "page_label": 5},
    {"index": 2, "doc_id": "g2", "file_name": "f.pdf", "page_label": 6},
]

PREDICTIONS = {
    "a1": {"text": "答案【1】", "evidence_index": EVIDENCE_INDEX},
    "a2": {"text": "答案【9】", "evidence_index": EVIDENCE_INDEX},
    "u1": {
        "text": "Insufficient evidence to answer from the provided materials.",
        "evidence_index": [],
    },
}

LABELS = {"a1": "correct", "a2": "partial", "u1": "not_judged"}
CITATION_LABELS = {
    "a1": [{"citation_index": 1, "supported": True}],
    "a2": [{"citation_index": 9, "supported": False}],
    "u1": [],
}

REQUIRED = (
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
)


def build():
    return build_case_records(
        CASES,
        RANKINGS,
        PREDICTIONS,
        LABELS,
        CITATION_LABELS,
        chunking_strategy="1024-256",
        retrieval_method="vector",
        top_k=5,
        simulated=True,
    )


def test_each_record_has_required_fields():
    records = build()
    assert len(records) == 3
    for record in records:
        payload = record.to_dict()
        for key in REQUIRED:
            assert key in payload, key


def test_retrieval_metrics_per_case():
    records = {record.case_id: record for record in build()}
    assert records["a1"].hit_at_k is True
    assert records["a1"].reciprocal_rank == 0.5
    assert records["a1"].recall_at_k == 1.0
    # a2 covers only one of its two evidence groups.
    assert records["a2"].hit_at_k is True
    assert records["a2"].recall_at_k == 0.5
    # Unanswerable questions are not scored for retrieval.
    assert records["u1"].hit_at_k is None


def test_citation_validity_and_support_are_separate():
    records = {record.case_id: record for record in build()}
    assert records["a1"].citation_valid is True
    assert records["a1"].citation_supported == "supported"
    # Index 9 does not exist in the evidence -> invalid, and the human label
    # says it does not support the sentence either.
    assert records["a2"].citation_valid is False
    assert records["a2"].citation_supported == "unsupported"
    # A correct abstention has no citation to check.
    assert records["u1"].citation_valid is None


def test_refusal_correctness_both_directions():
    records = {record.case_id: record for record in build()}
    assert records["u1"].refusal_correct is True
    assert records["a1"].refusal_correct is True
    assert records["a1"].expected_refusal is False


def test_aggregate_overall_and_by_category():
    report = aggregate(build())
    overall = report["overall"]
    assert overall["cases"] == 3
    assert overall["hit_at_k"] == 1.0
    assert overall["mrr_at_k"] == round((0.5 + 1.0) / 2, 6)
    assert overall["recall_at_k"] == round((1.0 + 0.5) / 2, 6)
    assert overall["answer_correctness"] == 0.75
    assert overall["answer_strict_accuracy"] == 0.5
    assert overall["citation_validity"] == 0.5
    assert overall["citation_support"] == 0.5
    assert overall["refusal_accuracy"] == 1.0
    assert overall["false_refusal_rate"] == 0.0

    by_category = report["by_category"]
    assert by_category["direct"]["cases"] == 1
    assert by_category["direct"]["hit_at_k"] == 1.0
    assert by_category["synthesis"]["citation_validity"] == 0.0
    assert by_category["detail"]["cases"] == 0
    assert by_category["detail"]["hit_at_k"] is None
    assert by_category["unanswerable"]["refusal_accuracy"] == 1.0
    assert by_category["unanswerable"]["hit_at_k"] is None


def test_markdown_marks_simulated_runs():
    report = aggregate(build())
    cell = {
        "chunking_strategy": "SIMULATED-1024-256",
        "retrieval_method": "SIMULATED-vector",
        "top_k": 5,
    }
    markdown = render_markdown(report, cell)
    assert "SIMULATED" in markdown
    assert "NOT experimental results" in markdown
    assert "Citation Validity" in markdown
    assert "Citation Support" in markdown


def test_missing_labels_stay_none_not_zero():
    records = build_case_records(
        CASES,
        RANKINGS,
        PREDICTIONS,
        {},
        {},
        chunking_strategy="1024-256",
        retrieval_method="vector",
        top_k=5,
    )
    report = aggregate(records)
    assert report["overall"]["answer_correctness"] is None
    assert report["overall"]["citation_support"] is None
    assert report["overall"]["cases_judged"] == 0
