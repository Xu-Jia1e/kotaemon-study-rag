from __future__ import annotations

from evaluation.metrics.retrieval import (
    evaluate_retrieval,
    evidence_matches,
    group_units,
    hit_at_k,
    mrr_at_k,
)


def unit(kind: str, value, file: str = "a.pdf", group: str | None = None):
    payload = {"file": file, "locator": {"kind": kind, "value": value}}
    if group:
        payload["group"] = group
    return payload


def item(kind: str, value, file: str = "a.pdf", chunk_id: str | None = None):
    return {
        "chunk_id": chunk_id,
        "file": file,
        "locator": {"kind": kind, "value": value},
    }


def test_page_locator_matches_int_and_page_label_text():
    assert evidence_matches(unit("page", 25), item("page_label", 25))
    assert evidence_matches(unit("page", 25), item("page", "Page 25"))
    assert evidence_matches(unit("page", 25), item("page", "第 25 页"))
    assert not evidence_matches(unit("page", 25), item("page", 26))


def test_file_must_match_and_ignore_directory():
    gold = unit("page", 25, file="a.pdf")
    assert evidence_matches(gold, item("page", 25, file="dir/a.pdf"))
    assert not evidence_matches(gold, item("page", 25, file="b.pdf"))


def test_slide_and_page_are_not_interchangeable():
    assert not evidence_matches(unit("slide", 25), item("page", 25))


def test_alternative_units_in_one_group_count_once():
    gold = [
        unit("slide", 26, "talk.pptx", "guarantee"),
        unit("slide", 27, "talk.pptx", "guarantee"),
    ]
    assert group_units(gold) == {"guarantee": [0, 1]}
    hits = [item("slide", 27, "talk.pptx")]
    assert hit_at_k(gold, hits, k=5) is True
    ranks = evaluate_retrieval(
        [{"id": "c", "question_type": "answerable", "gold_evidence": gold}],
        {"c": hits},
        k=5,
    )
    assert ranks.per_case[0].evidence_recall == 1.0


def test_two_required_groups_give_partial_recall():
    gold = [
        unit("slide", 4, "talk.pptx", "tab"),
        unit("slide", 17, "talk.pptx", "banded"),
    ]
    hits = [item("slide", 4, "talk.pptx")]
    report = evaluate_retrieval(
        [{"id": "c", "question_type": "answerable", "gold_evidence": gold}],
        {"c": hits},
        k=5,
    )
    case = report.per_case[0]
    assert case.hit is True
    assert case.matched_groups == 1 and case.total_groups == 2
    assert case.evidence_recall == 0.5


def test_hit_and_mrr_respect_top_k_cutoff():
    gold = [unit("page", 25)]
    hits = [item("page", 17), item("page", 18), item("page", 25)]
    assert hit_at_k(gold, hits, k=2) is False
    assert hit_at_k(gold, hits, k=3) is True
    assert mrr_at_k(gold, hits, k=3) == 1 / 3
    assert mrr_at_k(gold, hits, k=2) == 0.0


def test_chunk_id_short_circuits_when_both_sides_have_one():
    gold = {
        "file": "a.pdf",
        "locator": {"kind": "page", "value": 25},
        "chunk_id": "c-1",
    }
    assert evidence_matches(gold, item("page", 99, chunk_id="c-1"))
    assert not evidence_matches(gold, item("page", 25, chunk_id="c-2"))


def test_evaluate_retrieval_skips_unanswerable_and_missing_gold():
    cases = [
        {"id": "a", "question_type": "answerable", "gold_evidence": [unit("page", 1)]},
        {"id": "b", "question_type": "unanswerable", "gold_evidence": []},
        {"id": "c", "question_type": "answerable", "gold_evidence": []},
    ]
    rankings = {"a": [item("page", 1)], "b": [], "c": []}
    report = evaluate_retrieval(cases, rankings, k=3)
    assert report.cases_scored == 1
    assert report.cases_skipped == 1
    assert report.skipped_ids == ["c"]
    assert report.hit_rate == 1.0
    assert report.mrr == 1.0


def test_evaluate_retrieval_rejects_non_positive_k():
    import pytest

    with pytest.raises(ValueError):
        evaluate_retrieval([], {}, k=0)
