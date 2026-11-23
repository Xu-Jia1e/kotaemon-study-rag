from __future__ import annotations

from evaluation.metrics.answer import (
    answer_correctness,
    evaluate_refusal,
    is_refusal,
)


def case(case_id: str, question_type: str) -> dict:
    return {"id": case_id, "question_type": question_type, "gold_evidence": []}


def test_answer_correctness_weights_partial_credit():
    records = [
        {"case_id": "a", "label": "correct"},
        {"case_id": "b", "label": "partial"},
        {"case_id": "c", "label": "incorrect"},
        {"case_id": "d", "label": "not_judged"},
    ]
    report = answer_correctness(records)
    assert report["cases_judged"] == 3
    assert report["counts"]["not_judged"] == 1
    assert report["accuracy"] == round(1.5 / 3, 6)
    assert report["strict_accuracy"] == round(1 / 3, 6)


def test_answer_correctness_keeps_unknown_labels_visible():
    report = answer_correctness([{"case_id": "a", "label": "maybe"}])
    assert report["unknown_labels"] == ["a"]
    assert report["cases_judged"] == 0
    assert report["accuracy"] is None


def test_refusal_detector_uses_project_implementation():
    assert is_refusal("Insufficient evidence to answer from the provided materials.")
    assert is_refusal("I don't know the answer.")
    assert not is_refusal("0→49、2→58、3→79【1】")
    assert not is_refusal(None)


def test_refusal_metrics_both_directions():
    cases = [case("ans", "answerable"), case("unans", "unanswerable")]
    predictions = {
        "ans": {"text": "答案是 49【1】"},
        "unans": {
            "text": "Insufficient evidence to answer from the provided materials."
        },
    }
    report = evaluate_refusal(cases, predictions)
    assert report["refusal_accuracy"] == 1.0
    assert report["false_refusal_rate"] == 0.0
    assert report["balanced_accuracy"] == 1.0


def test_refusing_everything_is_not_rewarded():
    cases = [case("ans", "answerable"), case("unans", "unanswerable")]
    predictions = {
        "ans": {"text": "I cannot answer that."},
        "unans": {"text": "I cannot answer that."},
    }
    report = evaluate_refusal(cases, predictions)
    assert report["refusal_accuracy"] == 1.0
    assert report["false_refusal_rate"] == 1.0
    assert report["balanced_accuracy"] == 0.5


def test_missing_predictions_are_reported_not_counted():
    report = evaluate_refusal([case("a", "answerable")], {})
    assert report["missing_predictions"] == ["a"]
    assert report["answerable_total"] == 0
    assert report["false_refusal_rate"] is None
