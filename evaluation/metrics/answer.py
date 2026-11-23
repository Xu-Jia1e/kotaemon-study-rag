"""Answer correctness and refusal metrics.

Both metrics need a *human* judgement for the semantic part; the code here only
aggregates pre-recorded labels. Nothing calls a model, so the numbers can never
be produced by guesswork: an unlabelled answer simply stays ``not_judged`` and
is excluded from the denominators.

Label scale (``ANSWER_LABELS``) is a three-way rubric, chosen because the task
asks for "答案正确率（人工判断）" while partial answers (e.g. half of a
two-part question) must not be counted as fully correct:

    correct    -> 1.0
    partial    -> 0.5
    incorrect  -> 0.0
    not_judged -> excluded

Refusal is detected with the project's own detector
(``kotaemon.indices.qa.utils.is_abstention``) so the metric measures the
behaviour the product actually ships, not a re-implementation. A local mirror is
used only when ``kotaemon`` cannot be imported.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

ANSWER_LABELS: dict[str, float | None] = {
    "correct": 1.0,
    "partial": 0.5,
    "incorrect": 0.0,
    "not_judged": None,
}

# Mirrors kotaemon.indices.qa.utils._ABSTAIN_PATTERNS (fork commit a5157c9).
_FALLBACK_PATTERNS = (
    "insufficient evidence",
    "insufficient information",
    "not enough evidence",
    "not enough information",
    "don't know",
    "do not know",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "no relevant information",
    "not provided in the context",
)


def _fallback_is_refusal(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(pattern in lowered for pattern in _FALLBACK_PATTERNS)


def is_refusal(text: str | None) -> bool:
    """True when ``text`` is an abstention/refusal.

    Prefers the project implementation so evaluation and product agree.
    """
    try:
        from kotaemon.indices.qa.utils import is_abstention  # type: ignore
    except Exception:  # pragma: no cover - only when kotaemon is unavailable
        return _fallback_is_refusal(text)
    return bool(is_abstention(text))


def answer_correctness(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate human labels into accuracy figures.

    ``records`` items: ``{"case_id": "sc-01", "label": "correct"}``.
    """
    counts = {name: 0 for name in ANSWER_LABELS}
    scored: list[tuple[str, float]] = []
    unknown: list[str] = []

    for record in records:
        label = str(record.get("label", "not_judged"))
        if label not in ANSWER_LABELS:
            unknown.append(str(record.get("case_id")))
            continue
        counts[label] += 1
        score = ANSWER_LABELS[label]
        if score is not None:
            scored.append((str(record.get("case_id")), score))

    total_judged = len(scored)
    weighted = sum(score for _, score in scored)
    strict = sum(1 for _, score in scored if score == 1.0)
    return {
        "cases_total": sum(counts.values()) + len(unknown),
        "cases_judged": total_judged,
        "counts": counts,
        "unknown_labels": unknown,
        "accuracy": round(weighted / total_judged, 6) if total_judged else None,
        "strict_accuracy": round(strict / total_judged, 6) if total_judged else None,
        "per_case": [{"case_id": cid, "score": score} for cid, score in scored],
    }


def evaluate_refusal(
    cases: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Report refusal behaviour in both directions.

    * unanswerable questions -> ``refusal_accuracy`` (higher is better)
    * answerable questions   -> ``false_refusal_rate`` (lower is better)
    * ``balanced_accuracy`` combines both so a system that refuses everything
      (or answers everything) cannot score well.
    """
    per_case: list[dict[str, Any]] = []
    unanswerable_total = unanswerable_refused = 0
    answerable_total = answerable_refused = 0
    missing: list[str] = []

    for case in cases:
        case_id = str(case.get("id"))
        prediction = predictions.get(case_id)
        if prediction is None:
            missing.append(case_id)
            continue
        text = prediction.get("text")
        refused = is_refusal(text if isinstance(text, str) else None)
        kind = case.get("question_type")
        per_case.append(
            {
                "case_id": case_id,
                "question_type": kind,
                "refused": refused,
            }
        )
        if kind == "unanswerable":
            unanswerable_total += 1
            unanswerable_refused += int(refused)
        elif kind == "answerable":
            answerable_total += 1
            answerable_refused += int(refused)

    refusal_accuracy = (
        unanswerable_refused / unanswerable_total if unanswerable_total else None
    )
    false_refusal_rate = (
        answerable_refused / answerable_total if answerable_total else None
    )
    balanced = None
    if refusal_accuracy is not None and false_refusal_rate is not None:
        balanced = (refusal_accuracy + (1.0 - false_refusal_rate)) / 2.0

    return {
        "unanswerable_total": unanswerable_total,
        "unanswerable_refused": unanswerable_refused,
        "refusal_accuracy": _round(refusal_accuracy),
        "answerable_total": answerable_total,
        "answerable_refused": answerable_refused,
        "false_refusal_rate": _round(false_refusal_rate),
        "balanced_accuracy": _round(balanced),
        "missing_predictions": missing,
        "per_case": per_case,
    }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)
