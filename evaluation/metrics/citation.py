"""Citation metrics: marker validity (automatic) and support (human labels).

Two different questions must not be conflated:

1. **Validity** — does every ``【n】`` in the answer point at an evidence block
   that was actually retrieved? This is fully automatic: we delegate to
   ``kotaemon.indices.qa.utils.validate_citations``, i.e. the exact function the
   product uses, so the metric cannot drift from the shipped behaviour.
2. **Support** — does the cited passage really support the sentence? No string
   comparison can decide this, so it is recorded as a human label per citation
   and only aggregated here.

Abstentions are excluded from citation scoring: an answer that correctly says
"insufficient evidence" has no citation to validate by design. (The current
``validate_citations`` returns ``citation_valid=False`` for a bare abstention —
see ``DESIGN.md`` §known-issues — so counting it would punish correct refusals.)
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

# Mirror of the project regex (fork commit a5157c9) used only as a fallback when
# kotaemon cannot be imported.
_CITATION_GROUP_PATTERN = re.compile(r"[【\[]([\d,\s\-–]+)[】\]]")
_ABSTAIN_PATTERNS = (
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


def _fallback_expand(group: str) -> list[int]:
    indices: list[int] = []
    for part in re.split(r"[,\s]+", group.strip()):
        if not part:
            continue
        match = re.match(r"^(\d+)[\-–](\d+)$", part)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if start <= end:
                indices.extend(range(start, end + 1))
            else:
                indices.extend(range(end, start + 1))
            continue
        if part.isdigit():
            indices.append(int(part))
    return indices


def _fallback_parse(text: str | None) -> list[int]:
    if not text:
        return []
    indices: list[int] = []
    for match in _CITATION_GROUP_PATTERN.finditer(text):
        indices.extend(_fallback_expand(match.group(1)))
    return indices


def _fallback_validate(
    text: str | None, evidence_index: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    available = {
        entry["index"]
        for entry in evidence_index or []
        if isinstance(entry.get("index"), int)
    }
    cited = _fallback_parse(text)
    seen: set[int] = set()
    valid: list[int] = []
    invalid: list[int] = []
    for index in cited:
        if index in seen:
            continue
        seen.add(index)
        (valid if index in available else invalid).append(index)
    return {
        "citation_valid": bool(cited) and not invalid,
        "cited_indices": valid,
        "invalid_citation_ids": invalid,
        "unsupported_citations": list(available) if not cited else list(invalid),
        "claim_count": 0,
        "cited_claim_count": 0,
        "coverage": 0.0,
    }


def validate_answer_citations(
    text: str | None, evidence_index: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate citation markers against the evidence index.

    Uses the project implementation when importable (authoritative), and a
    documented local mirror otherwise.
    """
    try:
        from kotaemon.indices.qa.utils import validate_citations  # type: ignore
    except Exception:  # pragma: no cover - only when kotaemon is unavailable
        return _fallback_validate(text, evidence_index)
    return validate_citations(text, list(evidence_index)).to_dict()


def _is_refusal(text: str | None) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(pattern in lowered for pattern in _ABSTAIN_PATTERNS)


def evaluate_citations(
    cases: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    support_labels: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Aggregate citation validity (automatic) and support (labelled).

    ``predictions[case_id]`` must provide ``text`` and ``evidence_index``.
    ``support_labels`` items: ``{"case_id", "citation_index", "supported": bool}``.
    """
    per_case: list[dict[str, Any]] = []
    scored = 0
    valid_answers = 0
    abstentions = 0
    missing = 0
    invalid_ids_total = 0
    coverages: list[float] = []

    for case in cases:
        case_id = str(case.get("id"))
        prediction = predictions.get(case_id)
        if prediction is None:
            missing += 1
            continue
        text = prediction.get("text") or ""
        if _is_refusal(text):
            abstentions += 1
            continue
        evidence_index = prediction.get("evidence_index") or []
        if not evidence_index:
            # No evidence retrieved: citation scoring is undefined, not zero.
            missing += 1
            continue

        validation = validate_answer_citations(text, evidence_index)
        scored += 1
        valid_answers += int(bool(validation.get("citation_valid")))
        invalid_ids_total += len(validation.get("invalid_citation_ids") or [])
        coverages.append(float(validation.get("coverage") or 0.0))
        per_case.append(
            {
                "case_id": case_id,
                "citation_valid": bool(validation.get("citation_valid")),
                "cited_indices": validation.get("cited_indices") or [],
                "invalid_citation_ids": validation.get("invalid_citation_ids") or [],
                "coverage": round(float(validation.get("coverage") or 0.0), 6),
            }
        )

    supported = total_labels = 0
    unsupported_ids: list[str] = []
    for label in support_labels:
        total_labels += 1
        if label.get("supported") is True:
            supported += 1
        else:
            unsupported_ids.append(
                f"{label.get('case_id')}#{label.get('citation_index')}"
            )

    return {
        "cases_scored": scored,
        "answers_without_evidence": missing,
        "excluded_abstentions": abstentions,
        "citation_validity_rate": round(valid_answers / scored, 6) if scored else None,
        "invalid_citation_id_count": invalid_ids_total,
        "mean_citation_coverage": (
            round(sum(coverages) / len(coverages), 6) if coverages else None
        ),
        "citation_support_total": total_labels,
        "citation_support_rate": (
            round(supported / total_labels, 6) if total_labels else None
        ),
        "unsupported_citations": unsupported_ids,
        "per_case": per_case,
    }
