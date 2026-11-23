"""Content-level retrieval matching, alongside (not replacing) strict matching.

Strict metrics anchor on ``file + page/section``. Two known limitations make them
understate retrieval:

* the same lecture exists as PDF and PPTX, so an exact-content hit in the other
  carrier counts as a miss;
* USST DOCX chunks carry no page/section metadata, so a ``table_row`` gold locator
  can never match.

The content metric therefore checks the **verified gold quote** (from the dataset)
against the retrieved chunk text, after whitespace/case normalisation. It is only
a fallback view: strict Hit@K / MRR@K / Recall@K remain untouched and are reported
separately.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def normalize(text: str | None) -> str:
    return " ".join(str(text or "").split()).casefold()


def content_match(gold_unit: Mapping[str, Any], item: Mapping[str, Any]) -> bool:
    """True when the gold quote appears verbatim in the retrieved chunk text."""
    quote = normalize(gold_unit.get("quote"))
    if not quote:
        return False
    return quote in normalize(item.get("text"))


def match_gold_units(
    gold_units: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """For each gold unit, report the first chunk containing its quote."""
    evidence: list[dict[str, Any]] = []
    for position, unit in enumerate(gold_units):
        found = None
        quote = normalize(unit.get("quote"))
        for rank, item in enumerate(items[:k], start=1):
            if quote and content_match(unit, item):
                found = {
                    "rank": rank,
                    "chunk_id": item.get("chunk_id"),
                    "file": item.get("file"),
                    "locator": item.get("locator"),
                }
                break
        evidence.append(
            {
                "group": unit.get("group") or f"unit-{position}",
                "quote": unit.get("quote"),
                "gold_file": unit.get("file"),
                "gold_locator": unit.get("locator"),
                "matched": found is not None,
                "matched_chunk": found,
            }
        )
    return evidence


def content_case_metrics(
    case: Mapping[str, Any], items: Sequence[Mapping[str, Any]], k: int
) -> dict[str, Any]:
    gold_units = list(case.get("gold_evidence") or [])
    if case.get("question_type") != "answerable" or not gold_units:
        return {"applicable": False}
    evidence = match_gold_units(gold_units, items, k)
    matched = sum(1 for row in evidence if row["matched"])
    ranks = [row["matched_chunk"]["rank"] for row in evidence if row["matched"]]
    return {
        "applicable": True,
        "hit": matched > 0,
        "matched_units": matched,
        "total_units": len(gold_units),
        "content_recall": matched / len(gold_units),
        "reciprocal_rank": (1.0 / min(ranks)) if ranks else 0.0,
        "evidence": evidence,
    }


def content_report(
    cases: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    k: int,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    for case in cases:
        metrics = content_case_metrics(case, rankings.get(str(case["id"]), []), k)
        if not metrics.get("applicable"):
            continue
        per_case.append({"case_id": str(case["id"]), **metrics})
    if not per_case:
        return {
            "k": k,
            "cases_scored": 0,
            "content_hit_at_k": None,
            "content_recall_at_k": None,
            "per_case": [],
        }
    return {
        "k": k,
        "cases_scored": len(per_case),
        "content_hit_at_k": round(
            sum(1 for row in per_case if row["hit"]) / len(per_case), 6
        ),
        "content_recall_at_k": round(
            sum(row["content_recall"] for row in per_case) / len(per_case), 6
        ),
        "per_case": per_case,
    }
