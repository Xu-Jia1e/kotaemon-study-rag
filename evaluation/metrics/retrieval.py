"""Retrieval metrics: Hit@K, MRR@K and evidence recall@K.

Matching policy (important for fairness across chunking schemes)
-----------------------------------------------------------------
Gold evidence is anchored on ``file + locator`` (page / slide / section /
table row), *not* on chunk ids, because two chunking schemes produce different
chunk ids and boundaries for the same source passage. A chunk id may still be
compared when both sides carry one (used as a fast path in tests and when the
retriever exposes stable ids).

Item shape (produced by ``evaluation.adapters.base.from_part_b_result``)::

    {"chunk_id": str | None,
     "file": "11-hash2.pdf",
     "locator": {"kind": "page", "value": 25},
     "text": "...",
     "score": 0.59}

Gold unit shape (see ``evaluation/datasets/schema.md``)::

    {"file": "11-hash2.pdf",
     "locator": {"kind": "page", "value": 25},
     "chunk_id": None}
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

# Locator kinds that mean "position inside a document". ``page_label`` is the
# metadata key used by the project's loaders, ``page`` the human-facing name.
_PAGE_KINDS = {"page", "page_label", "page_or_slide"}
_SLIDE_KINDS = {"slide", "slide_label"}
_SECTION_KINDS = {"section", "table_row", "heading"}


def _basename(name: str | None) -> str:
    if not name:
        return ""
    return str(name).replace("\\", "/").rsplit("/", 1)[-1].strip().casefold()


def _kind_group(kind: str | None) -> str:
    kind = (kind or "").strip().casefold()
    if kind in _SLIDE_KINDS:
        return "slide"
    if kind in _PAGE_KINDS:
        return "page"
    if kind in _SECTION_KINDS:
        return "section"
    return kind or "unknown"


def _normalise_value(group: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else round(float(value), 6)
    text = str(value).strip()
    if group in {"page", "slide"}:
        digits = "".join(ch for ch in text if ch.isdigit())
        return int(digits) if digits else None
    return " ".join(text.casefold().split())


def evidence_matches(gold: Mapping[str, Any], item: Mapping[str, Any]) -> bool:
    """Return True when a retrieved item satisfies one gold evidence unit."""
    if _basename(gold.get("file")) != _basename(item.get("file")):
        return False

    gold_chunk = gold.get("chunk_id")
    item_chunk = item.get("chunk_id")
    if gold_chunk and item_chunk:
        return str(gold_chunk) == str(item_chunk)

    gold_loc = gold.get("locator") or {}
    item_loc = item.get("locator") or {}
    if not gold_loc:
        # Evidence pinned to a chunk id only; cannot match by position.
        return False
    if not item_loc:
        return False

    gold_group = _kind_group(gold_loc.get("kind"))
    item_group = _kind_group(item_loc.get("kind"))
    if gold_group != item_group:
        return False

    gold_value = _normalise_value(gold_group, gold_loc.get("value"))
    item_value = _normalise_value(item_group, item_loc.get("value"))
    if gold_value is None or item_value is None:
        return False
    return gold_value == item_value


def match_ranks(
    gold_units: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    k: int,
) -> list[int | None]:
    """First rank (1-based) of each gold unit inside the top-``k`` items."""
    ranks: list[int | None] = []
    for gold in gold_units:
        rank = None
        for position, item in enumerate(items[:k], start=1):
            if evidence_matches(gold, item):
                rank = position
                break
        ranks.append(rank)
    return ranks


def hit_at_k(
    gold_units: Sequence[Mapping[str, Any]], items: Sequence[Mapping[str, Any]], k: int
) -> bool:
    """True when at least one gold evidence unit appears in the top-``k``."""
    return any(rank is not None for rank in match_ranks(gold_units, items, k))


def mrr_at_k(
    gold_units: Sequence[Mapping[str, Any]], items: Sequence[Mapping[str, Any]], k: int
) -> float:
    """Reciprocal rank of the first correctly retrieved evidence unit."""
    ranks = [rank for rank in match_ranks(gold_units, items, k) if rank is not None]
    return 1.0 / min(ranks) if ranks else 0.0


def _group_key(unit: Mapping[str, Any], position: int) -> str:
    """Evidence group: units in the same group are interchangeable.

    A two-part question needs both groups; "page 26 **or** 27" is one group with
    two alternative units. Units without an explicit ``group`` are their own.
    """
    group = unit.get("group")
    return str(group) if group else f"__unit_{position}"


def group_units(gold_units: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    """Map ``group -> [unit index, ...]`` preserving first-seen order."""
    groups: dict[str, list[int]] = {}
    for position, unit in enumerate(gold_units):
        groups.setdefault(_group_key(unit, position), []).append(position)
    return groups


@dataclass
class CaseRetrieval:
    """Per-question retrieval result."""

    case_id: str
    question: str
    hit: bool
    first_rank: int | None
    reciprocal_rank: float
    matched_units: int
    total_units: int
    matched_groups: int = 0
    total_groups: int = 0

    @property
    def evidence_recall(self) -> float:
        """Fraction of required evidence groups covered (alternatives count once)."""
        return self.matched_groups / self.total_groups if self.total_groups else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_recall"] = round(self.evidence_recall, 6)
        return payload


@dataclass
class RetrievalReport:
    """Aggregate retrieval metrics plus the per-question breakdown."""

    k: int
    cases_scored: int
    cases_skipped: int = 0
    skipped_ids: list[str] = field(default_factory=list)
    per_case: list[CaseRetrieval] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return (
            sum(1 for case in self.per_case if case.hit) / self.cases_scored
            if self.cases_scored
            else 0.0
        )

    @property
    def mrr(self) -> float:
        return (
            sum(case.reciprocal_rank for case in self.per_case) / self.cases_scored
            if self.cases_scored
            else 0.0
        )

    @property
    def evidence_recall(self) -> float:
        return (
            sum(case.evidence_recall for case in self.per_case) / self.cases_scored
            if self.cases_scored
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "cases_scored": self.cases_scored,
            "cases_skipped": self.cases_skipped,
            "skipped_ids": list(self.skipped_ids),
            "hit_at_k": round(self.hit_rate, 6),
            "mrr_at_k": round(self.mrr, 6),
            "evidence_recall_at_k": round(self.evidence_recall, 6),
            "per_case": [case.to_dict() for case in self.per_case],
        }


def evaluate_retrieval(
    cases: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    k: int = 5,
) -> RetrievalReport:
    """Score answerable cases against retrieved rankings.

    Unanswerable cases are skipped: Hit@K is defined over questions that *have*
    a correct evidence unit (see ``DESIGN.md``). Cases without gold evidence are
    skipped as well and reported in ``skipped_ids`` so they cannot silently
    inflate or deflate the score.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    report = RetrievalReport(k=k, cases_scored=0)
    for case in cases:
        if case.get("question_type") != "answerable":
            continue
        gold_units = list(case.get("gold_evidence") or [])
        if not gold_units:
            report.cases_skipped += 1
            report.skipped_ids.append(str(case.get("id")))
            continue

        items = list(rankings.get(str(case.get("id")), []))
        ranks = match_ranks(gold_units, items, k)
        matched = [rank for rank in ranks if rank is not None]
        groups = group_units(gold_units)
        matched_groups = sum(
            1
            for positions in groups.values()
            if any(ranks[position] is not None for position in positions)
        )
        report.per_case.append(
            CaseRetrieval(
                case_id=str(case.get("id")),
                question=str(case.get("question", "")),
                hit=bool(matched),
                first_rank=min(matched) if matched else None,
                reciprocal_rank=(1.0 / min(matched)) if matched else 0.0,
                matched_units=len(matched),
                total_units=len(gold_units),
                matched_groups=matched_groups,
                total_groups=len(groups),
            )
        )
    report.cases_scored = len(report.per_case)
    return report
