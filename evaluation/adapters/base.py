"""Adapter interfaces.

The evaluation code must not depend on *how* retrieval and answering are
implemented. Each part plugs in through a narrow protocol:

* ``RetrievalAdapter``  — question + top_k -> ranked items  (Part B today, A's
  main index later)
* ``AnswerAdapter``     — case + evidence -> answer text + evidence index
  (Part C's pipeline, or a stub for offline exercises)

Converters in this module keep a single definition of the item shape so metric
code never has to know about a specific script's JSON schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class RetrievedItem:
    """One ranked retrieval hit, normalised across adapters."""

    chunk_id: str | None
    file: str
    locator: dict[str, Any]
    text: str = ""
    score: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "file": self.file,
            "locator": dict(self.locator),
            "text": self.text,
            "score": self.score,
        }


@dataclass
class AnswerResult:
    """One answer produced by an answer adapter."""

    text: str
    evidence_index: list[dict[str, Any]] = field(default_factory=list)
    citation_validation: dict[str, Any] | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)


@runtime_checkable
class RetrievalAdapter(Protocol):
    """Anything that can rank passages for a question."""

    name: str

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        case: Mapping[str, Any] | None = None,
    ) -> Sequence[RetrievedItem]:
        ...


@runtime_checkable
class AnswerAdapter(Protocol):
    """Anything that can answer a question from retrieved evidence."""

    name: str

    def answer(
        self,
        case: Mapping[str, Any],
        evidence: Sequence[RetrievedItem],
    ) -> AnswerResult:
        ...


def from_part_b_result(result: Mapping[str, Any]) -> RetrievedItem:
    """Convert one entry of ``scripts/part_b_retrieval.py`` JSON output.

    Part B emits ``{id, file_name, page, section, score, score_type, text}``.
    ``page`` is used when present, otherwise ``section``.
    """
    page = result.get("page")
    section = result.get("section")
    if page is not None:
        locator = {"kind": "page", "value": page}
    elif section:
        locator = {"kind": "section", "value": section}
    else:
        locator = {"kind": "unknown", "value": None}
    score = result.get("score")
    return RetrievedItem(
        chunk_id=str(result["id"]) if result.get("id") is not None else None,
        file=str(result.get("file_name") or ""),
        locator=locator,
        text=str(result.get("text") or ""),
        score=float(score) if isinstance(score, (int, float)) else None,
        extra={"score_type": result.get("score_type")},
    )


def from_mapping(item: Mapping[str, Any]) -> RetrievedItem:
    """Build an item from a already-normalised dict (run files, fixtures)."""
    locator = item.get("locator") or {}
    raw_score = item.get("score")
    return RetrievedItem(
        chunk_id=str(item["chunk_id"]) if item.get("chunk_id") else None,
        file=str(item.get("file") or ""),
        locator={"kind": locator.get("kind"), "value": locator.get("value")},
        text=str(item.get("text") or ""),
        score=float(raw_score) if isinstance(raw_score, (int, float)) else None,
    )
