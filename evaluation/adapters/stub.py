"""Stub adapters for offline exercises and unit tests.

These never claim to be experimental results: the retrieval stub replays a
fixture file, and the answer stub returns pre-written texts. They exist so the
whole pipeline (dataset -> adapter -> metrics -> report) can be exercised with
no model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.adapters.base import (
    AnswerResult,
    RetrievedItem,
    from_mapping,
)


class FixtureRetrievalAdapter:
    """Replay a JSON fixture of the shape ``{case_id: [item, ...]}``."""

    name = "fixture-retrieval"

    def __init__(self, rankings: Mapping[str, Sequence[Mapping[str, Any]]]):
        self._rankings = {str(key): list(value) for key, value in rankings.items()}

    @classmethod
    def from_file(cls, path: str | Path) -> "FixtureRetrievalAdapter":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("retrieval fixture must be a JSON object")
        return cls(payload)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        case: Mapping[str, Any] | None = None,
    ) -> Sequence[RetrievedItem]:
        case_id = str((case or {}).get("id", ""))
        rows = self._rankings.get(case_id, [])
        return [from_mapping(row) for row in rows[:top_k]]


class FixtureAnswerAdapter:
    """Return pre-written answers from ``{case_id: answer}``."""

    name = "fixture-answer"

    def __init__(self, answers: Mapping[str, Mapping[str, Any]]):
        self._answers = {str(key): dict(value) for key, value in answers.items()}

    @classmethod
    def from_file(cls, path: str | Path) -> "FixtureAnswerAdapter":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("answer fixture must be a JSON object")
        return cls(payload)

    def answer(
        self,
        case: Mapping[str, Any],
        evidence: Sequence[RetrievedItem],
    ) -> AnswerResult:
        case_id = str(case.get("id"))
        payload = self._answers.get(case_id)
        if payload is None:
            raise KeyError(f"no fixture answer for case {case_id!r}")
        return AnswerResult(
            text=str(payload.get("text", "")),
            evidence_index=list(payload.get("evidence_index") or []),
            citation_validation=payload.get("citation_validation"),
        )
