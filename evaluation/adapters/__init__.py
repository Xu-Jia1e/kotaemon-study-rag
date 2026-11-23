"""Adapter layer: normalise retrieval/answer backends for the metrics."""

from evaluation.adapters.base import (
    AnswerAdapter,
    AnswerResult,
    RetrievalAdapter,
    RetrievedItem,
    from_mapping,
    from_part_b_result,
)
from evaluation.adapters.part_b import PartBRetrievalAdapter
from evaluation.adapters.stub import FixtureAnswerAdapter, FixtureRetrievalAdapter

__all__ = [
    "AnswerAdapter",
    "AnswerResult",
    "FixtureAnswerAdapter",
    "FixtureRetrievalAdapter",
    "PartBRetrievalAdapter",
    "RetrievalAdapter",
    "RetrievedItem",
    "from_mapping",
    "from_part_b_result",
]
