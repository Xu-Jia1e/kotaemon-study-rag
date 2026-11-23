"""Metric implementations used by the evaluation scripts.

All functions are pure (no network, no model calls, no file writes) so they can
be unit-tested offline.
"""

from evaluation.metrics.retrieval import (
    CaseRetrieval,
    evaluate_retrieval,
    evidence_matches,
    hit_at_k,
    mrr_at_k,
)
from evaluation.metrics.answer import (
    ANSWER_LABELS,
    answer_correctness,
    evaluate_refusal,
    is_refusal,
)
from evaluation.metrics.citation import evaluate_citations, validate_answer_citations

__all__ = [
    "ANSWER_LABELS",
    "CaseRetrieval",
    "answer_correctness",
    "evaluate_citations",
    "evaluate_refusal",
    "evaluate_retrieval",
    "evidence_matches",
    "hit_at_k",
    "is_refusal",
    "mrr_at_k",
    "validate_answer_citations",
]
