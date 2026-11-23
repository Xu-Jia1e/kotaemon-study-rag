"""Run B's ``VectorRetrieval`` against A's real index (D-owned adapter).

This is the bridge the integration needed: A's index is opened through
``evaluation.adapters.a_index`` and B's own ``VectorRetrieval`` is pointed at the
same vector store / document store / embedding client. No code in B is changed.

Only ``retrieval_mode="vector"`` is wired here; Hybrid/RRF/reranking stay out of
scope for this stage.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.adapters.a_index import (
    IndexBundle,
    create_embedding,
    load_index,
)
from evaluation.adapters.base import RetrievedItem


def _locator(file_name: str, page_label: Any, section: Any) -> dict[str, Any]:
    if page_label is not None:
        kind = "slide" if str(file_name).lower().endswith(".pptx") else "page"
        return {"kind": kind, "value": page_label}
    if section:
        return {"kind": "section", "value": section}
    return {"kind": "unknown", "value": None}


class AIndexVectorAdapter:
    """Retrieval adapter: question -> B VectorRetrieval -> deduplicated Top-K."""

    name = "a-index-vector"

    def __init__(
        self,
        index_dir: str | Path,
        *,
        env_file: str | Path | None = None,
        top_k_default: int = 5,
        embedding: Any | None = None,
    ) -> None:
        self.bundle: IndexBundle = load_index(
            index_dir, embedding=embedding or create_embedding(env_file)
        )
        from kotaemon.indices import VectorRetrieval

        self._retriever = VectorRetrieval(
            vector_store=self.bundle.vector_store,
            doc_store=self.bundle.doc_store,
            embedding=self.bundle.embedding,
            retrieval_mode="vector",
            rerankers=[],
            top_k=top_k_default,
        )
        self.last_latency_s: float | None = None

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        case: Mapping[str, Any] | None = None,
    ) -> Sequence[RetrievedItem]:
        started = time.perf_counter()
        documents = self._retriever(question, top_k=top_k)
        self.last_latency_s = round(time.perf_counter() - started, 3)

        items: list[RetrievedItem] = []
        for document in documents:
            metadata = document.metadata or {}
            file_name = str(metadata.get("file_name") or "")
            items.append(
                RetrievedItem(
                    chunk_id=document.doc_id,
                    file=file_name,
                    locator=_locator(
                        file_name,
                        metadata.get("page_label"),
                        metadata.get("section"),
                    ),
                    text=document.text or "",
                    score=float(document.score) if document.score is not None else None,
                    extra={"type": metadata.get("type")},
                )
            )
        return items

    @staticmethod
    def to_record(item: RetrievedItem) -> dict[str, Any]:
        return item.to_dict()

    def close(self) -> None:
        """Best-effort release of the underlying stores (Chroma/LanceDB)."""
        for store in (self.bundle.vector_store, self.bundle.doc_store):
            for attribute in ("_collection", "_conn", "_db", "db_connection"):
                value = getattr(store, attribute, None)
                if value is not None and hasattr(value, "close"):
                    try:
                        value.close()
                    except Exception:  # pragma: no cover - best effort only
                        pass
