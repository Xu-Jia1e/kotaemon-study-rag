"""D-owned adapter around C's evidence preparation and answer pipeline.

Input is the **real** B retrieval output (``retrieval.json``): each result becomes
a ``RetrievedDocument`` so C's own ``PrepareEvidencePipeline`` can number the
chunks and ``AnswerWithContextPipeline`` can generate + validate citations.

No C code is modified. Relevance/abstention logic stays inside C; this adapter
only converts shapes and captures the result.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"


def _ensure_settings() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("THEFLOW_SETTINGS_MODULE", "flowsettings")


def deepseek_key(env_file: str | Path | None = None) -> str:
    """Read the key from the local env file; never log it."""
    from dotenv import dotenv_values

    values = dotenv_values(env_file) if env_file else {}
    key = values.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not configured (pass --env-file or set the "
            "environment variable); answer generation cannot run without it"
        )
    return key


LLM_CALLS = {"count": 0}


def llm_call_count() -> int:
    return LLM_CALLS["count"]


def reset_llm_call_count() -> None:
    LLM_CALLS["count"] = 0


def build_llm(env_file: str | Path | None = None):
    _ensure_settings()
    from kotaemon.llms import ChatOpenAI

    class CountingChatOpenAI(ChatOpenAI):
        """Counts real model invocations (answer + citation extraction)."""

        def invoke(self, *args, **kwargs):
            LLM_CALLS["count"] += 1
            return super().invoke(*args, **kwargs)

        def stream(self, *args, **kwargs):
            LLM_CALLS["count"] += 1
            return super().stream(*args, **kwargs)

    return CountingChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=deepseek_key(env_file),
        base_url=DEEPSEEK_BASE_URL,
        timeout=120,
    )


def to_retrieved_documents(items: Sequence[Mapping[str, Any]]):
    """Convert B's saved results into C's ``RetrievedDocument`` inputs."""
    from kotaemon.base import RetrievedDocument

    documents = []
    for item in items:
        locator = item.get("locator") or {}
        kind = locator.get("kind")
        metadata = {
            "file_name": item.get("file") or "",
            "page_label": locator.get("value") if kind in {"page", "slide"} else None,
            "section": locator.get("value") if kind == "section" else None,
            "type": "text",
        }
        documents.append(
            RetrievedDocument(
                text=str(item.get("text") or ""),
                doc_id=item.get("chunk_id"),
                metadata=metadata,
            )
        )
    return documents


class EvidenceAnswerer:
    """C's pipelines, wired to the real DeepSeek model."""

    def __init__(
        self,
        env_file: str | Path | None = None,
        *,
        lang: str = "Chinese",
        enable_citation: bool = True,
    ) -> None:
        _ensure_settings()
        from kotaemon.indices.qa.citation import CitationPipeline
        from kotaemon.indices.qa.citation_qa import AnswerWithContextPipeline
        from kotaemon.indices.qa.format_context import PrepareEvidencePipeline

        self.llm = build_llm(env_file)
        self.evidence_pipeline = PrepareEvidencePipeline(
            max_context_length=32000, number_chunks=True
        )
        self.answer_pipeline = AnswerWithContextPipeline(
            llm=self.llm,
            citation_pipeline=CitationPipeline(llm=self.llm),
            enable_citation=enable_citation,
            enable_abstention=True,
            enable_citation_validation=True,
            enable_mindmap=False,
            enable_citation_viz=False,
            lang=lang,
        )

    def answer(
        self, question: str, items: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        started = time.perf_counter()
        prepared = self.evidence_pipeline(to_retrieved_documents(items))
        evidence_mode, evidence, images = prepared.content
        evidence_index = list(prepared.metadata.get("evidence_index") or [])
        self.answer_pipeline.evidence_index = evidence_index

        generator = self.answer_pipeline.stream(
            question=question, evidence=evidence, evidence_mode=evidence_mode
        )
        streamed: list[str] = []
        try:
            while True:
                chunk = next(generator)
                if getattr(chunk, "channel", None) == "chat" and chunk.content:
                    streamed.append(str(chunk.content))
        except StopIteration as stop:
            answer = stop.value

        latency = round(time.perf_counter() - started, 2)
        validation = (answer.metadata or {}).get("citation_validation")
        return {
            "question": question,
            "answer_text": answer.text,
            "streamed_chars": sum(len(part) for part in streamed),
            "latency_s": latency,
            "evidence_mode": evidence_mode,
            "evidence_chars": len(evidence or ""),
            "evidence_index": evidence_index,
            "citation_status": (answer.metadata or {}).get("citation_status"),
            "citation_validation": validation,
            "has_images": bool(images),
        }
