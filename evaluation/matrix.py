"""Experiment matrix definition and availability status.

Design rule: the matrix never assumes a capability exists. Every dimension
declares whether it is ``ready`` or ``blocked``, with a reason and the code
location that justifies the verdict. A cell is only ``ready`` when *all* of its
dimensions are ready, so a blocked index or a missing reranker cannot silently
produce a "result".

The same dataset file is used by every cell; ``dataset_manifest`` records its
SHA-256 so "same 24 questions" is verifiable rather than asserted.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "evaluation" / "datasets" / "qa_cases_v2.jsonl"


@dataclass(frozen=True)
class Dimension:
    """Availability of one value of one experiment dimension."""

    value: str
    status: str  # "ready" | "blocked"
    reason: str = ""
    reference: str = ""

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, str]:
        return {
            "value": self.value,
            "status": self.status,
            "reason": self.reason,
            "reference": self.reference,
        }


# Two token-chunking profiles from the task description. Neither is queryable
# from this checkout: the app database is not built here, and Part B's index is
# page-level text, not token chunks.
CHUNKING_PROFILES: dict[str, Dimension] = {
    "1024-256": Dimension(
        value="1024-256",
        status="blocked",
        reason=(
            "本机没有 A 的主库索引；Part B 的自建索引按页抽取文本，"
            "不是 1024/256 token 分块，无法代表该方案"
        ),
        reference=(
            "docs/part-b-retrieval.md; "
            "libs/ktem/ktem/index/file/pipelines.py:788"
        ),
    ),
    "512-128": Dimension(
        value="512-128",
        status="blocked",
        reason="A 尚未提供第二套 512/128 索引，也没有可复现的分块参数记录",
        reference="evaluation/DESIGN.md §6",
    ),
}

# Retrieval methods exist in code, but every path needs a real index and a
# DashScope embedding key, so none can run in this offline stage.
RETRIEVAL_METHODS: dict[str, Dimension] = {
    "vector": Dimension(
        value="vector",
        status="blocked",
        reason="需要已建好的索引 + DASHSCOPE_API_KEY；本阶段不调用真实 API",
        reference="scripts/part_b_retrieval.py:134",
    ),
    "hybrid": Dimension(
        value="hybrid",
        status="blocked",
        reason="同 vector，且混合检索无分数融合，只能比较排名类指标",
        reference="libs/kotaemon/kotaemon/indices/vectorindex.py:224",
    ),
    "rrf": Dimension(
        value="rrf",
        status="blocked",
        reason="RRF 只存在于 B 的脚本中，仍需索引与 embedding key",
        reference="scripts/part_b_retrieval.py:41",
    ),
}

TOP_K_VALUES: tuple[int, ...] = (3, 5, 10)

# Reranking is explicitly optional. "off" needs no capability and is the only
# value used by the main matrix; "on" is blocked because no reranker is wired.
RERANKING_OPTIONS: dict[str, Dimension] = {
    "off": Dimension(
        value="off",
        status="ready",
        reason="不使用重排；候选深度需固定为 top_k，避免 do_extend 隐式放大",
        reference="libs/ktem/ktem/index/file/pipelines.py:156",
    ),
    "on": Dimension(
        value="on",
        status="blocked",
        reason=(
            "Cohere key 为空时重排原样返回；qwen3-rerank 尚未接入，"
            "因此“已实现重排”不成立"
        ),
        reference="libs/kotaemon/kotaemon/indices/rankings/cohere.py:41",
    ),
}


@dataclass(frozen=True)
class ExperimentCell:
    """One (chunking, retrieval, top_k, reranking) combination."""

    chunking: str
    retrieval: str
    top_k: int
    reranking: str = "off"
    blocked_by: tuple[str, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def cell_id(self) -> str:
        return f"c{self.chunking}__{self.retrieval}__k{self.top_k}__rr-{self.reranking}"

    @property
    def status(self) -> str:
        return "blocked" if self.blocked_by else "ready"

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "chunking_strategy": self.chunking,
            "retrieval_method": self.retrieval,
            "top_k": self.top_k,
            "reranking": self.reranking,
            "status": self.status,
            "blocked_by": list(self.blocked_by),
            "reasons": list(self.reasons),
        }


def build_matrix(
    chunkings: Sequence[str] | None = None,
    retrievals: Sequence[str] | None = None,
    top_ks: Sequence[int] | None = None,
    rerankings: Sequence[str] | None = None,
) -> list[ExperimentCell]:
    """Return the full matrix, each cell carrying its availability verdict."""
    chunkings = list(chunkings or CHUNKING_PROFILES)
    retrievals = list(retrievals or RETRIEVAL_METHODS)
    top_ks = list(top_ks or TOP_K_VALUES)
    rerankings = list(rerankings or ("off",))

    cells: list[ExperimentCell] = []
    for chunking, retrieval, top_k, reranking in itertools.product(
        chunkings, retrievals, top_ks, rerankings
    ):
        cell_dimensions = {
            f"chunking:{chunking}": CHUNKING_PROFILES[chunking],
            f"retrieval:{retrieval}": RETRIEVAL_METHODS[retrieval],
            f"reranking:{reranking}": RERANKING_OPTIONS[reranking],
        }
        blocked = [
            name
            for name, dimension in cell_dimensions.items()
            if not dimension.ready
        ]
        reasons = [
            f"{name}: {dimension.reason}"
            for name, dimension in cell_dimensions.items()
            if not dimension.ready
        ]
        cells.append(
            ExperimentCell(
                chunking=chunking,
                retrieval=retrieval,
                top_k=int(top_k),
                reranking=reranking,
                blocked_by=tuple(blocked),
                reasons=tuple(reasons),
            )
        )
    return cells


def summarise(cells: Sequence[ExperimentCell]) -> dict[str, object]:
    """Count cells by status and list the distinct blocking dimensions."""
    by_status: dict[str, int] = {}
    blocking: dict[str, int] = {}
    for cell in cells:
        by_status[cell.status] = by_status.get(cell.status, 0) + 1
        for name in cell.blocked_by:
            blocking[name] = blocking.get(name, 0) + 1
    return {
        "cells_total": len(cells),
        "by_status": by_status,
        "blocking_dimensions": blocking,
        "ready_cell_ids": [cell.cell_id for cell in cells if cell.status == "ready"],
    }


def dataset_manifest(path: Path = DATASET_PATH) -> dict[str, object]:
    """Identify the exact dataset shared by every cell."""
    raw = path.read_bytes()
    cases = sum(1 for line in raw.decode("utf-8").splitlines() if line.strip())
    return {
        "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "cases": cases,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def status_table(cells: Sequence[ExperimentCell]) -> str:
    """Markdown table of the matrix status, safe to paste into a report."""
    lines = [
        "| cell | chunking | retrieval | top_k | reranking | status | blocked by |",
        "|---|---|---|---:|---|---|---|",
    ]
    for cell in cells:
        blocked = ", ".join(cell.blocked_by) or "-"
        lines.append(
            f"| `{cell.cell_id}` | {cell.chunking} | {cell.retrieval} | {cell.top_k} "
            f"| {cell.reranking} | {cell.status} | {blocked} |"
        )
    return "\n".join(lines)
