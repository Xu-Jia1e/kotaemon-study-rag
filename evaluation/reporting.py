"""Per-question result records and the aggregate report.

Each record keeps the twelve fields required by the task, plus a few audit
fields (category, reranking, simulated). ``citation_valid`` and
``citation_supported`` are deliberately separate: the first only checks that
``【n】`` points at an evidence block that was returned, the second is the human
judgement that the cited text really supports the sentence.

Nothing here invents a judgement: a missing label stays ``None`` and is excluded
from the denominator instead of being counted as 0 or 1.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluation.metrics.answer import ANSWER_LABELS, is_refusal
from evaluation.metrics.citation import (
    normalize_support_label,
    validate_answer_citations,
)
from evaluation.metrics.retrieval import group_units, match_ranks

CATEGORIES = ("direct", "synthesis", "detail", "unanswerable")


@dataclass
class CaseRecord:
    """One question under one experiment cell."""

    case_id: str
    category: str
    chunking_strategy: str
    retrieval_method: str
    top_k: int
    reranking: str
    simulated: bool
    retrieved_evidence: list[dict[str, Any]] = field(default_factory=list)
    hit_at_k: bool | None = None
    reciprocal_rank: float | None = None
    recall_at_k: float | None = None
    generated_answer: str | None = None
    answer_correct: str | None = None
    citation_valid: bool | None = None
    citation_supported: str | None = None
    refusal_correct: bool | None = None
    expected_refusal: bool = False
    citation_support_supported: int = 0
    citation_support_unsupported: int = 0
    citation_support_uncertain: int = 0
    citation_support_assessable: int = 0
    citation_support_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["reciprocal_rank"] is not None:
            payload["reciprocal_rank"] = round(payload["reciprocal_rank"], 6)
        if payload["recall_at_k"] is not None:
            payload["recall_at_k"] = round(payload["recall_at_k"], 6)
        return payload


def _recall_at_k(gold: Sequence[Mapping[str, Any]], items, k: int) -> float:
    ranks = match_ranks(gold, items, k)
    groups = group_units(gold)
    covered = sum(
        1
        for positions in groups.values()
        if any(ranks[position] is not None for position in positions)
    )
    return covered / len(groups) if groups else 0.0


def _citation_support(
    labels: Sequence[Mapping[str, Any]]
) -> tuple[str | None, int, int, int]:
    if not labels:
        return None, 0, 0, 0
    supported = unsupported = uncertain = 0
    for label in labels:
        category = normalize_support_label(label)
        if category == "supported":
            supported += 1
        elif category == "unsupported":
            unsupported += 1
        else:
            uncertain += 1
    if supported and not unsupported and not uncertain:
        verdict = "supported"
    elif unsupported and not supported and not uncertain:
        verdict = "unsupported"
    elif uncertain and not supported and not unsupported:
        verdict = "uncertain"
    else:
        verdict = "mixed"
    return verdict, supported, unsupported, uncertain


def build_case_records(
    cases: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions: Mapping[str, Mapping[str, Any]],
    answer_labels: Mapping[str, str],
    citation_labels: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    chunking_strategy: str,
    retrieval_method: str,
    top_k: int,
    reranking: str = "off",
    simulated: bool = False,
) -> list[CaseRecord]:
    """Join dataset + rankings + predictions + human labels into records."""
    records: list[CaseRecord] = []
    for case in cases:
        case_id = str(case["id"])
        question_type = case.get("question_type")
        expected_refusal = question_type == "unanswerable"
        gold = list(case.get("gold_evidence") or [])
        items = list(rankings.get(case_id, []))[:top_k]

        hit = rank = recall = None
        if question_type == "answerable" and gold:
            ranks = match_ranks(gold, items, top_k)
            matched = [value for value in ranks if value is not None]
            hit = bool(matched)
            rank = 1.0 / min(matched) if matched else 0.0
            recall = _recall_at_k(gold, items, top_k)

        prediction = predictions.get(case_id) or {}
        text = prediction.get("text")
        evidence_index = prediction.get("evidence_index") or []
        refused = is_refusal(text) if isinstance(text, str) else None

        citation_valid: bool | None = None
        if isinstance(text, str) and not refused and evidence_index:
            citation_valid = bool(
                validate_answer_citations(text, evidence_index).get("citation_valid")
            )

        verdict, supported, unsupported, uncertain = _citation_support(
            citation_labels.get(case_id, [])
        )

        records.append(
            CaseRecord(
                case_id=case_id,
                category=str(case.get("category") or "unknown"),
                chunking_strategy=chunking_strategy,
                retrieval_method=retrieval_method,
                top_k=top_k,
                reranking=reranking,
                simulated=simulated,
                retrieved_evidence=[
                    {
                        "chunk_id": item.get("chunk_id"),
                        "file": item.get("file"),
                        "locator": item.get("locator"),
                        "score": item.get("score"),
                        "text": item.get("text", ""),
                    }
                    for item in items
                ],
                hit_at_k=hit,
                reciprocal_rank=rank,
                recall_at_k=recall,
                generated_answer=text if isinstance(text, str) else None,
                answer_correct=answer_labels.get(case_id),
                citation_valid=citation_valid,
                citation_supported=verdict,
                refusal_correct=(
                    None if refused is None else (refused == expected_refusal)
                ),
                expected_refusal=expected_refusal,
                citation_support_supported=supported,
                citation_support_unsupported=unsupported,
                citation_support_uncertain=uncertain,
                citation_support_assessable=supported + unsupported,
                citation_support_total=supported + unsupported + uncertain,
            )
        )
    return records


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _subset_metrics(records: Sequence[CaseRecord]) -> dict[str, Any]:
    answerable = [record for record in records if record.hit_at_k is not None]
    hits = [1.0 if record.hit_at_k else 0.0 for record in answerable]
    rrs = [
        record.reciprocal_rank
        for record in answerable
        if record.reciprocal_rank is not None
    ]
    recalls = [
        record.recall_at_k for record in answerable if record.recall_at_k is not None
    ]

    judged = [
        record
        for record in records
        if record.answer_correct in ANSWER_LABELS
        and ANSWER_LABELS[record.answer_correct] is not None
    ]
    weighted = [ANSWER_LABELS[record.answer_correct] for record in judged]
    weighted = [score for score in weighted if score is not None]
    strict = sum(1 for record in judged if record.answer_correct == "correct")

    citations = [record for record in records if record.citation_valid is not None]
    supported = sum(record.citation_support_supported for record in records)
    unsupported = sum(record.citation_support_unsupported for record in records)
    uncertain = sum(record.citation_support_uncertain for record in records)
    assessable = supported + unsupported
    total_citations = supported + unsupported + uncertain
    citation_support_rate = (
        round(supported / assessable, 6) if assessable else None
    )

    unanswerable = [record for record in records if record.expected_refusal]
    answerable_refusals = [
        record
        for record in records
        if not record.expected_refusal and record.refusal_correct is not None
    ]
    refusal_ok = [record for record in unanswerable if record.refusal_correct]
    false_refusals = [
        record for record in answerable_refusals if record.refusal_correct is False
    ]

    refusal_scored = [r for r in unanswerable if r.refusal_correct is not None]
    refusal_accuracy = (
        round(len(refusal_ok) / len(refusal_scored), 6) if refusal_scored else None
    )
    false_refusal_rate = (
        round(len(false_refusals) / len(answerable_refusals), 6)
        if answerable_refusals
        else None
    )
    balanced = None
    if refusal_accuracy is not None and false_refusal_rate is not None:
        balanced = round(
            (refusal_accuracy + (1.0 - false_refusal_rate)) / 2.0, 6
        )

    return {
        "cases": len(records),
        "answerable_cases": len(answerable),
        "hit_at_k": _mean(hits),
        "mrr_at_k": _mean(rrs),
        "recall_at_k": _mean(recalls),
        "answer_correctness": _mean(weighted),
        "answer_strict_accuracy": (
            round(strict / len(judged), 6) if judged else None
        ),
        "cases_judged": len(judged),
        "citation_validity": (
            round(sum(1 for r in citations if r.citation_valid) / len(citations), 6)
            if citations
            else None
        ),
        "cases_with_citation_check": len(citations),
        "citation_support": citation_support_rate,
        "citation_support_rate": citation_support_rate,
        "supported_count": supported,
        "unsupported_count": unsupported,
        "uncertain_count": uncertain,
        "assessable_count": assessable,
        "total_citations": total_citations,
        "citation_support_total": total_citations,
        "citation_support_assessable": assessable,
        "citations_labelled": total_citations,
        "refusal_accuracy": refusal_accuracy,
        "false_refusal_rate": false_refusal_rate,
        "balanced_accuracy": balanced,
    }


def aggregate(records: Sequence[CaseRecord]) -> dict[str, Any]:
    """Overall metrics plus a breakdown per category."""
    payload: dict[str, Any] = {
        "simulated": any(record.simulated for record in records),
        "overall": _subset_metrics(records),
        "by_category": {},
    }
    for category in CATEGORIES:
        subset = [record for record in records if record.category == category]
        payload["by_category"][category] = _subset_metrics(subset)
    unknown = {record.category for record in records} - set(CATEGORIES)
    for category in sorted(unknown):
        subset = [record for record in records if record.category == category]
        payload["by_category"][category] = _subset_metrics(subset)
    return payload


METRIC_ROWS: tuple[tuple[str, str], ...] = (
    ("Hit@K", "hit_at_k"),
    ("MRR", "mrr_at_k"),
    ("Recall@K", "recall_at_k"),
    ("Answer Correctness", "answer_correctness"),
    ("Citation Validity", "citation_validity"),
    ("Citation Support", "citation_support"),
    ("Refusal Accuracy", "refusal_accuracy"),
)


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(report: Mapping[str, Any], cell: Mapping[str, Any]) -> str:
    """Render the aggregate report as markdown."""
    banner = (
        "> **SIMULATED — synthetic fixture, NOT experimental results.**\n"
        "> Values below validate the pipeline only. Do not cite them.\n\n"
        if report.get("simulated")
        else ""
    )
    lines = [
        "# 评测报告",
        "",
        banner,
        "## 实验配置",
        "",
        f"- chunking: `{cell.get('chunking_strategy')}`",
        f"- retrieval: `{cell.get('retrieval_method')}`",
        f"- top_k: `{cell.get('top_k')}`",
        f"- reranking: `{cell.get('reranking', 'off')}`",
        f"- dataset: `{cell.get('dataset_path', 'datasets/qa_cases_v2.jsonl')}`",
        f"- dataset sha256: `{cell.get('dataset_sha256', '-')}`",
        "",
        "## 总体与分题型结果",
        "",
        "| 指标 | 总体 | 直接事实 | 综合理解 | 细节定位 | 无答案 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    overall = report["overall"]
    by_category = report["by_category"]
    for label, key in METRIC_ROWS:
        cells = [overall.get(key)]
        for category in ("direct", "synthesis", "detail", "unanswerable"):
            cells.append(by_category.get(category, {}).get(key))
        lines.append(
            f"| {label} | " + " | ".join(_format(value) for value in cells) + " |"
        )
    lines.extend(
        [
            f"| 题目数 | {overall['cases']} | "
            + " | ".join(
                str(by_category.get(category, {}).get("cases", 0))
                for category in ("direct", "synthesis", "detail", "unanswerable")
            )
            + " |",
            "",
            "## 口径说明",
            "",
            "- `Hit@K` / `MRR` / `Recall@K` 只在有答案题上计算，且证据按 `file + 页码/章节` 匹配。",
            "- `Citation Validity` 只检查 `【n】` 是否落在本次检索到的证据编号内；"
            "`Citation Support` 是人工逐条标注“该引用是否真的支持这句话”，两者必须分开看。",
            "- 拒答样本不计入引用指标，避免把正确拒答误判为“引用无效”。",
            "- 拒答类指标只统计有预测/标签的样本，缺标签记 `—`，不按 0 处理。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    cell: Mapping[str, Any],
    records: Sequence[CaseRecord],
    report: Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Path]:
    """Write per-question JSONL, JSON report and markdown report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case = out_dir / "per_case.jsonl"
    with per_case.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    report_json = out_dir / "report.json"
    report_json.write_text(
        json.dumps(
            {"cell": dict(cell), "report": report},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_md = out_dir / "report.md"
    report_md.write_text(render_markdown(report, cell), encoding="utf-8")
    return {"per_case": per_case, "report_json": report_json, "report_md": report_md}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def index_by(records: Iterable[Mapping[str, Any]], key: str) -> dict[str, Any]:
    return {str(record[key]): record for record in records}
