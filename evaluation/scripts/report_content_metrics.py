"""Report content-level retrieval metrics next to the strict ones.

Reads an existing 30-case ``retrieval.json`` (no embedding calls, no re-retrieval),
writes ``content_metrics.json`` and a human-review markdown for the four cases whose
strict match failed. ``retrieval.json`` itself is never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETRIEVAL = Path(
    r"E:/dsh_workplace/.eval-data/retrieval-real-30/1024-256-vector-k5/retrieval.json"
)
DEFAULT_CASES = REPO_ROOT / "evaluation" / "datasets" / "qa_cases_v2.jsonl"
FOCUS_IDS = ("v2-d05", "v2-d10", "v2-s08", "v2-l05")


def _load_cases(path: Path) -> dict[str, dict]:
    return {
        json.loads(line)["id"]: json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _md_quote(text: str, limit: int = 220) -> str:
    body = " ".join(str(text).split())
    return body if len(body) <= limit else body[:limit] + "…"


def render_focus_markdown(payload: dict, cases: dict[str, dict]) -> str:
    lines = [
        "# 内容级匹配依据（4 题）",
        "",
        "> 依据：数据集中已核验的 gold `quote` 是否逐字出现在 Top-5 chunk 文本中。",
        "> 严格指标（文件+页码）未改动；本表只做内容级复核。",
        "",
        "| case | 严格 Hit@5 | 内容 Hit@5 | 内容 Recall@5 |",
        "|---|---:|---:|---:|",
    ]
    strict = {row["case_id"]: row for row in payload["strict_metrics"]["per_case"]}
    for case_id in FOCUS_IDS:
        row = payload["by_case"].get(case_id)
        if not row:
            continue
        strict_hit = strict.get(case_id, {}).get("hit")
        lines.append(
            f"| `{case_id}` | {strict_hit} | {row['hit']} | "
            f"{row['content_recall']:.3f} |"
        )
    lines.append("")

    for case_id in FOCUS_IDS:
        row = payload["by_case"].get(case_id)
        if not row:
            continue
        lines.extend([f"## {case_id}", "", f"问题：{cases[case_id]['question']}", ""])
        for entry in row["evidence"]:
            status = "✅ 命中" if entry["matched"] else "❌ 未命中"
            lines.append(f"### {entry['group']} · {status}")
            lines.append("")
            lines.append(f"- gold：`{entry['gold_file']}` · {entry['gold_locator']}")
            lines.append(f"- gold 原文：{_md_quote(entry['quote'])}")
            chunk = entry["matched_chunk"]
            if chunk:
                locator = chunk["locator"] or {}
                lines.append(
                    f"- 命中 chunk：`{chunk['file']}` · "
                    f"{locator.get('kind')}={locator.get('value')} · "
                    f"`{chunk['chunk_id']}` · 排名 {chunk['rank']}"
                )
            lines.append("")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)

    from evaluation.metrics.content import content_report

    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    if retrieval.get("simulated"):
        raise SystemExit("refusing to score simulated retrieval as real data")
    cases = _load_cases(args.cases)
    rankings = {row["case_id"]: row["results"] for row in retrieval["cases"]}
    selected = [cases[case_id] for case_id in rankings if case_id in cases]
    k = int(retrieval.get("top_k") or 5)

    content = content_report(selected, rankings, k)
    by_case = {row["case_id"]: row for row in content["per_case"]}
    payload = {
        "simulated": False,
        "retrieval_file": str(args.retrieval),
        "k": k,
        "strict_metrics": retrieval["metrics"],
        "content_metrics": {
            "content_hit_at_k": content["content_hit_at_k"],
            "content_recall_at_k": content["content_recall_at_k"],
            "cases_scored": content["cases_scored"],
        },
        "by_case": by_case,
        "focus_ids": list(FOCUS_IDS),
    }

    out_dir = args.out_dir or args.retrieval.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "content_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md_path = out_dir / "content_evidence_4cases.md"
    md_path.write_text(render_focus_markdown(payload, cases), encoding="utf-8")

    strict = payload["strict_metrics"]
    print(
        f"strict:     Hit@{k}={strict['hit_at_k']} MRR@{k}={strict['mrr_at_k']} "
        f"Recall@{k}={strict['evidence_recall_at_k']}"
    )
    print(
        f"content:    Hit@{k}={content['content_hit_at_k']} "
        f"Recall@{k}={content['content_recall_at_k']} "
        f"(cases={content['cases_scored']})"
    )
    strict_by_case = {row["case_id"]: row for row in strict["per_case"]}
    for case_id in FOCUS_IDS:
        row = by_case.get(case_id)
        if row:
            print(
                f"  {case_id}: strict_hit={strict_by_case.get(case_id, {}).get('hit')} "
                f"content_hit={row['hit']} recall={row['content_recall']}"
            )
    print(f"-> {metrics_path}")
    print(f"-> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
