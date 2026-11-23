"""Generate a human grading sheet for any number of answered cases.

Generalises the 4-case pilot sheet: works from a batch ``answers.json`` plus the
retrieval results, supports ``--case-ids``/``--all``, marks failed or missing
cases without inventing answers, and never overwrites an existing sheet unless
``--force`` is given (so the pilot ``grading.md`` stays untouched).

No API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CATEGORY_LABEL = {
    "direct": "直接事实",
    "synthesis": "综合理解",
    "detail": "细节定位",
    "unanswerable": "无答案",
}
TYPE_LABEL = {"answerable": "有答案", "unanswerable": "无答案"}


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _block(text: str) -> str:
    return "```text\n" + text.strip() + "\n```"


def _locator(item: dict) -> dict:
    return item.get("locator") or {}


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def _citation_table(
    case: dict, answer: dict, by_chunk: dict[str, dict]
) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    details: list[str] = []
    validation = answer.get("citation_validation") or {}
    cited = validation.get("cited_indices") or []
    evidence_index = {
        entry["index"]: entry for entry in answer.get("evidence_index", [])
    }
    if not cited:
        lines.append("无引用（拒答或未产生引用）。")
        return lines, details

    lines.append("| 编号 | 文件 | 页码/章节 | chunk_id | 编号有效 | 含 gold 引文（自动提示） |")
    lines.append("|---|---|---|---|---|---|")
    gold_quotes = [
        _norm(unit.get("quote") or "")
        for unit in case.get("gold_evidence") or []
        if unit.get("quote")
    ]
    for index in cited:
        entry = evidence_index.get(index, {})
        file_name = entry.get("file_name") or ""
        page = entry.get("page_label")
        if page is None:
            kind, place = "section", entry.get("section")
        else:
            kind = "slide" if str(file_name).endswith(".pptx") else "page"
            place = page
        chunk = by_chunk.get(str(entry.get("doc_id")))
        chunk_text = _norm((chunk or {}).get("text", ""))
        contained = any(quote and quote in chunk_text for quote in gold_quotes)
        lines.append(
            f"| 【{index}】 | `{file_name}` | {kind}={place} | "
            f"`{entry.get('doc_id')}` | ✅ | {'是' if contained else '否'} |"
        )
        details.append(
            f"<details><summary>【{index}】{file_name} {place}</summary>\n\n"
            + _block(chunk.get("text", "") if chunk else "（未找到 chunk 文本）")
            + f"\n\nretrieval score: `{(chunk or {}).get('score')}`\n\n</details>"
        )
    lines.append("")
    lines.append("> 「含 gold 引文」仅做字符串包含检测，不作为评分依据；请以原文为准。")
    return lines, details


def render(
    cases: list[dict],
    answers: dict,
    retrieval: dict,
    failures: dict | None,
    missing: list[str],
) -> str:
    by_id = {case["id"]: case for case in cases}
    retrieval_by_case = {case["case_id"]: case for case in retrieval.get("cases", [])}
    lines: list[str] = []
    add = lines.append

    add("# 人工评分表")
    add("")
    add("> 由已记录的答案离线生成；没有调用任何 API，也没有填写人工评分。")
    add("> `citation_valid`（编号是否存在）与 `citation_supported`（引用是否支持）是两件事。")
    add("")
    add("## 运行状态")
    add("")
    add(f"- 结果类型：{'模拟（不可当作真实实验）' if answers.get('simulated') else '真实运行'}")
    add(f"- 已作答题目：{len(answers.get('cases', {}))}")
    if failures and failures.get("cases"):
        failed_ids = sorted(failures["cases"])
        add(f"- 失败题目：{len(failed_ids)}（{', '.join(failed_ids)}）")
    if missing:
        add(f"- 未运行题目：{len(missing)}（{', '.join(sorted(missing))}）")
    add("")
    add("## 评分口径")
    add("")
    add("| 字段 | 含义 | 取值 |")
    add("|---|---|---|")
    add("| `answer_correct` | 对照参考答案的事实正确性 | 正确 / 部分正确 / 错误 / 待判断 |")
    add("| `citation_supported` | 引用内容是否支持对应句子（不是编号有效性） "
        "| 全部支持 / 部分支持 / 不支持 / 不适用（无引用）/ 待判断 |")
    add("| `refusal_correct` | 无答案题是否恰当拒答；有答案题通常不适用 | 正确 / 错误 / 不适用 |")
    add("| `comments` | 缺失要点、不支持的引用等 | 自由文本 |")
    add("")

    for case_id, case in by_id.items():
        answer = answers.get("cases", {}).get(case_id)
        if answer is None:
            add("---")
            add("")
            add(f"## {case_id}（未作答）")
            add("")
            add("该题没有可用答案（失败或未运行），本次不生成评分项。")
            add("")
            continue

        category = CATEGORY_LABEL.get(case.get("category"), case.get("category"))
        if case.get("category") == "unanswerable":
            add("---")
            add("")
            add(f"## {case_id}（无答案题）")
        else:
            question_type = TYPE_LABEL.get(
                case.get("question_type"), case.get("question_type")
            )
            add("---")
            add("")
            add(f"## {case_id}（{category}题 / {question_type}）")
        add("")
        add("### 1. 原始问题")
        add("")
        add(case["question"])
        add("")
        add("### 2. 参考答案")
        add("")
        add(case["reference_answer"])
        add("")
        if case.get("gold_evidence"):
            add("**标注证据（数据集，已逐字核验）**")
            add("")
            for unit in case["gold_evidence"]:
                locator = unit["locator"]
                add(
                    f"- `{unit['file']}` · {locator['kind']}=`{locator['value']}` "
                    f"（组 `{unit.get('group')}`）"
                )
                if unit.get("quote"):
                    add(_block(unit["quote"]))
            add("")
        add("### 3. 模型实际答案")
        add("")
        add(_block(answer["answer_text"]))
        add("")
        add(f"- citation_status：`{answer.get('citation_status')}`；"
            f"延迟：{answer.get('latency_s')}s；尝试次数：{answer.get('attempts')}")
        add("")

        results = retrieval_by_case.get(case_id, {}).get("results", [])
        by_chunk = {str(item.get("chunk_id")): item for item in results}
        add("### 4. 引用编号 → 真实来源")
        add("")
        table_lines, detail_lines = _citation_table(case, answer, by_chunk)
        lines.extend(table_lines)
        add("")
        add("### 5. 被引 chunk 的原始证据文本")
        add("")
        if detail_lines:
            lines.extend(detail_lines)
        else:
            add("无被引 chunk。")
        add("")

        add("### 6. 检索命中情况")
        add("")
        if case.get("question_type") == "unanswerable":
            add("- 不适用：无答案题不参与 Hit@K 统计。")
        else:
            add("- 见 `report_auto.json` 中的逐题检索指标。")
        add("")

        if case.get("question_type") == "unanswerable":
            add("### 7. 无答案题：预期与实际")
            add("")
            add("- 预期：明确拒答，不编造。")
            add("")
            add("实际输出：")
            add("")
            add(_block(answer["answer_text"]))
            add("")

        add("### 8. 人工评分（请填写）")
        add("")
        add("| 字段 | 评分 |")
        add("|---|---|")
        add("| `answer_correct` | ☐ 正确　☐ 部分正确　☐ 错误　☐ 待判断 |")
        add("| `citation_supported` | ☐ 全部支持　☐ 部分支持　☐ 不支持　☐ 不适用（无引用）　☐ 待判断 |")
        add("| `refusal_correct` | ☐ 正确　☐ 错误　☐ 不适用 |")
        add("| `comments` |  |")
        add("")

    add("---")
    add("")
    add("## 评分汇总（人工填写）")
    add("")
    add("| case | answer_correct | citation_supported | refusal_correct | comments |")
    add("|---|---|---|---|---|")
    for case_id in by_id:
        if case_id in answers.get("cases", {}):
            add(f"| `{case_id}` |  |  |  |  |")
    add("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        default=REPO_ROOT / "evaluation/datasets/qa_cases_v2.jsonl",
    )
    parser.add_argument(
        "--case-ids", help="comma separated subset; default all dataset cases"
    )
    parser.add_argument(
        "--only-answered",
        action="store_true",
        help="render only cases that have answers (pilot mode)",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    answers = json.loads(args.answers.read_text(encoding="utf-8"))
    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    cases = _load_jsonl(args.cases)
    by_id = {case["id"]: case for case in cases}

    if args.case_ids:
        wanted = [item.strip() for item in args.case_ids.split(",") if item.strip()]
    elif args.only_answered:
        wanted = [case_id for case_id in by_id if case_id in answers.get("cases", {})]
    else:
        wanted = list(by_id)
    selected = [by_id[case_id] for case_id in wanted if case_id in by_id]
    missing = [case_id for case_id in wanted if case_id not in answers.get("cases", {})]

    failures = None
    failures_path = args.answers.parent / "failures.json"
    if failures_path.is_file():
        failures = json.loads(failures_path.read_text(encoding="utf-8"))

    out = args.out or args.answers.parent / "grading_all.md"
    if out.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite existing sheet: {out} (use --force or --out)"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render(selected, answers, retrieval, failures, missing), encoding="utf-8"
    )
    print(f"wrote {out} ({len(selected)} cases, {len(missing)} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
