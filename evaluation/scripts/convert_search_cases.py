"""Convert ``test-materials/search-cases.md`` into the unified QA format.

The markdown table stores human-readable locations ("第 25 页", "Headings 章节",
"第 26 或 27 张"). Turning those into machine-checkable locators is a judgement
call, so the mapping lives here explicitly, and the script *verifies* that every
question and reference answer still appears in the markdown before writing
anything. If the markdown changes, the conversion fails loudly instead of
silently drifting.

Original quotes are deliberately **not** filled in: ``quote`` stays ``null`` and
``quote_status`` stays ``pending`` until someone extracts the real text from the
document. Nothing here is invented.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "test-materials" / "search-cases.md"
DEFAULT_OUT = REPO_ROOT / "evaluation" / "datasets" / "qa_cases.jsonl"

# Hand-reviewed mapping: markdown row -> machine-checkable evidence.
# ``group`` semantics: units sharing a group are interchangeable alternatives.
CASES: list[dict[str, Any]] = [
    {
        "id": "sc-01",
        "md_row": 1,
        "question": "按二次探测依次把 89、18、49、58、79 插入大小为 10 的空哈希表，哪些槽位有值？",
        "question_type": "answerable",
        "reference_answer": "第 25 页。0→49、2→58、3→79、8→18、9→89，其余为空。",
        "expected_files": ["11-hash2.pdf"],
        "gold_evidence": [
            {
                "file": "11-hash2.pdf",
                "locator": {"kind": "page", "value": 25},
                "quote": None,
                "quote_status": "pending",
                "group": "slots",
            }
        ],
        "evidence_status": "locator_only",
        "notes": "原文片段（第 25 页表格）待抽取后补录；不要手工编造。",
    },
    {
        "id": "sc-02",
        "md_row": 2,
        "question": "原槽位被占后，二次探测的前四个偏移是多少？当表大小为素数且负载因子小于一半时，能保证什么？",
        "question_type": "answerable",
        "reference_answer": "第 24 张给出 1、4、9、16。第 26 或 27 张说明一定能找到空槽。",
        "expected_files": ["11-hash2.pptx"],
        "gold_evidence": [
            {
                "file": "11-hash2.pptx",
                "locator": {"kind": "slide", "value": 24},
                "quote": None,
                "quote_status": "pending",
                "group": "offsets",
            },
            {
                "file": "11-hash2.pptx",
                "locator": {"kind": "slide", "value": 26},
                "quote": None,
                "quote_status": "pending",
                "group": "guarantee",
            },
            {
                "file": "11-hash2.pptx",
                "locator": {"kind": "slide", "value": 27},
                "quote": None,
                "quote_status": "pending",
                "group": "guarantee",
            },
        ],
        "evidence_status": "locator_only",
        "notes": "两组证据；`guarantee` 组内第 26 或 27 张任一命中即可（原文写作「第 26 或 27 张」）。",
    },
    {
        "id": "sc-03",
        "md_row": 3,
        "question": "Mac 版 Word 的样式窗格中，Normal 和 Heading 1 之间是什么？截图列出了哪些标题级别？",
        "question_type": "answerable",
        "reference_answer": (
            "Headings 章节。中间是 No Spacing，"
            "标题级别为 Heading 1、Heading 2、Heading 3。"
        ),
        "expected_files": ["UTS-accessible-word.docx"],
        "gold_evidence": [
            {
                "file": "UTS-accessible-word.docx",
                "locator": {"kind": "section", "value": "Headings"},
                "quote": None,
                "quote_status": "pending",
                "group": "headings",
            }
        ],
        "evidence_status": "locator_only",
        "notes": "locator 用原文出现的章节标题 Headings（与 ingestion 的 section 元数据一致）。",
    },
    {
        "id": "sc-04",
        "md_row": 4,
        "question": "「桥梁工程的基本知识」模块支持哪些课程目标？学生有哪两项学习任务？",
        "question_type": "answerable",
        "reference_answer": "「教学模块」表的桥梁示例行。支持目标 1、4、5。学生在蓝墨云班课平台观看桥梁视频，并撰写桥梁发展与创新报告。",
        "expected_files": ["USST-syllabus.docx"],
        "gold_evidence": [
            {
                "file": "USST-syllabus.docx",
                "locator": {"kind": "table_row", "value": "教学模块"},
                "quote": None,
                "quote_status": "pending",
                "group": "module_table",
            }
        ],
        "evidence_status": "locator_only",
        "notes": "表格题：Word 原生表格单独成块，locator 记表所在章节「教学模块」。",
    },
    {
        "id": "sc-05",
        "md_row": 5,
        "question": "打开 Check Accessibility 用哪个选项卡？表格列用交替底色的选项叫什么？",
        "question_type": "answerable",
        "reference_answer": "第 4 张的 Review 选项卡和第 17 张的 Banded Columns。",
        "expected_files": ["UTS-accessible-powerpoint.pptx"],
        "gold_evidence": [
            {
                "file": "UTS-accessible-powerpoint.pptx",
                "locator": {"kind": "slide", "value": 4},
                "quote": None,
                "quote_status": "pending",
                "group": "tab",
            },
            {
                "file": "UTS-accessible-powerpoint.pptx",
                "locator": {"kind": "slide", "value": 17},
                "quote": None,
                "quote_status": "pending",
                "group": "banded",
            },
        ],
        "evidence_status": "locator_only",
        "notes": "两个独立证据组，都必须命中（与 sc-02 的备选组不同）。",
    },
    {
        "id": "sc-06",
        "md_row": 6,
        "question": "云盘支持哪些课件格式？「打开云盘课件内容上课」这页截图中，与试卷和授权有关的列表项叫什么？",
        "question_type": "answerable",
        "reference_answer": (
            "第 8 页。格式为 EDB 板书、PPT、PDF、Word、图片、音频、视频和 TXT，"
            "截图里有「试卷资源」和「授权资源」。"
        ),
        "expected_files": ["USTC-ClassIn.pdf"],
        "gold_evidence": [
            {
                "file": "USTC-ClassIn.pdf",
                "locator": {"kind": "page", "value": 8},
                "quote": None,
                "quote_status": "pending",
                "group": "page8",
            }
        ],
        "evidence_status": "locator_only",
        "notes": "答案同时来自第 8 页正文与截图 OCR 文本，属跨模态证据，原文片段待补。",
    },
]


def _squash(text: str) -> str:
    """Collapse whitespace so markdown line wrapping cannot break matching."""
    return re.sub(r"\s+", "", text).replace("`", "")


def parse_markdown(source: Path) -> list[str]:
    """Return the question cells of the acceptance table, in order."""
    questions: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        if cells[0] in {"资料", "---"} or set(cells[0]) <= {"-", ":"}:
            continue
        questions.append(cells[1])
    return questions


def build_cases(source: Path) -> list[dict[str, Any]]:
    markdown = source.read_text(encoding="utf-8")
    squashed_markdown = _squash(markdown)
    questions = parse_markdown(source)
    if len(questions) != len(CASES):
        raise SystemExit(
            f"{source} has {len(questions)} question rows, expected {len(CASES)}. "
            "Update evaluation/scripts/convert_search_cases.py before converting."
        )

    records: list[dict[str, Any]] = []
    for case, markdown_question in zip(CASES, questions, strict=True):
        for field in ("question", "reference_answer"):
            if _squash(case[field]) not in squashed_markdown:
                raise SystemExit(
                    f"{case['id']}: {field} not found verbatim in {source.name}; "
                    "the markdown changed, re-check the mapping."
                )
        if _squash(case["question"]) != _squash(markdown_question):
            raise SystemExit(
                f"{case['id']}: question differs from markdown row "
                f"{case['md_row']}: {markdown_question!r}"
            )
        record = {key: value for key, value in case.items() if key != "md_row"}
        record["source"] = f"test-materials/search-cases.md#{case['md_row']}"
        records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    records = build_cases(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    pending = sum(
        1
        for record in records
        for unit in record["gold_evidence"]
        if unit["quote_status"] == "pending"
    )
    print(
        f"wrote {len(records)} cases to {args.out} "
        f"({pending} evidence quotes pending, no quotes invented)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
