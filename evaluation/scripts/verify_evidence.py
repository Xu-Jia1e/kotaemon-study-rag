"""Verify every dataset quote against the real source documents.

This is the quality gate for the QA set: a case may only claim ``quote`` if the
text really appears in the cited page / slide / section. It also checks the
"no answer" cases:

* ``checked_scope.max_mentions``  - term must not be mentioned more often than
  allowed (e.g. ``cuckoo`` appears exactly once, as a name only);
* ``checked_scope.absent_phrases`` - phrases that would make the question
  answerable must be absent everywhere;
* ``checked_scope.blank_fields``   - the field exists but is blank (no digits).

Documents are the checksum-verified samples downloaded by
``test-materials/download_samples.py`` into ``downloads/real-samples/``.
Missing files are reported and skipped unless ``--strict`` is used.

Usage::

    python -m evaluation.scripts.verify_evidence
    python -m evaluation.scripts.verify_evidence \\
        --dataset evaluation/datasets/qa_cases_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "evaluation" / "datasets" / "qa_cases_v2.jsonl"
DEFAULT_SAMPLES = REPO_ROOT / "downloads" / "real-samples"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON ({exc})") from exc
    return cases


def extract_document(path: Path) -> dict[str, Any]:
    """Return ``{"kind", "units": {locator_value: text}, "rows": [[cell]]}``."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        units: dict[Any, str] = {}
        for number, page in enumerate(reader.pages, start=1):
            units[number] = norm(page.extract_text() or "")
        return {"kind": "pdf", "units": units, "rows": []}
    if suffix == ".pptx":
        from pptx import Presentation

        presentation = Presentation(str(path))
        units = {}
        for number, slide in enumerate(presentation.slides, start=1):
            chunks: list[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    chunks.append(shape.text_frame.text)
                if getattr(shape, "has_table", False) and shape.has_table:
                    for row in shape.table.rows:
                        chunks.append(" ".join(cell.text for cell in row.cells))
            units[number] = norm(" ".join(chunks))
        return {"kind": "pptx", "units": units, "rows": []}
    if suffix == ".docx":
        from docx import Document as DocxDocument

        document = DocxDocument(str(path))
        rows: list[list[str]] = []
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text for cell in row.cells]
                rows.append(cells)
        text = norm(" ".join(paragraphs + [" ".join(row) for row in rows]))
        return {"kind": "docx", "units": {"__whole__": text}, "rows": rows}
    raise ValueError(f"unsupported document type: {path}")


def build_index(samples_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    index: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    expected = (
        "11-hash2.pdf",
        "11-hash2.pptx",
        "USTC-ClassIn.pdf",
        "USST-syllabus.docx",
    )
    for name in expected:
        path = samples_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        index[name] = extract_document(path)
    return index, missing


def _whole_text(document: dict[str, Any]) -> str:
    return norm(" ".join(str(value) for value in document["units"].values()))


def verify_case(
    case: dict[str, Any], index: dict[str, dict[str, Any]]
) -> list[str]:
    problems: list[str] = []
    case_id = case.get("id", "<missing>")
    for unit in case.get("gold_evidence") or []:
        file = unit.get("file")
        document = index.get(file)
        if document is None:
            problems.append(f"{case_id}: document not available: {file}")
            continue
        quote = norm(unit.get("quote") or "")
        if not quote:
            problems.append(f"{case_id}: empty quote in {file}")
            continue
        locator = unit.get("locator") or {}
        if locator.get("kind") == "table_row":
            haystack = _whole_text(document)
        else:
            haystack = document["units"].get(locator.get("value"), "")
        if quote not in haystack:
            problems.append(
                f"{case_id}: quote missing in {file} "
                f"{locator.get('kind')}={locator.get('value')}"
            )
    return problems


def verify_unanswerable(
    case: dict[str, Any], index: dict[str, dict[str, Any]]
) -> list[str]:
    problems: list[str] = []
    case_id = case.get("id", "<missing>")
    scope = case.get("checked_scope") or {}
    files = scope.get("files") or []

    for file in files:
        if file not in index:
            problems.append(f"{case_id}: checked_scope file not available: {file}")

    for term, maximum in (scope.get("max_mentions") or {}).items():
        count = sum(
            _whole_text(index[file]).lower().count(term.lower())
            for file in files
            if file in index
        )
        if count > maximum:
            problems.append(
                f"{case_id}: {term!r} mentioned {count} times, allowed {maximum}"
            )

    for phrase in scope.get("absent_phrases") or []:
        for file in files:
            if file in index and phrase.lower() in _whole_text(index[file]).lower():
                problems.append(f"{case_id}: forbidden phrase {phrase!r} in {file}")

    blank = scope.get("blank_fields")
    if blank:
        document = index.get(blank.get("file"))
        if document is None:
            problems.append(f"{case_id}: blank_fields document missing")
        else:
            for keyword in blank.get("row_keywords") or []:
                matched = [
                    row
                    for row in document["rows"]
                    if any(keyword in cell for cell in row)
                ]
                if not matched:
                    problems.append(f"{case_id}: field {keyword!r} not found")
                for row in matched:
                    joined = " ".join(row)
                    has_digits = bool(re.search(r"\d", joined))
                    if blank.get("must_not_contain_digits") and has_digits:
                        problems.append(
                            f"{case_id}: field {keyword!r} is filled: {joined!r}"
                        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail when a referenced document is missing",
    )
    args = parser.parse_args(argv)

    cases = load_cases(args.dataset)
    index, missing = build_index(args.samples_dir)
    for name in missing:
        print(f"WARNING document missing: {args.samples_dir / name}", file=sys.stderr)
    if missing and args.strict:
        return 2

    problems: list[str] = []
    checked = 0
    for case in cases:
        if case.get("question_type") == "unanswerable":
            problems.extend(verify_unanswerable(case, index))
        else:
            problems.extend(verify_case(case, index))
        checked += 1

    for problem in problems:
        print(f"ERROR {problem}", file=sys.stderr)

    quotes = sum(len(case.get("gold_evidence") or []) for case in cases)
    print(
        f"{args.dataset.name}: {checked} cases, {quotes} evidence quotes, "
        f"{len(index)} documents loaded, {len(problems)} error(s)"
    )
    return 1 if problems else 0


def _iter_cases(path: Path) -> Iterable[dict[str, Any]]:  # pragma: no cover - helper
    return iter(load_cases(path))


if __name__ == "__main__":
    sys.exit(main())
