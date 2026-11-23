"""Validate an evaluation dataset (JSONL) against the documented schema.

Exit code 0 means structurally valid; the target-size check is a warning unless
``--strict-target`` is given, so the harness can be run before the 20-case set is
complete without pretending it is finished.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

VALID_QUESTION_TYPES = {"answerable", "unanswerable", "partially_answerable"}
VALID_LOCATOR_KINDS = {"page", "slide", "section", "table_row"}
REQUIRED_FIELDS = (
    "id",
    "source",
    "question",
    "question_type",
    "reference_answer",
    "expected_files",
    "gold_evidence",
    "evidence_status",
)
VALID_EVIDENCE_STATUS = {"locator_only", "quote_verified", "pending"}


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"{path}:{line_number}: expected a JSON object")
            cases.append(record)
    return cases


def validate_cases(cases: Iterable[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)``."""
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for case in cases:
        case_id = str(case.get("id", "<missing-id>"))
        for field in REQUIRED_FIELDS:
            if field not in case:
                errors.append(f"{case_id}: missing field {field!r}")
        if case_id in seen:
            errors.append(f"{case_id}: duplicate id")
        seen.add(case_id)

        question_type = case.get("question_type")
        if question_type not in VALID_QUESTION_TYPES:
            errors.append(f"{case_id}: invalid question_type {question_type!r}")

        if case.get("evidence_status") not in VALID_EVIDENCE_STATUS:
            errors.append(
                f"{case_id}: invalid evidence_status "
                f"{case.get('evidence_status')!r}"
            )

        if not case.get("expected_files"):
            errors.append(f"{case_id}: expected_files must not be empty")

        units = case.get("gold_evidence")
        if units is None:
            errors.append(f"{case_id}: gold_evidence must be a list")
            continue
        if not isinstance(units, list):
            errors.append(f"{case_id}: gold_evidence must be a list")
            continue
        if question_type == "answerable" and not units:
            errors.append(
                f"{case_id}: answerable case needs at least one gold evidence unit"
            )
        if question_type == "unanswerable" and units:
            warnings.append(f"{case_id}: unanswerable case has gold evidence units")

        for position, unit in enumerate(units):
            prefix = f"{case_id}: gold_evidence[{position}]"
            if not unit.get("file"):
                errors.append(f"{prefix}: file is empty")
            locator = unit.get("locator") or {}
            if locator.get("kind") not in VALID_LOCATOR_KINDS:
                errors.append(f"{prefix}: invalid locator.kind {locator.get('kind')!r}")
            if locator.get("value") in (None, ""):
                errors.append(f"{prefix}: locator.value is empty")
            if unit.get("quote_status") not in {"pending", "verified"}:
                errors.append(
                    f"{prefix}: invalid quote_status {unit.get('quote_status')!r}"
                )
            quote = unit.get("quote")
            if unit.get("quote_status") == "pending" and quote is not None:
                errors.append(f"{prefix}: quote_status=pending requires quote=null")
            if unit.get("quote_status") == "verified" and not quote:
                errors.append(f"{prefix}: quote_status=verified requires a quote")

    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-cases", type=int, default=1)
    parser.add_argument("--target-cases", type=int, default=20)
    parser.add_argument("--strict-target", action="store_true")
    args = parser.parse_args(argv)

    cases = load_cases(args.dataset)
    errors, warnings = validate_cases(cases)

    for warning in warnings:
        print(f"WARNING {warning}", file=sys.stderr)
    for error in errors:
        print(f"ERROR {error}", file=sys.stderr)

    answerable = sum(1 for case in cases if case.get("question_type") == "answerable")
    unanswerable = sum(
        1 for case in cases if case.get("question_type") == "unanswerable"
    )
    print(
        f"{args.dataset}: {len(cases)} cases "
        f"(answerable={answerable}, unanswerable={unanswerable}), "
        f"{len(errors)} error(s), {len(warnings)} warning(s)"
    )

    if len(cases) < args.min_cases:
        print(
            f"ERROR only {len(cases)} cases, --min-cases is {args.min_cases}",
            file=sys.stderr,
        )
        return 2
    if len(cases) < args.target_cases:
        message = (
            f"dataset has {len(cases)} cases, target is {args.target_cases}; "
            "remaining cases must be authored from real documents (see "
            "evaluation/datasets/candidates.md)"
        )
        level = "ERROR" if args.strict_target else "WARNING"
        print(f"{level} {message}", file=sys.stderr)
        if args.strict_target:
            return 2
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
