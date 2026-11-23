from __future__ import annotations

import json
from pathlib import Path

from evaluation.scripts.convert_search_cases import (
    DEFAULT_OUT,
    DEFAULT_SOURCE,
    build_cases,
)
from evaluation.scripts.validate_dataset import load_cases, validate_cases


def test_converted_dataset_is_schema_valid():
    cases = load_cases(DEFAULT_OUT)
    errors, warnings = validate_cases(cases)
    assert errors == []
    assert warnings == []
    assert len(cases) == 6
    assert all(case["question_type"] == "answerable" for case in cases)


def test_every_pending_quote_is_null():
    for case in load_cases(DEFAULT_OUT):
        for unit in case["gold_evidence"]:
            if unit["quote_status"] == "pending":
                assert unit["quote"] is None, case["id"]


def test_slide_question_keeps_alternative_group():
    cases = {case["id"]: case for case in load_cases(DEFAULT_OUT)}
    units = cases["sc-02"]["gold_evidence"]
    groups: dict[str, list[int]] = {}
    for unit in units:
        groups.setdefault(unit["group"], []).append(unit["locator"]["value"])
    assert groups == {"offsets": [24], "guarantee": [26, 27]}


def test_two_part_question_requires_both_groups():
    cases = {case["id"]: case for case in load_cases(DEFAULT_OUT)}
    groups = {unit["group"] for unit in cases["sc-05"]["gold_evidence"]}
    assert groups == {"tab", "banded"}


def test_conversion_is_reproducible_from_markdown():
    rebuilt = build_cases(DEFAULT_SOURCE)
    on_disk = load_cases(DEFAULT_OUT)
    assert json.dumps(rebuilt, ensure_ascii=False) == json.dumps(
        on_disk, ensure_ascii=False
    )


def test_dataset_path_exists():
    assert DEFAULT_OUT.is_file()
    assert Path(DEFAULT_OUT).parent.name == "datasets"
