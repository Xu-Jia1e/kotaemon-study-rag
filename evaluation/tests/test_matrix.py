from __future__ import annotations

import hashlib

from evaluation.matrix import (
    CHUNKING_PROFILES,
    DATASET_PATH,
    RERANKING_OPTIONS,
    RETRIEVAL_METHODS,
    TOP_K_VALUES,
    build_matrix,
    dataset_manifest,
    status_table,
    summarise,
)


def test_matrix_has_all_18_cells():
    cells = build_matrix()
    expected = len(CHUNKING_PROFILES) * len(RETRIEVAL_METHODS) * len(TOP_K_VALUES)
    assert len(cells) == expected
    assert len(cells) == 18
    assert len({cell.cell_id for cell in cells}) == 18


def test_every_cell_is_blocked_and_says_why():
    cells = build_matrix()
    assert all(cell.status == "blocked" for cell in cells)
    assert summarise(cells)["ready_cell_ids"] == []
    for cell in cells:
        assert any(name.startswith("chunking:") for name in cell.blocked_by)
        assert any(name.startswith("retrieval:") for name in cell.blocked_by)
        assert cell.reasons
        assert all(reason.strip() for reason in cell.reasons)


def test_top_k_values_are_3_5_10():
    ks = sorted({cell.top_k for cell in build_matrix()})
    assert ks == [3, 5, 10]


def test_reranking_is_off_by_default_and_on_is_blocked():
    default_cells = build_matrix()
    assert {cell.reranking for cell in default_cells} == {"off"}

    rerank_cells = build_matrix(rerankings=["on"])
    assert len(rerank_cells) == 18
    assert all(
        any(name.startswith("reranking:") for name in cell.blocked_by)
        for cell in rerank_cells
    )
    assert RERANKING_OPTIONS["on"].status == "blocked"


def test_all_cells_share_the_same_dataset():
    manifest = dataset_manifest()
    assert manifest["cases"] == 24
    assert manifest["path"] == "evaluation/datasets/qa_cases_v2.jsonl"
    expected = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    assert manifest["sha256"] == expected


def test_status_table_lists_each_cell():
    table = status_table(build_matrix())
    lines = table.splitlines()
    assert lines[0].startswith("| cell |")
    assert len(lines) == 2 + 18
    assert "`c1024-256__vector__k3__rr-off`" in table


def test_blocking_dimension_counts_only_affect_their_cells():
    summary = summarise(build_matrix())
    assert summary["cells_total"] == 18
    assert summary["by_status"] == {"blocked": 18}
    counts = summary["blocking_dimensions"]
    # 2 chunkings x 3 retrieval x 3 k: each chunking blocks 9 cells.
    assert counts["chunking:1024-256"] == 9
    assert counts["chunking:512-128"] == 9
    # Each retrieval method appears in 2 chunkings x 3 k = 6 cells.
    assert counts["retrieval:vector"] == 6
    assert counts["retrieval:hybrid"] == 6
    assert counts["retrieval:rrf"] == 6
    # Reranking is off (ready), so it blocks nothing in the default matrix.
    assert "reranking:off" not in counts
