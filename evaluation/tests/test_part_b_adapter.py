from __future__ import annotations

import json

import pytest

from evaluation.adapters.part_b import PartBRetrievalAdapter, _parse_payload

PAYLOAD = {
    "question": "q",
    "model": "text-embedding-v4",
    "dimensions": 2048,
    "retrieval_mode": "vector",
    "results": [
        {
            "id": "chunk-1",
            "file_name": "11-hash2.pdf",
            "page": 25,
            "section": None,
            "score": 0.59,
            "score_type": "vector",
            "text": "SYNTHETIC FIXTURE",
        },
        {
            "id": "chunk-2",
            "file_name": "11-hash2.pdf",
            "page": 17,
            "section": None,
            "score": 0.62,
            "score_type": "vector",
            "text": "SYNTHETIC FIXTURE",
        },
    ],
}


def test_adapter_maps_part_b_results_to_items():
    seen: list[list[str]] = []

    def runner(args):
        seen.append(list(args))
        return json.dumps(PAYLOAD)

    adapter = PartBRetrievalAdapter(index_dir="unused", runner=runner)
    items = adapter.retrieve("哈希表问题", top_k=5)
    assert [item.chunk_id for item in items] == ["chunk-1", "chunk-2"]
    assert items[0].file == "11-hash2.pdf"
    assert items[0].locator == {"kind": "page", "value": 25}
    assert items[0].score == pytest.approx(0.59)
    assert "--mode" in seen[0] and seen[0][seen[0].index("--mode") + 1] == "vector"


def test_adapter_uses_expected_file_when_single():
    seen: list[list[str]] = []

    def runner(args):
        seen.append(list(args))
        return json.dumps(PAYLOAD)

    adapter = PartBRetrievalAdapter(index_dir="unused", runner=runner)
    adapter.retrieve("q", top_k=3, case={"expected_files": ["11-hash2.pdf"]})
    assert "--file" in seen[0]
    assert seen[0][seen[0].index("--file") + 1] == "11-hash2.pdf"


def test_parser_tolerates_log_prefix():
    noisy = "some log line\n" + json.dumps(PAYLOAD) + "\n"
    assert _parse_payload(noisy)["results"][0]["id"] == "chunk-1"


def test_parser_rejects_output_without_results():
    adapter = PartBRetrievalAdapter(
        index_dir="unused", runner=lambda args: json.dumps({"question": "q"})
    )
    with pytest.raises(ValueError):
        adapter.retrieve("q", top_k=1)


def test_missing_index_directory_fails_fast_without_runner():
    adapter = PartBRetrievalAdapter(index_dir="does-not-exist")
    with pytest.raises(FileNotFoundError):
        adapter.retrieve("q", top_k=1)
