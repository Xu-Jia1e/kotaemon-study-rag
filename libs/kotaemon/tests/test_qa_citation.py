import pytest

from kotaemon.base import RetrievedDocument
from kotaemon.indices.qa.citation import CiteEvidence
from kotaemon.indices.qa.citation_qa import (
    build_abstention_prompt,
    build_citation_metadata,
)
from kotaemon.indices.qa.citation_qa_inline import AnswerWithInlineCitation
from kotaemon.indices.qa.format_context import (
    TRUNCATION_MARKER,
    PrepareEvidencePipeline,
)
from kotaemon.indices.qa.utils import (
    ABSTAIN_MARKER,
    find_start_end_phrase,
    find_text,
    is_abstention,
    parse_citation_indices,
    validate_citations,
)


def _doc(text: str, doc_id: str, **metadata) -> RetrievedDocument:
    return RetrievedDocument(text=text, doc_id=doc_id, metadata=metadata)


def _evidence_index(*docs: RetrievedDocument) -> list[dict]:
    return [
        {
            "index": idx,
            "doc_id": doc.doc_id,
            "file_name": doc.metadata.get("file_name"),
            "page_label": doc.metadata.get("page_label"),
            "section": doc.metadata.get("section"),
            "type": doc.metadata.get("type", ""),
        }
        for idx, doc in enumerate(docs, start=1)
    ]


# ---------------------------------------------------------------------------
# Step 1: context numbering and evidence_index
# ---------------------------------------------------------------------------


def test_evidence_pipeline_numbers_chunks_and_builds_index():
    docs = [
        _doc("Fixed-size chunking is efficient.", "a", file_name="chunking.pdf"),
        _doc("It can degrade retrieval.", "b", file_name="chunking.pdf"),
    ]
    output = PrepareEvidencePipeline()(docs)

    evidence_mode, evidence, images = output.content
    assert evidence_mode == 0
    assert "[1]" in evidence
    assert "[2]" in evidence
    assert "Content from chunking.pdf" in evidence

    index = output.metadata["evidence_index"]
    assert [entry["index"] for entry in index] == [1, 2]
    assert index[0]["doc_id"] == "a"
    assert index[1]["doc_id"] == "b"
    assert index[0]["file_name"] == "chunking.pdf"


def test_evidence_pipeline_keeps_page_and_slide_labels():
    docs = [
        _doc("pdf text", "a", file_name="lecture.pdf", page_label="25"),
        _doc("slide text", "b", file_name="lecture.pptx", page_label="4"),
    ]
    _, evidence, _ = PrepareEvidencePipeline()(docs).content

    assert "(Page 25)" in evidence
    assert "(Slide 4)" in evidence


def test_evidence_pipeline_empty_docs():
    output = PrepareEvidencePipeline()([])
    evidence_mode, evidence, images = output.content
    assert evidence_mode == 0
    assert evidence == ""
    assert images == []
    assert output.metadata["evidence_index"] == []


def test_evidence_pipeline_dedup_does_not_misalign_indices():
    docs = [
        _doc("shared sentence", "a", file_name="first.pdf"),
        _doc("shared sentence", "b", file_name="second.pdf"),
        _doc("unique sentence", "c", file_name="third.pdf"),
    ]
    output = PrepareEvidencePipeline()(docs)
    _, evidence, _ = output.content
    index = output.metadata["evidence_index"]

    # The duplicate chunk is skipped, but its reference must not consume an
    # index reserved for a rendered block.
    assert len(index) == 2
    assert [entry["doc_id"] for entry in index] == ["a", "c"]
    assert evidence.count("[1]") == 1
    assert evidence.count("[2]") == 1


def test_evidence_pipeline_marks_truncation():
    docs = [_doc("long context " * 200, "a", file_name="doc.pdf")]
    pipeline = PrepareEvidencePipeline(max_context_length=20)
    _, evidence, _ = pipeline(docs).content

    assert evidence.endswith(TRUNCATION_MARKER)


# ---------------------------------------------------------------------------
# Step 3: citation parsing and validation
# ---------------------------------------------------------------------------


def test_parse_citation_indices_various_forms():
    assert parse_citation_indices("Fact one【1】and fact two【2】.") == [1, 2]
    assert parse_citation_indices("Merged【1,2】citation.") == [1, 2]
    assert parse_citation_indices("Alternate [1] style.") == [1]
    assert parse_citation_indices("Range form【1-3】.") == [1, 2, 3]
    assert parse_citation_indices("No citations here.") == []


def test_validate_citations_flags_out_of_range():
    docs = [_doc("alpha", "a", file_name="a.pdf")]
    index = _evidence_index(*docs)

    result = validate_citations("Answer with citation【99】.", index)

    assert result.invalid_citation_ids == [99]
    assert result.cited_indices == []
    assert result.sources == []
    assert result.citation_valid is False


def test_validate_citations_maps_valid_indices():
    docs = [
        _doc("alpha", "a", file_name="a.pdf", page_label="1"),
        _doc("beta", "b", file_name="b.pptx", page_label="2"),
    ]
    index = _evidence_index(*docs)

    result = validate_citations("Claim one【1】and two【2】.", index)

    assert result.cited_indices == [1, 2]
    assert result.invalid_citation_ids == []
    assert result.citation_valid is True
    assert [source["doc_id"] for source in result.sources] == ["a", "b"]
    assert result.sources[1]["page_label"] == "2"


def test_validate_citations_without_markers_is_invalid():
    docs = [_doc("alpha", "a", file_name="a.pdf")]
    index = _evidence_index(*docs)

    result = validate_citations("An ungrounded answer.", index)

    assert result.cited_indices == []
    assert result.citation_valid is False
    assert result.unsupported_citations == [1]


def test_validate_citations_reports_coverage():
    docs = [_doc("alpha", "a", file_name="a.pdf")]
    index = _evidence_index(*docs)

    result = validate_citations("First claim【1】. Second claim.", index)

    assert result.claim_count == 2
    assert result.cited_claim_count == 1
    assert result.coverage == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Step 2: abstention detection
# ---------------------------------------------------------------------------


def test_is_abstention_positive():
    assert is_abstention(ABSTAIN_MARKER)
    assert is_abstention("I don't know the answer.")
    assert is_abstention("There is insufficient evidence to answer.")


def test_is_abstention_negative():
    assert not is_abstention("Fixed-size chunking is efficient【1】.")
    assert not is_abstention("")
    assert not is_abstention(None)


# ---------------------------------------------------------------------------
# Step 3: highlight mode handles a failed citation pass
# ---------------------------------------------------------------------------


def test_highlight_mode_marks_failed_citation():
    evidence_index = [
        {"index": 1, "doc_id": "a", "file_name": "a.pdf", "page_label": "1"}
    ]
    status = build_citation_metadata(
        output="Some grounded answer【1】.",
        evidence="[1] Content from a.pdf: some grounded answer.",
        citation=None,
        enable_citation=True,
        enable_citation_validation=True,
        evidence_index=evidence_index,
    )

    assert status["citation_status"] == "failed"
    assert status["citation_validation"]["citation_valid"] is True


def test_citation_metadata_ok_when_citation_present():
    status = build_citation_metadata(
        output="Claim【1】.",
        evidence="[1] Content from a.pdf: claim.",
        citation=object(),
        enable_citation=True,
        enable_citation_validation=True,
        evidence_index=[{"index": 1, "doc_id": "a", "file_name": "a.pdf"}],
    )

    assert status["citation_status"] == "ok"
    assert status["citation_validation"]["citation_valid"] is True


def test_highlight_mode_empty_evidence_uses_abstention_prompt():
    prompt = build_abstention_prompt("What is chunking?")

    assert ABSTAIN_MARKER in prompt
    assert "What is chunking?" in prompt
    assert is_abstention(ABSTAIN_MARKER)


# ---------------------------------------------------------------------------
# Step 3: inline mode drops out-of-range links
# ---------------------------------------------------------------------------


def test_inline_replace_citation_drops_invalid_ids():
    pipeline = AnswerWithInlineCitation()
    rendered = pipeline.replace_citation_with_link("Claim【1】and bogus【99】.", max_id=2)

    assert "id='mark-1'" in rendered
    assert "id='mark-99'" not in rendered
    assert "【99】" in rendered


def test_inline_replace_citation_renders_valid_links():
    pipeline = AnswerWithInlineCitation()
    rendered = pipeline.replace_citation_with_link("Claim【1】and【2】.", max_id=2)

    assert "id='mark-1'" in rendered
    assert "id='mark-2'" in rendered


# ---------------------------------------------------------------------------
# Utility edge cases
# ---------------------------------------------------------------------------


def test_find_text_requires_min_length():
    assert find_text("x", "some longer context text") == []
    assert find_text("", "some longer context text") == []


def test_find_text_matches_substring():
    matches = find_text("fixed size chunking", "a fixed size chunking method")
    assert matches
    start, end = matches[0]
    assert "fixed size chunking" in "a fixed size chunking method"[start:end]


def test_find_start_end_phrase_returns_none_for_absent_text():
    match, _ = find_start_end_phrase("not here", "neither here", "short")
    assert match is None


def test_find_start_end_phrase_matches_span():
    context = "the quick brown fox jumps over the lazy dog"
    match, length = find_start_end_phrase("quick brown", "lazy dog", context)
    assert match is not None
    start, end = match
    assert "quick brown" in context[start:end]
    assert length > 0


def test_cite_evidence_model_holds_quotes():
    evidence = CiteEvidence(evidences=["direct quote one", "direct quote two"])
    assert evidence.evidences == ["direct quote one", "direct quote two"]
