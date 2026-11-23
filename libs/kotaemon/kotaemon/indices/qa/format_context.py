import html
from functools import partial

import tiktoken

from kotaemon.base import BaseComponent, Document, RetrievedDocument
from kotaemon.indices.splitters import TokenSplitter

EVIDENCE_MODE_TEXT = 0
EVIDENCE_MODE_TABLE = 1
EVIDENCE_MODE_CHATBOT = 2
EVIDENCE_MODE_FIGURE = 3

TRUNCATION_MARKER = "... [context truncated]"


def _evidence_entry(retrieved_item: RetrievedDocument, idx: int) -> dict:
    """Build the structured reference for a numbered evidence chunk."""
    metadata = retrieved_item.metadata
    return {
        "index": idx,
        "doc_id": retrieved_item.doc_id,
        "file_name": metadata.get("file_name", "-"),
        "page_label": metadata.get("page_label"),
        "section": metadata.get("section"),
        "type": metadata.get("type", ""),
    }


class PrepareEvidencePipeline(BaseComponent):
    """Prepare the evidence text from the list of retrieved documents

    This step usually happens after `DocumentRetrievalPipeline`.

    Args:
        trim_func: a callback function or a BaseComponent, that splits a large
            chunk of text into smaller ones. The first one will be retained.

    Attributes:
        number_chunks: when True, prefix each evidence block with a 1-based
            `[n]` marker and expose the matching `evidence_index` on the returned
            document metadata.
    """

    max_context_length: int = 32000
    trim_func: TokenSplitter | None = None
    number_chunks: bool = True

    def run(self, docs: list[RetrievedDocument]) -> Document:
        evidence = ""
        images = []
        table_found = 0
        evidence_modes = []
        evidence_index: list[dict] = []

        evidence_trim_func = (
            self.trim_func
            if self.trim_func
            else TokenSplitter(
                chunk_size=self.max_context_length,
                chunk_overlap=0,
                separator=" ",
                tokenizer=partial(
                    tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
                    allowed_special=set(),
                    disallowed_special="all",
                ),
            )
        )

        def marker(retrieved_item: RetrievedDocument) -> str:
            """Return the `[n] ` prefix and register the chunk when numbered."""
            if not self.number_chunks:
                return ""
            idx = len(evidence_index) + 1
            evidence_index.append(_evidence_entry(retrieved_item, idx))
            return f"[{idx}] "

        for _, retrieved_item in enumerate(docs):
            retrieved_content = ""
            page = retrieved_item.metadata.get("page_label", None)
            source = filename = retrieved_item.metadata.get("file_name", "-")
            if page:
                label = "Slide" if filename.lower().endswith(".pptx") else "Page"
                source += f" ({label} {page})"
            if section := retrieved_item.metadata.get("section"):
                source += f" · {html.escape(str(section))}"
            if retrieved_item.metadata.get("type", "") == "table":
                evidence_modes.append(EVIDENCE_MODE_TABLE)
                if table_found < 5:
                    retrieved_content = retrieved_item.metadata.get(
                        "table_origin", retrieved_item.text
                    )
                    if retrieved_content not in evidence:
                        table_found += 1
                        evidence += (
                            f"<br><b>{marker(retrieved_item)}"
                            f"Table from {source}</b>\n" + retrieved_content + "\n<br>"
                        )
            elif retrieved_item.metadata.get("type", "") == "chatbot":
                evidence_modes.append(EVIDENCE_MODE_CHATBOT)
                retrieved_content = retrieved_item.metadata["window"]
                evidence += (
                    f"<br><b>{marker(retrieved_item)}"
                    f"Chatbot scenario from {filename} (Row {page})</b>\n"
                    + retrieved_content
                    + "\n<br>"
                )
            elif retrieved_item.metadata.get("type", "") == "image":
                evidence_modes.append(EVIDENCE_MODE_FIGURE)
                retrieved_content = retrieved_item.metadata.get("image_origin", "")
                retrieved_caption = html.escape(retrieved_item.get_content())
                evidence += (
                    f"<br><b>{marker(retrieved_item)}"
                    f"Figure from {source}</b>\n"
                    + "<img width='85%' src='<src>' "
                    + f"alt='{retrieved_caption}'/>"
                    + "\n<br>"
                )
                images.append(retrieved_content)
            else:
                if "window" in retrieved_item.metadata:
                    retrieved_content = retrieved_item.metadata["window"]
                else:
                    retrieved_content = retrieved_item.text
                retrieved_content = retrieved_content.replace("\n", " ")
                if retrieved_content not in evidence:
                    evidence += (
                        f"<br><b>{marker(retrieved_item)}"
                        f"Content from {source}: </b> " + retrieved_content + " \n<br>"
                    )

        # resolve evidence mode
        evidence_mode = EVIDENCE_MODE_TEXT
        if EVIDENCE_MODE_FIGURE in evidence_modes:
            evidence_mode = EVIDENCE_MODE_FIGURE
        elif EVIDENCE_MODE_TABLE in evidence_modes:
            evidence_mode = EVIDENCE_MODE_TABLE

        # trim context by trim_len
        print("len (original)", len(evidence))
        truncated = False
        if evidence:
            texts = evidence_trim_func([Document(text=evidence)])
            trimmed = texts[0].text
            truncated = len(trimmed) < len(evidence)
            evidence = trimmed
            print("len (trimmed)", len(evidence))
            if truncated:
                evidence += " " + TRUNCATION_MARKER

        return Document(
            content=(evidence_mode, evidence, images),
            metadata={
                "evidence_index": evidence_index,
                "evidence_truncated": truncated,
            },
        )
