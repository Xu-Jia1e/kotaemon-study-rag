import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# Canonical sentence the model is instructed to emit when the evidence is
# missing or insufficient. Detection is case-insensitive and tolerant to the
# common phrasings models produce.
ABSTAIN_MARKER = "Insufficient evidence to answer from the provided materials."

_ABSTAIN_PATTERNS = (
    "insufficient evidence",
    "insufficient information",
    "not enough evidence",
    "not enough information",
    "don't know",
    "do not know",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "no relevant information",
    "not provided in the context",
)

# Matches 【n】, [n] and comma/range forms such as 【1,2】, [1, 2], 【1-3】.
_CITATION_GROUP_PATTERN = re.compile(r"[【\[]([\d,\s\-–]+)[】\]]")


def is_abstention(text: str | None) -> bool:
    """Return True when the answer text is an abstention/refusal."""
    if not text:
        return False
    lowered = text.lower()
    return any(pattern in lowered for pattern in _ABSTAIN_PATTERNS)


@dataclass
class CitationValidation:
    """Result of validating citation markers against the evidence index."""

    citation_valid: bool = True
    cited_indices: list[int] = field(default_factory=list)
    invalid_citation_ids: list[int] = field(default_factory=list)
    unsupported_citations: list[int] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    claim_count: int = 0
    cited_claim_count: int = 0
    coverage: float = 0.0

    def to_dict(self) -> dict:
        return {
            "citation_valid": self.citation_valid,
            "cited_indices": self.cited_indices,
            "invalid_citation_ids": self.invalid_citation_ids,
            "unsupported_citations": self.unsupported_citations,
            "sources": self.sources,
            "claim_count": self.claim_count,
            "cited_claim_count": self.cited_claim_count,
            "coverage": self.coverage,
        }


def _expand_indices(group: str) -> list[int]:
    """Expand a citation group like ``1,2`` or ``1-3`` into integer indices."""
    indices: list[int] = []
    for part in re.split(r"[,\s]+", group.strip()):
        if not part:
            continue
        range_match = re.match(r"^(\d+)[\-–](\d+)$", part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if start <= end:
                indices.extend(range(start, end + 1))
            else:
                indices.extend(range(end, start + 1))
            continue
        if part.isdigit():
            indices.append(int(part))
    return indices


def parse_citation_indices(text: str | None) -> list[int]:
    """Return all citation indices referenced in the answer, in order."""
    if not text:
        return []
    indices: list[int] = []
    for match in _CITATION_GROUP_PATTERN.finditer(text):
        indices.extend(_expand_indices(match.group(1)))
    return indices


def _count_claims(text: str) -> tuple[int, int]:
    """Estimate (total claims, cited claims) from answer sentences."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", text)
        if sentence.strip()
    ]
    if not sentences:
        return 0, 0
    cited = sum(1 for sentence in sentences if parse_citation_indices(sentence))
    return len(sentences), cited


def validate_citations(
    text: str | None, evidence_index: list[dict]
) -> CitationValidation:
    """Validate cited indices in ``text`` against ``evidence_index``.

    Out-of-range indices are dropped and recorded in ``invalid_citation_ids``.
    Valid indices are mapped to their source metadata. When ``text`` contains no
    citation markers at all, the answer is treated as unsupported so callers can
    surface a warning.
    """
    result = CitationValidation()
    available: dict[int, dict] = {}
    for entry in evidence_index or []:
        index = entry.get("index")
        if isinstance(index, int):
            available[index] = entry

    cited = parse_citation_indices(text)
    seen: set[int] = set()
    for idx in cited:
        if idx in seen:
            continue
        seen.add(idx)
        if idx in available:
            result.cited_indices.append(idx)
            result.sources.append(available[idx])
        else:
            result.invalid_citation_ids.append(idx)

    result.claim_count, result.cited_claim_count = _count_claims(text or "")
    if result.claim_count:
        result.coverage = result.cited_claim_count / result.claim_count

    if not cited:
        # No citation markers: the answer is not grounded in any source.
        result.citation_valid = False
        result.unsupported_citations = list(available.keys())
    elif result.invalid_citation_ids:
        result.citation_valid = False
        result.unsupported_citations = list(result.invalid_citation_ids)
    elif is_abstention(text):
        # An abstention with no (or only dropped) citations is valid.
        result.citation_valid = not result.invalid_citation_ids

    return result


def find_text(search_span, context, min_length=5):
    search_span, context = search_span.lower(), context.lower()

    sentence_list = search_span.split("\n")
    context = context.replace("\n", " ")

    matches_span = []
    # don't search for small text
    if len(search_span) > min_length:
        for sentence in sentence_list:
            match_results = SequenceMatcher(
                None,
                sentence,
                context,
                autojunk=False,
            ).get_matching_blocks()

            matched_blocks = []
            for _, start, length in match_results:
                if length > max(len(sentence) * 0.25, min_length):
                    matched_blocks.append((start, start + length))

            if matched_blocks:
                start_index = min(start for start, _ in matched_blocks)
                end_index = max(end for _, end in matched_blocks)
                length = end_index - start_index

                if length > max(len(sentence) * 0.35, min_length):
                    matches_span.append((start_index, end_index))

    if matches_span:
        # merge all matches into one span
        final_span = min(start for start, _ in matches_span), max(
            end for _, end in matches_span
        )
        matches_span = [final_span]

    return matches_span


def find_start_end_phrase(
    start_phrase, end_phrase, context, min_length=5, max_excerpt_length=300
):
    start_phrase, end_phrase = start_phrase.lower(), end_phrase.lower()
    context = context.lower()

    context = context.replace("\n", " ")

    matches = []
    matched_length = 0
    for sentence in [start_phrase, end_phrase]:
        if sentence is None:
            continue

        match = SequenceMatcher(
            None, sentence, context, autojunk=False
        ).find_longest_match()
        if match.size > max(len(sentence) * 0.35, min_length):
            matches.append((match.b, match.b + match.size))
            matched_length += match.size

    # check if second match is before the first match
    if len(matches) == 2 and matches[1][0] < matches[0][0]:
        # if so, keep only the first match
        matches = [matches[0]]

    if matches:
        start_idx = min(start for start, _ in matches)
        end_idx = max(end for _, end in matches)

        # check if the excerpt is too long
        if end_idx - start_idx > max_excerpt_length:
            end_idx = start_idx + max_excerpt_length

        final_match = (start_idx, end_idx)
    else:
        final_match = None

    return final_match, matched_length


def replace_think_tag_with_details(text):
    text = text.replace(
        "<think>",
        '<details><summary><span style="color:grey">Thought</span></summary><blockquote>',  # noqa
    )
    text = text.replace("</think>", "</blockquote></details>")
    return text


def strip_think_tag(text):
    if "</think>" in text:
        text = text.split("</think>")[1]
    return text
