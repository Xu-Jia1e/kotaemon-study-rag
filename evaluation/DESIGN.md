# Evaluation Design and Experimental Analysis

This document describes the technical design of the `evaluation/` module and
analyses the real experiments completed for the study-material RAG project. It
complements `README.md`: the README explains how to use the module, while this
document explains why it is built this way and what the results mean.

The evaluation module is a consumer of the application libraries. It does not
modify `libs/kotaemon/`, `libs/ktem/`, or `app.py`.

## 1. System Overview

The project separates four concerns:

| Concern | Implementation | Evaluation boundary |
|---|---|---|
| A. Parsing, chunking, index build | `IndexDocumentPipeline` in `libs/ktem` | `evaluation/adapters/a_index.py` builds and loads indexes |
| B. Retrieval | `VectorRetrieval` in `libs/kotaemon` | `evaluation/adapters/b_a_index.py` connects it to an A index |
| C. Grounded answer, citation, abstention | `PrepareEvidencePipeline`, `AnswerWithContextPipeline`, `CitationPipeline` | `evaluation/adapters/c_answer.py` |
| D. Evaluation | `evaluation/` | Datasets, metrics, batch runners, reports |

The two real chunking configurations are evaluated on the same 30-question
dataset and the same four source documents:

- `1024/256`: 1024-token chunks with 256-token overlap.
- `512/128`: 512-token chunks with 128-token overlap.

Only the 1024/256 configuration has an end-to-end answer and human-grading run.
The 512/128 configuration has a real retrieval run but no answer generation and
no human grading.

## 2. Evaluation Architecture

```
                    evaluation/datasets/qa_cases_v2.jsonl
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │ validate_dataset / verify_evidence │
                  │ (schema + verbatim quote checks)  │
                  └─────────────────────────────────┘

  .eval-index/<profile>/                run_batch_retrieval.py
  ├── index.json          ──────────►  ├── per_case.jsonl (append + fsync)
  ├── vectors (Chroma)                 ├── retrieval.json
  ├── documents (LanceDB)              ├── failures.json
  └── pipeline.sqlite3                 ├── summary.json
                                       └── run_manifest.json
                                                │
                                                ▼
  retrieval.json ──────────────► run_batch_e2e.py ──────────► answers.json
                                  (C answer + citation)       predictions.jsonl
                                                              report_auto.json
                                                              run_manifest.json
                                                                      │
                                                                      ▼
  answers.json + human labels ──► run_answer_eval.py ─────────► report_scored.json
  retrieval.json ───────────────► report_content_metrics.py ──► content_metrics.json
```

### Components

| Component | Responsibility |
|---|---|
| `adapters/a_index.py` | Build and load A indexes; normalise D and A manifest schemas; select the correct Chroma collection and LanceDB table; reject empty collections; verify counts |
| `adapters/b_a_index.py` | Wrap B's `VectorRetrieval` over the loaded A stores and return evaluator items |
| `adapters/c_answer.py` | Prepare evidence, generate an answer, extract citations, validate citation markers, count real LLM calls |
| `batch.py` | Shared resilient execution: per-case isolation, append-and-fsync, resume, retries, manifests, rebuilds |
| `scripts/run_batch_retrieval.py` | 30-question retrieval runner with resume and rebuild |
| `scripts/run_batch_e2e.py` | 30-question answer runner with resume, rebuild, and LLM call accounting |
| `metrics/retrieval.py` | Strict Hit@K, MRR@K, grouped evidence Recall@K |
| `metrics/content.py` | Supplementary content-level Hit@K and Recall@K from gold quotes |
| `metrics/answer.py` | Answer correctness aggregation and refusal metrics |
| `metrics/citation.py` | Citation validity and citation-support aggregation |
| `reporting.py` | Per-case records, aggregate metrics, Markdown rendering |
| `scripts/make_grading.py` | Human grading sheets with cited chunk text |
| `scripts/report_content_metrics.py` | Strict-vs-content comparison and four-case evidence appendix |

### Position of the experiment matrix

`evaluation/matrix.py` defines a planning grid of 2 chunking configurations ×
3 retrieval modes × 3 Top-K values = 18 cells. The grid is a planning scaffold,
not the executed experiment set. The results reported here are the two
vector/k=5 cells executed through the dedicated batch runners. Hybrid retrieval,
RRF, other Top-K values, and reranking were not run.

## 3. Dataset Design

### Composition

`evaluation/datasets/qa_cases_v2.jsonl` is the only official evaluation set:

| Type | Count | Design intent |
|---|---:|---|
| `direct` | 12 | One evidence unit, answer stated explicitly |
| `synthesis` | 8 | At least two evidence groups; cross-page or cross-section integration |
| `detail` | 5 | Exact line, symbol, or table cell |
| `unanswerable` | 5 | Knowledge base genuinely lacks the answer; must be refused |

Totals: **30 questions, 25 answerable, 5 unanswerable, 43 evidence units/groups**.
The dataset SHA-256 is recorded in the run manifests; the current value is
`9c2c72ad79004f4b…`.

### Evidence anchoring

Every gold unit is anchored on:

```json
{"file": "11-hash2.pdf", "locator": {"kind": "page", "value": 23},
 "group": "clustering", "quote": "..."}
```

`chunk_id` is intentionally not the anchor. The two chunking configurations
produce different boundaries and ids for the same passage, so a chunk-id anchor
would make the comparison invalid. `locator.kind` may be `page`, `slide`,
`section`, or `table_row`.

### Evidence groups

- Units in the same `group` are interchangeable: "page 26 **or** page 27" is one
  requirement.
- Different groups must all be covered: a two-part question needs both groups.
- Group coverage is what Strict Recall@K measures.

### Unanswerable questions

Each unanswerable case carries a `checked_scope` with machine-checkable
constraints:

- `absent_phrases`: phrases that would make the question answerable must be
  absent from the corpus;
- `max_mentions`: a term may appear only a bounded number of times (for example,
  "cuckoo" appears once as a topic name only);
- `blank_fields`: a template field exists but contains no digits/value.

`verify_evidence.py` evaluates these constraints; a violation fails the check.

### Source documents

| File | SHA-256 (prefix) | Role |
|---|---|---|
| `11-hash2.pdf` | `7aa2d367…` | English hash-table lecture, 37 pages |
| `11-hash2.pptx` | `8d92e93c…` | Same lecture, PPTX, 37 slides |
| `USTC-ClassIn.pdf` | `ced8a509…` | Chinese ClassIn teaching guide, 16 pages |
| `USST-syllabus.docx` | `09762a9c…` | Syllabus template with the bridge example row |

The two UTS accessibility documents return HTTP 403 from the source host and are
not used. `qa_cases.jsonl` (6 legacy search cases) is retained only for
historical reference and legacy smoke fixtures; it is not used in any reported
metric.

## 4. Metric Definitions

### 4.1 Strict retrieval metrics

For the answerable set `A` (|A| = 25), with `Gold(q)` the set of gold units and
`TopK(q)` the first K retrieved items:

```
Strict Hit@K      = |{q ∈ A : TopK(q) contains ≥1 gold unit}| / |A|
MRR@K             = (1/|A|) Σ_q 1/rank_q(first matching unit)   (0 if none)
Strict Recall@K   = (1/|A|) Σ_q covered_groups(q) / total_groups(q)
```

Matching rules:

- file names are compared case-insensitively and by basename;
- page and slide values are normalised to integers (`"p. 23"` → `23`);
- section/table-row values are whitespace-normalised and case-folded;
- `page`/`page_label`, `slide`/`slide_label`, and `section`/`table_row`/`heading`
  are grouped as equivalent locator families;
- a chunk-id fast path is used only when both gold and prediction carry one;
- unanswerable questions are excluded by definition; only answerable questions
  without gold evidence are counted in `cases_skipped` and listed in
  `skipped_ids`, not silently counted as zero.

### 4.2 Content-level metrics (supplementary)

Strict matching has two known blind spots:

1. the same lecture exists as PDF and PPTX, so an exact-content hit in the other
   carrier is a strict miss;
2. DOCX chunks carry no page or section metadata, so a `table_row` gold locator
   cannot match positionally.

The content metric checks the verified gold quote against the retrieved chunk
text after whitespace and case normalisation:

```
Content Hit@K    = |{q : ≥1 gold quote appears in TopK(q) text}| / |A|
Content Recall@K = (1/|A|) Σ_q matched_quotes(q) / total_quotes(q)
```

This is a diagnostic view only. It does not replace Strict Hit@K, MRR@K, or
Strict Recall@K. In particular, Content Hit@5 = 1.0 means only that at least one
quote was found per question; it does not mean group coverage or ranking is
perfect.

### 4.3 Answer correctness

Human three-way labels:

```
correct    = 1.0
partial    = 0.5
incorrect  = 0.0
not_judged = excluded
```

```
Answer Accuracy  = Σ score / judged_count
Strict Accuracy  = count(correct) / judged_count
```

Only the 25 answerable questions are judged. The five unanswerable questions are
evaluated by refusal behaviour, not by answer accuracy. Unlabelled answers are
`null` in automatic reports and are never converted to zero.

### 4.4 Citation metrics

Two distinct questions are measured separately:

| Metric | Question | Method |
|---|---|---|
| Citation Validity | Does every `【n】` point to an evidence block retrieved for this question? | automatic, reuses C's `validate_citations` |
| Citation Support | Does the cited passage support the sentence? | human label per citation |

```
Citation Validity = valid non-abstained answers / scored non-abstained answers
Citation Support  = supported citation labels / total citation labels
```

Abstentions are excluded from both denominators: an abstention has no citation
by design, and counting it as "invalid" would punish correct refusal. Empty
evidence is also excluded as undefined, not zero.

### 4.5 Refusal metrics

```
Refusal Accuracy  = refused unanswerable / 5
False Refusal     = refused answerable / 25
Balanced Accuracy = (Refusal Accuracy + 1 - False Refusal) / 2
```

Refusal is detected with the project's `is_abstention` implementation, so the
metric matches shipped behaviour. The detector is text-based; the automatic
values are a first pass, and the five unanswerable cases were also reviewed in
the grading sheet.

### 4.6 Reporting rules

- Metrics without labels are `null`, never zero.
- `report_auto.json` records automatic metrics only; human fields stay `null`.
- `report_scored.json` records human-labelled metrics and includes the SHA-256
  of `labels.jsonl`, the SHA-256 of `citation_labels.jsonl`, and
  `grading_version`.
- Per-case records distinguish `real`, `simulated`, `failed`, `missing`,
  `unscored`, and `not applicable` states.

## 5. Experimental Configuration

### 5.1 Common controls

| Control | Value |
|---|---|
| Dataset | `evaluation/datasets/qa_cases_v2.jsonl`, SHA-256 `9c2c72ad…` |
| Questions | 30 (25 answerable, 5 unanswerable) |
| Source documents | The same four files in both configurations |
| Embedding | `text-embedding-v4`, 2048 dimensions, DashScope-compatible endpoint |
| Retrieval | Vector only, Top-K = 5 |
| Reranker | Disabled (`rerankers=[]`) |
| Chunking A | 1024 tokens / 256 overlap |
| Chunking B | 512 tokens / 128 overlap |
| LLM (answers) | DeepSeek `deepseek-flash`, 1024/256 only |

The two indexes are built in separate processes because `theflow` caches its
settings module. The 1024/256 index was built with
`evaluation.adapters.a_index`; the 512/128 index was built with A's
`scripts/part_a_index.py` and then loaded through the D adapter. Both are real
indexes, not simulated fixtures.

### 5.2 Index statistics

| Metric | 1024/256 | 512/128 |
|---|---:|---:|
| Text chunks | 106 | 111 |
| Text vectors | 106 | 111 |
| Docstore rows (including thumbnails) | 159 | 164 |
| Page thumbnails (not embedded) | 53 | 53 |
| Text chunks with a vector | 106/106 | 111/111 |
| Source records | 4 | 4 |
| Manifest schema | D (nested `embedding`) | A (flat fields + `collection`) |
| Chroma collection | `default` | `index_1` |
| LanceDB table | `docstore` | `index_1` |

Per-document text vectors:

| File | 1024/256 | 512/128 |
|---|---:|---:|
| `11-hash2.pdf` | 37 | 37 |
| `11-hash2.pptx` | 37 | 37 |
| `USTC-ClassIn.pdf` | 17 | 22 |
| `USST-syllabus.docx` | 15 | 15 |

The 512/128 delta is concentrated in `USTC-ClassIn.pdf`, whose page-level content
splits into more text chunks.

### 5.3 Run status

| Run | Status |
|---|---|
| 1024/256 retrieval, 30 questions | 30/30 success, 0 failed, 0 missing |
| 512/128 retrieval, 30 questions | 30/30 success, 0 failed, 0 missing |
| 1024/256 answers, 30 questions | 30/30 success; 60 real LLM calls (30 answers + 30 citation extractions) |
| 512/128 answers | **not run** |
| Human grading | 25 answerable answers and 51 citations; 5 refusal cases reviewed |

## 6. Experimental Results

### 6.1 Retrieval

| Metric | 1024/256 | 512/128 | Δ |
|---|---:|---:|---:|
| Strict Hit@5 | 0.8400 | 0.8400 | 0.0000 |
| MRR@5 | 0.5833 | 0.5633 | −0.0200 |
| Strict Recall@5 | 0.7560 | 0.7560 | 0.0000 |
| Content Hit@5 | 1.0000 | 1.0000 | 0.0000 |
| Content Recall@5 | 0.9107 | 0.8973 | −0.0133 |

By question type:

| Type | n | 1024 Hit | 512 Hit | 1024 MRR | 512 MRR | 1024 Recall | 512 Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 12 | 0.8333 | 0.8333 | 0.6458 | 0.6458 | 0.8333 | 0.8333 |
| synthesis | 8 | 0.8750 | 0.8750 | 0.6250 | 0.5625 | 0.6125 | 0.6125 |
| detail | 5 | 0.8000 | 0.8000 | 0.3667 | 0.3667 | 0.8000 | 0.8000 |
| unanswerable | 5 | — | — | — | — | — | — |

All of the aggregate difference is in synthesis MRR; Hit and Recall are
unchanged.

### 6.2 Analysis of `v2-s06`

`v2-s06` asks for the three stages of USTC's online-teaching flow, the steps to
create and enter a ClassIn classroom, and the screen-sharing entry point.

Gold evidence:

- page 2: "登录瀚海教学网";
- page 4: "点击新建 ClassIn 课堂";
- page 10: "选择共享工具".

With 1024/256, the page-4 chunk contains the full set of steps and ranks first.
With 512/128, page 4 is split into several chunks; the chunk containing the gold
sentence falls outside the Top-5. Another page-4 chunk ranks second, so strict
matching by page still finds page 4 at rank 2, but the first match is no longer
rank 1:

- reciprocal rank: 1.0 → 0.5;
- content recall: 3/3 → 2/3.

This explains the complete MRR drop and the content-recall drop. The other 24
scored questions keep their first-match ranks. It is a chunk-boundary effect, not
a general degradation of the 512/128 retriever.

### 6.3 Answer and citation results (1024/256 only)

| Metric | Value |
|---|---:|
| Answer Accuracy | **0.9400** (22 correct, 3 partial, 0 incorrect; 25 judged) |
| Strict Accuracy | **0.8800** |
| Citation Validity | **1.0000** (0 invalid ids) |
| Citation Support | **0.9608** (49/51 citations) |
| Refusal Accuracy | **1.0000** (5/5) |
| False Refusal Rate | **0.0000** |
| Balanced Accuracy | **1.0000** |

The three partial answers:

- `v2-s01`: linear probing's `+1` probing and primary clustering were omitted;
- `v2-s04`: the role of the prime table size was not explained, and the answer
  incorrectly claimed the material did not explain it;
- `v2-s07`: independent chat/question areas and like/pin behaviour were omitted.

Citation support: 49 of 51 citations were supported. In `v2-s04`, citations
【2】【3】 support "use a prime table size" but not the claim "the material does not
explain why". This is why Citation Validity = 1.0 but Citation Support = 0.9608:
valid marker indices do not guarantee supporting content.

All five unanswerable questions returned the fixed abstention sentence even
though each had five retrieved evidence entries. This is a model-led refusal,
not an empty-evidence short circuit.

### 6.4 Interpretation

Within this dataset:

- 512/128 does not improve Strict Hit@5 or Strict Recall@5;
- 512/128 has a small MRR regression, fully attributable to `v2-s06`;
- content coverage is slightly lower for the same reason;
- 1024/256 has verified answer quality; 512/128 does not, because no answer run
  was performed.

The results do not justify switching the default chunking configuration to
512/128. They also do not show that 512/128 is materially worse overall: apart
from one synthesis question, ranking behaviour is identical.

## 7. Implementation Details

### 7.1 Index adapter compatibility

The adapter supports two manifest schemas:

- **D schema**: nested `embedding.model/dimensions`, `files` as a list,
  `vector_count`, `indexed_documents`.
- **A schema**: flat `embedding_model`/`dimensions`, `files` as a mapping of
  `source_id`/`error`, `collection`, `complete`.

`normalize_manifest()` produces one canonical view for both. The loader:

- reads the Chroma collection name from the manifest and opens that exact
  collection (falling back to `default`/`docstore` only for the D schema);
- checks collection existence with a read-only SQLite query **before** opening
  Chroma, so a missing collection cannot be silently created;
- rejects empty Chroma or LanceDB stores;
- verifies the vector count against the manifest when one is recorded;
- reports document-count differences (the 1024 manifest records 158 while the
  store has 159 rows including thumbnails) as a warning rather than hiding it;
- loads read-only: it does not create ORM tables in an existing index.

### 7.2 Resilient batch execution

`batch.py` and the two batch runners implement:

- **Per-question isolation**: one exception produces one `failed` record; the
  batch continues.
- **Append and fsync**: every finished question is appended to
  `per_case.jsonl` and flushed/fsynced, so a crash keeps completed work.
- **Resume**: `--resume` skips only successful, complete records whose
  configuration fingerprint matches the current dataset, index, model, Top-K,
  chunking, and adapter version. Failed records are retried.
- **Bounded retries**: transient 429/5xx/timeout errors are retried a limited
  number of times with exponential backoff; other errors fail immediately.
- **Run manifest**: git commit, code hashes, dataset/index/retrieval hashes,
  model and chunking configuration, Top-K, timestamps, and counts. No API keys
  are written.
- **Rebuild**: `--rebuild-only` reconstructs outputs from `per_case.jsonl`
  without calling a model.
- **Interrupted state**: an interrupted run writes `status: interrupted` and
  keeps the partial outputs built from completed records.

### 7.3 Result integrity

- A question cannot be silently counted as both success and failure; rebuilds
  use the latest record per case.
- `failures.json` separates failed questions from questions never run.
- Unscored metrics stay `null`; `report_auto.json` records
  `scored: 0, unscored: 30` for the answer run before human labels exist.
- `text_chunks == text_vectors` is checked per index; the 53 PDF thumbnails are
  deliberately excluded from the vector count.
- The 1024/256 and 512/128 runs write to separate output directories; the
  4-question pilot artifacts are not overwritten.
- Human labels and the scored report are separate files; the scored report
  records the label file hashes and `grading_version`.

### 7.4 Tests

`evaluation/tests/` contains 95 tests covering:

- dataset schema and 30-case distribution;
- verbatim evidence verification;
- strict and content retrieval metrics;
- answer, citation, and refusal metrics;
- manifest compatibility, collection selection, and empty-collection rejection;
- batch isolation, retries, resume, interrupted runs, and rebuilds;
- grading-sheet generation and overwrite protection;
- report rendering and null handling.

The tests are offline: they use fixtures or local files and do not call
embedding or chat APIs. `pytest evaluation -q` passes 95 tests.

## 8. Limitations and Future Improvements

### Limitations

- **No reranker experiments.** Retrieval uses `rerankers=[]`; no reranking
  result is claimed.
- **Only vector retrieval at K = 5.** Hybrid retrieval, RRF, and K = 3/10 were
  not run.
- **512/128 is retrieval-only.** No answers, no citation labels, no refusal
  review, and therefore no answer-quality claim for 512/128.
- **Small, single-domain dataset.** 30 questions and four documents; the result
  is descriptive, not a general benchmark.
- **Single grader.** Human labels were produced by one evaluator; no
  inter-annotator agreement or significance testing.
- **Strict matching blind spots.** PDF/PPTX duplicates and DOCX chunks without
  page/section metadata can create strict misses; content metrics mitigate but
  do not replace strict metrics.
- **Local-only artifacts.** Indexes and experiment outputs are gitignored; they
  must be rebuilt or distributed separately.
- **Different index builders.** 1024/256 was built by the D adapter; 512/128 was
  built with A's script. Both are real and verified, but A's six-document counts
  are not directly comparable with this four-document experiment.

### Future improvements

1. Enable and evaluate a reranker as a separate factor, with candidate depth
   fixed.
2. Run hybrid/RRF retrieval and K = 3/10 to test whether the chunking conclusion
   holds beyond vector/K=5.
3. Generate and grade 512/128 answers with the same rubric, so the two
   configurations can be compared end to end.
4. Add a second grader and report agreement.
5. Extend the dataset and document set beyond the current course/tool domain.
6. Improve DOCX chunk locators (or add a content-anchored matching mode) to
   reduce known strict false misses.
7. Publish index artifacts or a reproducible build recipe alongside the code if
   external reproduction is required.
