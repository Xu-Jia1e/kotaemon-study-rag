# Evaluation Module

Offline-first evaluation harness for the study-material RAG project. It measures
retrieval quality, grounded answer quality, citation behaviour, and refusal
behaviour, and it compares the two chunking strategies used by the project. The
module lives entirely under `evaluation/`; it calls the application libraries as
a consumer and does not modify `libs/kotaemon/`, `libs/ktem/`, or `app.py`.

## Project Overview

The RAG pipeline has four separable layers:

| Layer | Owner | Artifact consumed by this module |
|---|---|---|
| Parsing / chunking / index build | A | Chroma vectors, LanceDB chunks, SQLite source records, `index.json` |
| Retrieval | B | `VectorRetrieval` over the A index, exposed by `adapters/b_a_index.py` |
| Grounded answer / citation / abstention | C | `PrepareEvidencePipeline`, `AnswerWithContextPipeline`, citation validation |
| Evaluation | D | This module: datasets, adapters, metrics, runners, reports |

The evaluation module has two modes:

- **Offline**: dataset validation, evidence verification, metric aggregation,
  reporting and unit tests never call a model API.
- **Real experiments**: the batch runners call the real embedding and chat APIs
  when explicitly invoked. They require a local `.env` and are never run by the
  test suite.

## Key Features

- **Dataset validation** — JSONL schema, question-type distribution, evidence
  rules, and machine-checkable "unanswerable" constraints.
- **Evidence verification** — every gold quote is checked verbatim against the
  cited page, slide, section, or DOCX table row; missing evidence is an error,
  not a warning.
- **Index adapters** — build and load A's indexes, with compatibility for both
  the D manifest schema (1024/256) and A's manifest schema (512/128).
- **Retrieval evaluation** — strict Hit@K, MRR@K, and grouped evidence
  Recall@K, plus a supplementary content-level view.
- **Answer evaluation** — accuracy from human labels only; unlabelled answers
  stay `null` and are excluded from denominators.
- **Citation evaluation** — marker validity (automatic) and citation support
  (human labels) are kept separate.
- **Refusal evaluation** — refusal accuracy on unanswerable questions and false
  refusal rate on answerable questions.
- **Resilient batch execution** — per-question isolation, append-and-fsync
  records, `--resume`, bounded retries, run manifests, and result rebuilds.
- **Reports** — per-case JSONL, aggregate JSON, human grading sheets, and
  Markdown summaries.

## Dataset

The only official evaluation set is:

`evaluation/datasets/qa_cases_v2.jsonl`

It contains **30 questions**:

| Type | Count | Purpose |
|---|---:|---|
| `direct` | 12 | Single evidence unit; answer stated explicitly |
| `synthesis` | 8 | Multiple evidence groups; cross-page/section integration |
| `detail` | 5 | Precise location of a line or symbol |
| `unanswerable` | 5 | Absent from the corpus; must be refused |

Of the 30 questions, **25 are answerable** and **5 are unanswerable**. The set
contains **43 verified evidence units/groups**.

Evidence is anchored on **file + locator** (`page`, `slide`, `section`, or
`table_row`), never on a chunk id. This is required because the two chunking
configurations produce different chunk ids and boundaries. Units in the same
`group` are interchangeable; different groups must all be covered.

The four source documents are:

| File | Source | SHA-256 (prefix) |
|---|---|---|
| `11-hash2.pdf` | UW CSE 332, Hash Tables II (37 pages) | `7aa2d367…` |
| `11-hash2.pptx` | Same lecture, PPTX version (37 slides) | `8d92e93c…` |
| `USTC-ClassIn.pdf` | USTC online-teaching guide (16 pages, Chinese) | `ced8a509…` |
| `USST-syllabus.docx` | USST syllabus template with the bridge example row | `09762a9c…` |

The samples are downloaded by `test-materials/download_samples.py` into
`downloads/real-samples/`. The two UTS accessibility documents return HTTP 403
from the upstream host and are not part of the dataset.

`evaluation/datasets/qa_cases.jsonl` is the deprecated 6-case v1 search set. It
is retained only as historical material and for legacy smoke fixtures; it is
not used by any reported experiment.

## Quick Start

Run from the repository root. The commands below use the Windows virtualenv
path; on macOS/Linux replace `.venv/Scripts/python.exe` with
`.venv/bin/python`.

```sh
# 1. Environment
uv sync --locked --package kotaemon --package ktem

# 2. Dataset structure and distribution
.venv/Scripts/python.exe -m evaluation.scripts.validate_dataset \
    evaluation/datasets/qa_cases_v2.jsonl --min-cases 30 \
    --target-cases 30 --strict-target

# 3. Verbatim evidence verification (requires downloads/real-samples/)
.venv/Scripts/python.exe -m evaluation.scripts.verify_evidence \
    --dataset evaluation/datasets/qa_cases_v2.jsonl --strict

# 4. Unit tests (offline; no model calls)
.venv/Scripts/python.exe -m pytest evaluation -q
```

A real retrieval run is optional and requires an index plus an embedding key:

```sh
# Build one index per process (1024/256 or 512/128)
.venv/Scripts/python.exe -m evaluation.scripts.build_indexes \
    --docs-dir downloads/real-samples --out-dir .eval-index/1024-256 \
    --profile 1024-256 --env-file .env

# 30-question retrieval; artifacts go outside the repository
.venv/Scripts/python.exe -m evaluation.scripts.run_batch_retrieval \
    --index-dir .eval-index/1024-256 \
    --cases evaluation/datasets/qa_cases_v2.jsonl \
    --out-dir ../.eval-data/retrieval-real-30 \
    --tag 1024-256-vector-k5 --top-k 5 --env-file .env --resume
```

A real answer run consumes a retrieval file and calls DeepSeek:

```sh
.venv/Scripts/python.exe -m evaluation.scripts.run_batch_e2e \
    --retrieval ../.eval-data/retrieval-real-30/1024-256-vector-k5/retrieval.json \
    --cases evaluation/datasets/qa_cases_v2.jsonl \
    --out-dir ../.eval-data/answer-e2e-30 --env-file .env --resume
```

Grading and scoring:

```sh
.venv/Scripts/python.exe -m evaluation.scripts.make_grading \
    --answers ../.eval-data/answer-e2e-30/answers.json \
    --retrieval ../.eval-data/retrieval-real-30/1024-256-vector-k5/retrieval.json \
    --out ../.eval-data/grading-30/grading_all.md

.venv/Scripts/python.exe -m evaluation.scripts.run_answer_eval \
    --cases evaluation/datasets/qa_cases_v2.jsonl \
    --predictions ../.eval-data/answer-e2e-30/predictions.jsonl \
    --labels ../.eval-data/grading-30/labels.jsonl \
    --citation-labels ../.eval-data/grading-30/citation_labels.jsonl \
    --out ../.eval-data/grading-30/report_scored.json
```

`.env` and all `../.eval-data/` / `.eval-index/` artifacts are local only and
are not committed.

## Evaluation Metrics

### Retrieval

| Metric | Definition | Denominator |
|---|---|---|
| Strict Hit@K | fraction of questions with at least one gold evidence unit in Top-K | 25 answerable questions |
| MRR@K | mean reciprocal rank of the first matching gold unit | 25 answerable questions |
| Strict Recall@K | fraction of required evidence **groups** covered | 25 answerable questions |
| Content Hit@K | fraction with at least one verified gold quote in the Top-K text | 25 answerable questions |
| Content Recall@K | mean fraction of gold quotes found in the Top-K text | 25 answerable questions |

Strict matching uses `file + page/slide/section`, with page-label
normalisation. Content matching is a **supplementary diagnostic** for two known
strict-matching artefacts: PDF/PPTX duplicates and DOCX chunks without page or
section metadata. It does not replace the strict numbers.

### Answers and Citations

| Metric | Definition | Source |
|---|---|---|
| Answer Accuracy | mean of `correct=1.0`, `partial=0.5`, `incorrect=0.0` | human labels |
| Strict Accuracy | fraction of answers labelled `correct` | human labels |
| Citation Validity | fraction of non-abstained answers whose `【n】` markers all exist in the evidence index | automatic |
| Citation Support | fraction of cited passages that actually support the sentence | human labels |
| Refusal Accuracy | fraction of unanswerable questions correctly refused | automatic detector |
| False Refusal Rate | fraction of answerable questions wrongly refused | automatic detector |
| Balanced Accuracy | `(refusal_accuracy + 1 - false_refusal_rate) / 2` | automatic |

Unlabelled answers are `not_judged` and excluded from the denominator; they are
never converted to zero. Abstentions have no citations and are excluded from the
citation metrics. Citation validity and citation support are reported
separately: a valid citation index can still point to a passage that does not
support the claim.

## Experimental Results

All numbers below are from the real 30-question runs. The two chunking
configurations use the same four documents, the same 30 questions, the same
embedding model (`text-embedding-v4`, 2048 dimensions), and Top-K = 5.

### Retrieval comparison

| Metric | 1024/256 | 512/128 |
|---|---:|---:|
| Strict Hit@5 | 0.8400 | 0.8400 |
| MRR@5 | 0.5833 | 0.5633 |
| Strict Recall@5 | 0.7560 | 0.7560 |
| Content Hit@5 | 1.0000 | 1.0000 |
| Content Recall@5 | 0.9107 | 0.8973 |

Hit@5 and Strict Recall@5 are unchanged. The only aggregate regression is
`v2-s06`: its page-4 evidence is split across chunks in 512/128, the chunk
containing "点击新建 ClassIn 课堂" falls outside the Top-5, and the first strict
match moves from rank 1 to rank 2. Content Hit@5 = 1.0 means only that at least
one gold quote was found in each Top-5; it does not mean every strict or
group-level retrieval requirement was met.

### Answer quality (1024/256 only)

| Metric | Value |
|---|---:|
| Answer Accuracy | **0.9400** (22 correct, 3 partial, 0 incorrect; 25 judged) |
| Strict Accuracy | **0.8800** |
| Citation Validity | **1.0000** |
| Citation Support | **0.9608** (49 / 51 citations) |
| Refusal Accuracy | **1.0000** (5 / 5) |
| False Refusal Rate | **0.0000** |

Citation support was 49/51: two citations in `v2-s04` supported "use a prime
table size" but not the answer's claim that the material did not explain why.

**512/128 retrieval was evaluated, but 512/128 answers were not generated and
were not graded.** The 1024/256 answer metrics must not be applied to 512/128.

## Output Files

Real experiment artifacts are written outside the repository (`../.eval-data/`)
so that indexes, logs and model outputs are never committed. `evaluation/results/`
is reserved for small, reviewed report files and currently contains only
`.gitkeep`.

| Artifact | Produced by | Contents |
|---|---|---|
| `<index>/index.json` (1024/256) | `build_indexes.py` | chunking params, embedding model, per-file SHA-256, `vector_count`, `indexed_documents` |
| `<index>/index.json` (512/128) | A's `scripts/part_a_index.py` | chunking params, embedding model/dimensions, `collection`, `complete`, per-file `source_id`/`error`; no per-file hashes or vector counts |
| `retrieval.json` | `run_batch_retrieval.py` | per-question Top-5 results and metrics |
| `per_case.jsonl` | batch runners | one append-only record per question |
| `failures.json` | batch runners | failed and missing question ids |
| `summary.json` | `run_batch_retrieval.py` | counts, metrics, per-category breakdown |
| `run_manifest.json` | batch runners | git commit, code/data/index hashes, model config, timestamps |
| `answers.json` | `run_batch_e2e.py` | generated answers, evidence index, citation validation |
| `predictions.jsonl` | `run_batch_e2e.py` | answer text per question for scoring |
| `report_auto.json` | `run_batch_e2e.py` | automatic metrics; human fields remain `null` |
| `grading_all.md` | `make_grading.py` | human grading sheet with cited chunk text |
| `labels.jsonl` / `citation_labels.jsonl` | human grading | answer and per-citation labels |
| `report_scored.json` | `run_answer_eval.py` | final answer/citation/refusal metrics with label hashes |

## Limitations

- **Only 1024/256 has end-to-end answer results.** 512/128 has retrieval only;
  its answer quality, citation support, and refusal behaviour are unverified.
- **No reranker is enabled.** The retrieval runs use `VectorRetrieval` with
  `rerankers=[]`; no reranking experiment was performed.
- **Only vector retrieval at Top-K = 5 was run.** Hybrid retrieval, RRF, and
  other Top-K values were not evaluated.
- **The dataset is small and single-domain.** 30 questions, four documents, one
  human grader; no inter-annotator agreement and no significance testing.
- **Strict matching has known blind spots.** PDF/PPTX duplicates and DOCX
  chunks without page/section metadata can be counted as misses; the content
  metrics are a diagnostic, not a replacement.
- **The 18-cell experiment matrix is a planning scaffold.** Only the two
  vector/k=5 cells were executed through the dedicated batch runners; the
  matrix's `blocked` flags were not used as experimental results.
- **Indexes and experiment artifacts are local.** They are gitignored and must
  be rebuilt or copied separately; reproduction requires the source documents
  and API keys.
