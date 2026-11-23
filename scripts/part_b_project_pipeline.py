"""Build and query a separate index using the project's actual file pipelines."""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

NAMES = ["11-hash2.pdf", "11-hash2.pptx", "UTS-accessible-word.docx",
         "USST-syllabus.docx", "UTS-accessible-powerpoint.pptx", "USTC-ClassIn.pdf"]
CASES = [
    {"file": NAMES[0], "question": "按二次探测依次把 89、18、49、58、79 插入大小为 10 的空哈希表，哪些槽位有值？", "expected": "第25页"},
    {"file": NAMES[1], "question": "原槽位被占后，二次探测的前四个偏移是多少？当表大小为素数且负载因子小于一半时，能保证什么？", "expected": "第24张及第26或27张"},
    {"file": NAMES[2], "question": "Mac 版 Word 的样式窗格中，Normal 和 Heading 1 之间是什么？截图列出了哪些标题级别？", "expected": "Headings章节"},
    {"file": NAMES[3], "question": "桥梁工程的基本知识模块支持哪些课程目标？学生有哪两项学习任务？", "expected": "教学模块的桥梁示例行"},
    {"file": NAMES[4], "question": "打开 Check Accessibility 用哪个选项卡？表格列用交替底色的选项叫什么？", "expected": "第4张和第17张"},
    {"file": NAMES[5], "question": "云盘支持哪些课件格式？打开云盘课件内容上课这页截图中，与试卷和授权有关的列表项叫什么？", "expected": "第8页"},
]
DIAGNOSTIC_CASES = [
    {"file": NAMES[1], "question": "原槽位被占后，二次探测的前四个偏移是多少？", "expected": "第24张"},
    {"file": NAMES[1], "question": "当表大小为素数且负载因子小于一半时，二次探测能保证什么？", "expected": "第26或27张"},
]


def fuse_query_results(rankings: list[list[dict]], top_k: int, rrf_k: int = 60) -> list[dict]:
    """Fuse manually supplied subqueries by chunk ID without mixing similarity scales."""
    merged: dict[str, dict] = {}
    for query_index, ranking in enumerate(rankings):
        seen: set[str] = set()
        for rank, doc in enumerate(ranking, start=1):
            doc_id = doc["id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            if doc_id not in merged:
                merged[doc_id] = {**doc, "score": 0.0, "score_type": "rrf",
                                  "subquery_hits": []}
            merged[doc_id]["score"] += 1 / (rrf_k + rank)
            merged[doc_id]["subquery_hits"].append({
                "query_index": query_index, "rank": rank,
                "original_score": doc["score"],
                "original_score_type": doc["score_type"],
            })
    return sorted(merged.values(), key=lambda doc: -doc["score"])[:top_k]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index-dir", type=Path, required=True)
    p.add_argument("--env-file", type=Path, required=True)
    p.add_argument("--samples-dir", type=Path, required=True)
    p.add_argument("--query-only", action="store_true", help="query a previously built experiment")
    p.add_argument("--diagnostic", action="store_true", help="run split-question diagnostics")
    p.add_argument("--file", choices=NAMES, help="query just this indexed file")
    p.add_argument("--question", help="custom query; requires --query-only and --file")
    p.add_argument("--subquestion", action="append", help="manual subquery; repeat at least twice to fuse by RRF")
    p.add_argument("--result-name", help="write query results to this JSON filename inside --index-dir")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--mode", choices=["vector", "hybrid"], default="vector")
    args = p.parse_args()
    if args.diagnostic and not args.query_only:
        p.error("--diagnostic requires --query-only")
    if (args.question is None) != (args.file is None) or (args.question is not None and (not args.query_only or args.diagnostic or not args.question.strip())):
        p.error("custom queries require --query-only, --file, and a nonempty --question")
    if args.subquestion and (len(args.subquestion) < 2 or not args.question or
                             any(not question.strip() for question in args.subquestion)):
        p.error("--subquestion requires a custom --question and at least two nonempty values")
    if args.top_k < 1:
        p.error("--top-k must be positive")
    if args.result_name and (not args.query_only or not args.result_name.endswith(".json") or
                             Path(args.result_name).name != args.result_name or
                             args.result_name in {"results.json", "results-query.json", "results-diagnostic.json", "results-after-fix.json"}):
        p.error("--result-name requires --query-only and a new JSON filename in --index-dir")
    args.index_dir = args.index_dir.resolve()
    if not args.env_file.is_file() or any(not (args.samples_dir / n).is_file() for n in NAMES):
        p.error("the env file and all six samples must exist")
    if args.query_only:
        if not all((args.index_dir / name).exists() for name in
                   ("pipeline.sqlite3", "vectors/chroma.sqlite3", "documents/docstore.lance", "results.json")):
            p.error("the experiment index is incomplete")
    else:
        args.index_dir.mkdir(parents=True, exist_ok=False)
        (args.index_dir / "files").mkdir()
    os.environ["PART_B_PIPELINE_DIR"] = str(args.index_dir)
    os.environ["THEFLOW_SETTINGS_MODULE"] = "part_b_pipeline_settings"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from sqlalchemy import Column, Integer, JSON, String
    from sqlalchemy.ext.mutable import MutableDict
    from sqlalchemy.orm import declarative_base
    from ktem.db.engine import engine
    from ktem.index.file.pipelines import IndexDocumentPipeline, DocumentRetrievalPipeline
    from kotaemon.storages import ChromaVectorStore, LanceDBDocumentStore
    from part_b_retrieval import create_embedding

    base = declarative_base()
    class Source(base):
        __tablename__ = "source"
        id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
        name = Column(String, unique=True)
        path = Column(String)
        size = Column(Integer, default=0)
        user = Column(String, default="")
        note = Column(MutableDict.as_mutable(JSON), default=dict)

    class Index(base):
        __tablename__ = "index_record"
        id = Column(Integer, primary_key=True, autoincrement=True)
        source_id = Column(String)
        target_id = Column(String)
        relation_type = Column(String)
        user = Column(String, default="")

    base.metadata.create_all(engine)
    embedding = create_embedding(args.env_file)
    vs = ChromaVectorStore(path=str(args.index_dir / "vectors"))
    ds = LanceDBDocumentStore(path=str(args.index_dir / "documents"))
    pipeline = IndexDocumentPipeline(embedding=embedding, reader_mode="default",
                                     run_embedding_in_thread=False)
    for obj in (pipeline,):
        obj.Source, obj.Index, obj.VS, obj.DS = Source, Index, vs, ds
        obj.FSPath, obj.user_id, obj.private = args.index_dir / "files", "", False
        obj.chunk_size, obj.chunk_overlap = 1024, 256

    if args.query_only:
        summary = json.loads((args.index_dir / "results.json").read_text(encoding="utf-8"))
        summary["cases"] = []
        output_name = args.result_name or ("results-query.json" if args.question else ("results-diagnostic.json" if args.diagnostic else "results-after-fix.json"))
        result_path = args.index_dir / output_name
        if args.result_name and result_path.exists():
            p.error(f"result file already exists: {result_path}")
    else:
        summary = {"files": {}, "cases": [], "model": "text-embedding-v4", "dimensions": 2048}
        result_path = args.index_dir / "results.json"
    summary["retrieval"] = {"mode": args.mode, "top_k": args.top_k, "reranking": False}
    if args.subquestion:
        summary["retrieval"].update({"strategy": "manual_subquestions_rrf",
                                     "per_subquery_top_k": args.top_k, "rrf_k": 60})
    def save():
        result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for name in ([] if args.query_only else NAMES):
        print(f"INDEXING {name}", file=sys.stderr, flush=True)
        gen = pipeline.stream(args.samples_dir / name, reindex=False)
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            file_ids, errors, _ = stop.value
        summary["files"][name] = {"source_id": file_ids[0], "error": errors[0]}
        save()
        print(f"INDEXED {name}: {'ok' if not errors[0] else 'failed'}", file=sys.stderr, flush=True)

    retriever = DocumentRetrievalPipeline(embedding=embedding, top_k=args.top_k, retrieval_mode=args.mode,
                                          rerankers=[], llm_scorer=None, get_extra_table=False)
    retriever.Source, retriever.Index, retriever.VS, retriever.DS = Source, Index, vs, ds
    retriever.FSPath, retriever.user_id = args.index_dir / "files", ""
    query_cases = [{"file": args.file, "question": args.question, "expected": None}] if args.question else (DIAGNOSTIC_CASES if args.diagnostic else CASES)
    for case in query_cases:
        file_info = summary["files"][case["file"]]
        item = dict(case)
        if file_info["error"] or not file_info["source_id"]:
            item["error"] = "ingestion failed"
        else:
            print(f"QUERY {case['file']}", file=sys.stderr, flush=True)
            try:
                rankings = []
                for query_text in (args.subquestion or [case["question"]]):
                    docs = retriever(text=query_text, doc_ids=[file_info["source_id"]])
                    rankings.append([{"id": d.doc_id, "file_name": d.metadata.get("file_name"),
                        "page": d.metadata.get("page_label"), "section": d.metadata.get("section"),
                        "type": d.metadata.get("type", "text"), "score": None if d.score == -1.0 else d.score,
                        "score_type": "text_unscored" if d.score == -1.0 else "vector_store_similarity",
                        "text": str(d.content) if d.content is not None else d.text} for d in docs])
                if args.subquestion:
                    item["subqueries"] = [{"question": question, "results": results}
                                          for question, results in zip(args.subquestion, rankings)]
                    item["results"] = fuse_query_results(rankings, args.top_k)
                else:
                    item["results"] = rankings[0]
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
        summary["cases"].append(item)
        save()
    print(json.dumps({"index_dir": str(args.index_dir), "result_file": str(result_path),
                      "files": summary["files"],
                      "cases_completed": len(summary["cases"])}, ensure_ascii=False, indent=2))
    if any(item.get("error") for item in summary["files"].values()) or any(
        item.get("error") for item in summary["cases"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
