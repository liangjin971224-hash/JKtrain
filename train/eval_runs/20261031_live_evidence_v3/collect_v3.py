"""Read-only live retrieval for v3 questions, excluding evaluation evidence and expectations."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
APP = ROOT / "项目代码"
sys.dont_write_bytecode = True
sys.path.insert(0, str(APP))
sys.path.insert(0, str(BASE.parent / "20261031_live_evidence"))

from collect_evidence import (
    Bm25SearchAdapter, CachedEmbeddingAdapter, ChunkMeta, GraphDatabase,
    IntentClassifier, MilvusClient, MilvusSearchAdapter, Neo4jGraphAdapter,
    READ_ACCESS, RetrieveUseCase,
)
from config.settings import get_settings


def save(path: Path, value: Any) -> None:
    """Write a non-secret JSON evidence artifact."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def protected_hashes() -> dict[str, str]:
    """Hash inputs and application source to detect accidental changes."""
    paths = [ROOT / "train/data/eval_questions_v3.jsonl", ROOT / "train/scripts/60_score_eval.py"]
    paths.extend((ROOT / "train/data").glob("samples_*.jsonl"))
    for folder in ("application", "config", "domain", "infrastructure", "interfaces"):
        paths.extend(p for p in (APP / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def main() -> None:
    """Capture fresh graph/RAG snapshots and first-pass fused Top10 results."""
    t0 = time.perf_counter()
    source = ROOT / "train/data/eval_questions_v3.jsonl"
    # Discard references/expect before building any search request.
    questions = [{"qid": row["qid"], "question": row["question"]} for row in map(json.loads, source.read_text(encoding="utf-8").splitlines())]
    assert len(questions) == 45 and len({row["qid"] for row in questions}) == 45
    save(BASE / "protected_hashes_before.json", protected_hashes())
    settings = get_settings()
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password), connection_timeout=5)
    client = MilvusClient(uri=settings.milvus_uri, timeout=15)
    try:
        graph = Neo4jGraphAdapter(driver, settings.neo4j_database)
        with driver.session(database=settings.neo4j_database, default_access_mode=READ_ACCESS) as session:
            pids = [row["pid"] for row in session.run("MATCH (p:Product) RETURN p.product_id AS pid ORDER BY pid")]
        products = {pid: asdict(graph.get_product_facts(pid)) for pid in pids}
        save(BASE / "neo4j_products_v3.json", products)
        save(BASE / "neo4j_catalog_v3.json", [asdict(row) for row in graph.list_efficacy_catalog()])
        chunks = client.query(collection_name=settings.coll_training,
            filter='review_status == "approved" and permission_level in ["public", "internal"]',
            output_fields=["chunk_id", "doc_id", "text", "title", "section", "permission_level", "review_status", "source_file"], limit=2000)
        chunks.sort(key=lambda row: row["chunk_id"])
        with (BASE / "rag_chunks_v3.jsonl").open("w", encoding="utf-8") as fh:
            for row in chunks:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        keyword = Bm25SearchAdapter()
        keyword.build_index([ChunkMeta(chunk_id=r["chunk_id"], doc_id=r["doc_id"], text=r["text"], title=r["title"], section=r["section"], permission_level=r["permission_level"]) for r in chunks])
        encoder = MilvusSearchAdapter(client, settings=settings)
        print(f"Live snapshot: {len(products)} products, {len(chunks)} approved chunks; encoding questions", flush=True)
        vectors = encoder.embed([row["question"] for row in questions])
        vector = CachedEmbeddingAdapter(client, settings=settings)
        vector.cached = dict(zip([row["question"] for row in questions], vectors))
        retriever = RetrieveUseCase(graph=graph, vector=vector, keyword=keyword)
        classifier = IntentClassifier()
        results = []
        with (BASE / "retrieval_v3.jsonl").open("w", encoding="utf-8") as fh:
            for question in questions:
                intent = classifier.classify(question["question"])
                result = retriever.execute(question["question"], intent=intent.label, vector_collection=settings.coll_training, allowed_permissions=("public", "internal"), top_k=20, fused_limit=10)
                record = {**question, "search_input": question["question"], "intent": asdict(intent), "retrieval": asdict(result), "facts": [products[hit.product_id] for hit in result.graph_hits]}
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                fh.flush()
                results.append(record)
                print(question["qid"], [hit.chunk_id for hit in result.fused], flush=True)
        save(BASE / "collection_meta_v3.json", {
            "collected_at_utc": datetime.now(timezone.utc).isoformat(), "eval_file": str(source),
            "eval_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "questions": 45,
            "graph_database": settings.neo4j_database, "graph_products": len(products),
            "rag_collection": settings.coll_training, "approved_chunks": len(chunks),
            "allowed_permissions": ["public", "internal"], "top_k": 20, "fused_limit": 10,
            "classification": "existing IntentClassifier, rules only, no LLM fallback",
            "embedding_model": Path(settings.embed_model_dir).name, "embedding_device": str(encoder._model.device),
            "embedding_batch_cache": "same project encoder; no search/evaluation labels supplied",
            "nonempty_retrieval": sum(bool(row["retrieval"]["fused"]) for row in results),
            "degraded_questions": sum(bool(row["retrieval"]["degraded"]) for row in results),
            "degraded_channel_count": sum(len(row["retrieval"]["degraded"]) for row in results),
            "graph_hit_questions": sum(bool(row["facts"]) for row in results),
            "generation_model_called": False, "candidate_status": "pending_human_review",
            "elapsed_sec": round(time.perf_counter() - t0, 3),
        })
    finally:
        client.close()
        driver.close()


if __name__ == "__main__":
    main()
