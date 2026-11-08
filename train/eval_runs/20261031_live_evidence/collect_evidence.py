"""Snapshot live knowledge and replay retrieval without calling a generation model."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / "项目代码"
sys.path.insert(0, str(APP))

from config.settings import get_settings
from domain.services.intent_classifier import IntentClassifier
from domain.value_objects import ChunkMeta
from application.use_cases.retrieve import RetrieveUseCase
from infrastructure.graph.neo4j_graph_adapter import Neo4jGraphAdapter
from infrastructure.search.bm25_adapter import Bm25SearchAdapter
from infrastructure.vector.milvus_search_adapter import MilvusSearchAdapter
from neo4j import GraphDatabase, READ_ACCESS
from pymilvus import MilvusClient


def write_json(path: Path, value: Any) -> None:
    """Save auditable non-secret data as UTF-8 JSON."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


class CachedEmbeddingAdapter(MilvusSearchAdapter):
    """Use batch-encoded question vectors with the existing search implementation."""

    cached: dict[str, list[float]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.cached[text] for text in texts]


def main() -> None:
    """Read live databases, then replay every evaluation question once."""
    start = time.perf_counter()
    output = Path(__file__).resolve().parent
    source = ROOT / "train/data/eval_questions_v2.jsonl"
    questions = [json.loads(s) for s in source.read_text(encoding="utf-8").splitlines() if s.strip()]
    settings = get_settings()
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password), connection_timeout=5)
    client = MilvusClient(uri=settings.milvus_uri, timeout=15)
    try:
        with driver.session(database=settings.neo4j_database, default_access_mode=READ_ACCESS) as session:
            pids = [row["pid"] for row in session.run("MATCH (p:Product) RETURN p.product_id AS pid ORDER BY pid")]
            raw_products = [dict(row["product"]) for row in session.run("MATCH (p:Product) RETURN p AS product ORDER BY p.product_id")]
        graph = Neo4jGraphAdapter(driver, settings.neo4j_database)
        products = {pid: asdict(graph.get_product_facts(pid)) for pid in pids}
        write_json(output / "neo4j_products.json", products)
        write_json(output / "neo4j_product_properties.json", raw_products)
        write_json(output / "neo4j_catalog.json", [asdict(item) for item in graph.list_efficacy_catalog()])

        rows = client.query(
            collection_name=settings.coll_training,
            filter='review_status == "approved" and permission_level in ["public", "internal"]',
            output_fields=["chunk_id", "doc_id", "text", "title", "section", "permission_level", "review_status", "source_file"],
            limit=2000,
        )
        rows.sort(key=lambda row: row["chunk_id"])
        with (output / "rag_chunks.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"snapshot: graph={len(products)}, approved RAG chunks={len(rows)}", flush=True)
        keyword = Bm25SearchAdapter()
        keyword.build_index([ChunkMeta(chunk_id=r["chunk_id"], doc_id=r["doc_id"], text=r["text"], title=r["title"], section=r["section"], permission_level=r["permission_level"]) for r in rows])
        encoder = MilvusSearchAdapter(client, settings=settings)
        print("encoding question batch with the project embedding model", flush=True)
        vectors = encoder.embed([row["question"] for row in questions])
        vector = CachedEmbeddingAdapter(client, settings=settings)
        vector.cached = {row["question"]: emb for row, emb in zip(questions, vectors)}
        retriever = RetrieveUseCase(graph=graph, vector=vector, keyword=keyword)
        classifier = IntentClassifier()
        results: list[dict[str, Any]] = []
        with (output / "retrieval.jsonl").open("w", encoding="utf-8") as fh:
            for question in questions:
                intent = classifier.classify(question["question"])
                result = retriever.execute(question["question"], intent=intent.label, vector_collection=settings.coll_training, allowed_permissions=("public", "internal"), top_k=20, fused_limit=10)
                record = {
                    "qid": question["qid"], "question": question["question"],
                    "intent": asdict(intent), "retrieval": asdict(result),
                    "facts": [products[hit.product_id] for hit in result.graph_hits],
                    "expected_doc_ids_for_diagnostics_only": [ref["doc_id"] for ref in question["references"]],
                }
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                fh.flush()
                results.append(record)
                docs = [hit.chunk.doc_id for hit in result.fused[:5]]
                print(f"{question['qid']}: graph={[hit.product_id for hit in result.graph_hits]}, top5={docs}, degraded={list(result.degraded)}", flush=True)
        write_json(output / "collection_meta.json", {
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "eval_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "questions": len(questions), "graph_products": len(products), "rag_chunks": len(rows),
            "permissions": ["public", "internal"], "review_status": "approved",
            "embedding_model": str(Path(settings.embed_model_dir).name),
            "embedding_device": str(encoder._model.device),
            "classification": "project rules; no LLM fallback", "top_k": 20, "fused_limit": 10,
            "generation_model_called": False,
            "degraded_questions": sum(bool(record["retrieval"]["degraded"]) for record in results),
            "elapsed_sec": round(time.perf_counter() - start, 3),
        })
    finally:
        client.close()
        driver.close()


if __name__ == "__main__":
    main()
