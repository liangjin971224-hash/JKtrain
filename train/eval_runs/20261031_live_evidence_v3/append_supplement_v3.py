"""Append two documented follow-up searches without rewriting first-pass evidence."""

from __future__ import annotations

import json
from dataclasses import asdict

from collect_v3 import (
    BASE, Bm25SearchAdapter, CachedEmbeddingAdapter, ChunkMeta, GraphDatabase,
    IntentClassifier, MilvusClient, MilvusSearchAdapter, Neo4jGraphAdapter,
    RetrieveUseCase, get_settings, save,
)

FOLLOW_UPS: dict[str, tuple[str, str]] = {
    "ev2-19": ("大豆蛋白 不适宜人群", "首轮把大豆分离蛋白粉误识别为P004；大豆蛋白为实时Neo4j P005已记录别名，重新解析身份"),
    "ev2-35": ("绝对化表述 专家推荐", "首轮及首次补检未取到专家推荐的直接规则；绝对化表述章节来自本轮ev2-06实际召回D005_c013，非题内references/expect"),
}


def main() -> None:
    """Run the existing retrieval stack and append auditable supplementary stages."""
    settings = get_settings()
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password), connection_timeout=5)
    client = MilvusClient(uri=settings.milvus_uri, timeout=15)
    try:
        chunks = [json.loads(s) for s in (BASE / "rag_chunks_v3.jsonl").read_text(encoding="utf-8").splitlines()]
        keyword = Bm25SearchAdapter()
        keyword.build_index([ChunkMeta(chunk_id=r["chunk_id"], doc_id=r["doc_id"], text=r["text"], title=r["title"], section=r["section"], permission_level=r["permission_level"]) for r in chunks])
        encoder = MilvusSearchAdapter(client, settings=settings)
        queries = [x[0] for x in FOLLOW_UPS.values()]
        vector = CachedEmbeddingAdapter(client, settings=settings)
        vector.cached = dict(zip(queries, encoder.embed(queries)))
        retriever = RetrieveUseCase(graph=Neo4jGraphAdapter(driver, settings.neo4j_database), vector=vector, keyword=keyword)
        classifier = IntentClassifier()
        supplement = json.loads((BASE / "supplemental_v3.json").read_text(encoding="utf-8"))
        extra = {}
        for qid, (query, reason) in FOLLOW_UPS.items():
            intent = classifier.classify(query)
            stage = {"query": query, "reason": reason, "intent": asdict(intent), "retrieval": asdict(retriever.execute(query, intent=intent.label, vector_collection=settings.coll_training, allowed_permissions=("public", "internal"), top_k=20, fused_limit=10))}
            extra[qid] = stage
            supplement["per_question"].setdefault(qid, {}).setdefault("additional_results", []).append(stage)
            print(qid, [h["chunk_id"] for h in stage["retrieval"]["fused"]], [h["product_id"] for h in stage["retrieval"]["graph_hits"]], flush=True)
        supplement["actual_query_count"] += len(extra)
        save(BASE / "supplemental_v3_extra_raw.json", extra)
        save(BASE / "supplemental_v3.json", supplement)
    finally:
        client.close()
        driver.close()


if __name__ == "__main__":
    main()
