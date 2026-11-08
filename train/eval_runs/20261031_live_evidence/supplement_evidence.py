"""Run explicit follow-up RAG searches; keep first-pass measurements unchanged."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from collect_evidence import (
    Bm25SearchAdapter, CachedEmbeddingAdapter, ChunkMeta, GraphDatabase,
    IntentClassifier, MilvusClient, MilvusSearchAdapter, Neo4jGraphAdapter,
    RetrieveUseCase, get_settings, write_json,
)

QUERIES: dict[str, str] = {
    "ev2-01": "视力 保健功能目录",
    "ev2-04": "防癌 禁用表述",
    "ev2-28": "普通食品 可以说什么",
    "ev2-37": "体脂仪 三条铁律",
    "ev2-43": "客户头晕 身体不适 红旗征",
}


def main() -> None:
    """Collect follow-up evidence and graph aliases using the same live databases."""
    output = Path(__file__).resolve().parent
    settings = get_settings()
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password), connection_timeout=5)
    client = MilvusClient(uri=settings.milvus_uri, timeout=15)
    try:
        chunks = [json.loads(s) for s in (output / "rag_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        keyword = Bm25SearchAdapter()
        keyword.build_index([ChunkMeta(chunk_id=r["chunk_id"], doc_id=r["doc_id"], text=r["text"], title=r["title"], section=r["section"], permission_level=r["permission_level"]) for r in chunks])
        encoder = MilvusSearchAdapter(client, settings=settings)
        embeddings = encoder.embed(list(QUERIES.values()))
        vector = CachedEmbeddingAdapter(client, settings=settings)
        vector.cached = dict(zip(QUERIES.values(), embeddings))
        graph = Neo4jGraphAdapter(driver, settings.neo4j_database)
        retriever = RetrieveUseCase(graph=graph, vector=vector, keyword=keyword)
        classifier = IntentClassifier()
        result = {}
        for qid, query in QUERIES.items():
            hits = retriever.execute(query, intent=classifier.classify(query).label, vector_collection=settings.coll_training, allowed_permissions=("public", "internal"), top_k=20, fused_limit=10)
            result[qid] = {"query": query, "retrieval": asdict(hits)}
            print(qid, [h.chunk_id for h in hits.fused[:5]], flush=True)
        write_json(output / "supplemental_retrieval.json", result)

        products = json.loads((output / "neo4j_products.json").read_text(encoding="utf-8"))
        questions = [json.loads(s) for s in (output / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()]
        enriched = {}
        for row in questions:
            found = set()
            for pid, product in products.items():
                terms = [product["name"], product["name"].removeprefix("康源牌"), *product["aliases"]]
                if any(term and term.lower() in row["question"].lower() for term in terms):
                    found.add(pid)
            if found:
                enriched[row["qid"]] = {"method": "question substring against names/aliases read from Neo4j; not evaluation labels", "facts": [asdict(graph.get_product_facts(pid)) for pid in sorted(found)]}
        write_json(output / "supplemental_graph_facts.json", enriched)
    finally:
        client.close()
        driver.close()


if __name__ == "__main__":
    main()
