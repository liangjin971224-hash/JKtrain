"""Explicit supplementary retrieval, isolated from the immutable first-pass file."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from collect_v3 import (
    BASE, Bm25SearchAdapter, CachedEmbeddingAdapter, ChunkMeta, GraphDatabase,
    IntentClassifier, MilvusClient, MilvusSearchAdapter, Neo4jGraphAdapter,
    RetrieveUseCase, get_settings, save,
)

QUERIES: dict[str, tuple[str, str]] = {
    "ev2-04": ("电视宣传以外，防癌说法 合规禁用分类", "问题中的防癌表述需直接分类条款；查询未使用题内references/expect"),
    "ev2-17": ("服用治疗药物 不适宜人群", "从实际召回D006_c013的服药语句扩展检索完整人群条款"),
    "ev2-25": ("P004 基础信息 包装规格", "P004来自首轮图谱身份；份数必须有原文，不能计算"),
    "ev2-34": ("这是什么东西 应答边界", "问题词加项目应答边界主题，补产品介绍流程"),
    "ev2-35": ("电视 专家推荐 合规要求", "直接改写问题中的电视专家推荐场景"),
    "ev2-37": ("体脂仪 三条铁律", "原问题对应直接规则条款未进入首轮Top10"),
    "ev2-43": ("客户头晕 身体不适 红旗征", "症状问题补用户要求的红旗征处理依据"),
}
SHARED_QUERY = "保健食品不是药物，不能代替药物治疗疾病 法定警示语"
HEALTH_TOPICS = [f"ev2-{i:02}" for i in range(1, 27)] + ["ev2-34", "ev2-35", "ev2-36"]


def main() -> None:
    """Execute documented query rewrites with the same existing retrieval pipeline."""
    settings = get_settings()
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password), connection_timeout=5)
    client = MilvusClient(uri=settings.milvus_uri, timeout=15)
    try:
        rows = [json.loads(line) for line in (BASE / "rag_chunks_v3.jsonl").read_text(encoding="utf-8").splitlines()]
        keyword = Bm25SearchAdapter()
        keyword.build_index([ChunkMeta(chunk_id=r["chunk_id"], doc_id=r["doc_id"], text=r["text"], title=r["title"], section=r["section"], permission_level=r["permission_level"]) for r in rows])
        queries = [value[0] for value in QUERIES.values()] + [SHARED_QUERY]
        encoder = MilvusSearchAdapter(client, settings=settings)
        embeddings = encoder.embed(queries)
        vector = CachedEmbeddingAdapter(client, settings=settings)
        vector.cached = dict(zip(queries, embeddings))
        retriever = RetrieveUseCase(graph=Neo4jGraphAdapter(driver, settings.neo4j_database), vector=vector, keyword=keyword)
        classifier = IntentClassifier()

        def retrieve(query: str) -> dict[str, Any]:
            """Run one query with existing classification and Top10 fusion."""
            intent = classifier.classify(query)
            return {"query": query, "intent": asdict(intent), "retrieval": asdict(retriever.execute(query, intent=intent.label, vector_collection=settings.coll_training, allowed_permissions=("public", "internal"), top_k=20, fused_limit=10))}

        per_question = {}
        for qid, (query, reason) in QUERIES.items():
            per_question[qid] = {"reason": reason, **retrieve(query)}
            print(qid, [h["chunk_id"] for h in per_question[qid]["retrieval"]["fused"]], flush=True)
        shared = {"warning": {"reason": "十条作答规则要求的法定警示语；只执行一次共同查询，候选按实际引用关联，不伪造逐题执行次数", "linked_qids": HEALTH_TOPICS, **retrieve(SHARED_QUERY)}}
        save(BASE / "supplemental_v3.json", {"collected_at_utc": datetime.now(timezone.utc).isoformat(), "per_question": per_question, "shared_queries": shared, "actual_query_count": len(QUERIES) + 1, "top_k": 20, "fused_limit": 10, "gold_evidence_used": False, "first_pass_file_overwritten": False})
        print("shared warning", [h["chunk_id"] for h in shared["warning"]["retrieval"]["fused"]], flush=True)
    finally:
        client.close()
        driver.close()


if __name__ == "__main__":
    main()
