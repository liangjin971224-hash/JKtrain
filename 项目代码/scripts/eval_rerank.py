"""Rerank 效果评测：对比「重排前」与「重排后」的 R@1 / R@3 / R@5。

★ 为什么必须做这个评测：
  单测只能证明「编排接对了」，证明不了「Rerank 真的有用」。
  如果实测 R@1 没提升，这个功能就该被移除——留着只是拖慢延迟。

用法::

    # 默认跑评测集全部 36 条
    .venv/Scripts/python.exe scripts/eval_rerank.py

    # 只看前 10 条（快速验证链路通不通）
    .venv/Scripts/python.exe scripts/eval_rerank.py --limit 10

    # 调整候选数，观察「送多少条给重排最划算」
    .venv/Scripts/python.exe scripts/eval_rerank.py --candidates 10 20 30

前置条件：Milvus 已启动且 4 个 collection 就绪（本脚本只走向量路，
不依赖 Neo4j —— 目的是单变量对比重排效果，掺入图谱会干扰归因）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_SET = ROOT.parent / "08_数据入库" / "milvus" / "eval_set.json"


def load_queries(limit: int | None = None) -> list[dict]:
    data = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    qs = data["queries"]
    return qs[:limit] if limit else qs


def hit_level(hit, labels: list[str]) -> bool:
    """该条命中是否算「命中标注」。

    与 `eval_result.json` 的口径一致：章节标题或正文包含任一 label 即算命中。
    """
    if hit.chunk is None:
        return False
    text = f"{hit.chunk.title} {hit.chunk.section} {hit.chunk.text}"
    return any(lab in text for lab in labels)


def rank_of_first_hit(hits: list, labels: list[str]) -> int | None:
    for i, h in enumerate(hits, 1):
        if hit_level(h, labels):
            return i
    return None


def recall_at(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--candidates", type=int, nargs="+", default=[20],
                    help="送重排的候选条数，可给多个值做对比")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    from config.settings import Settings
    from config.assembly import build_container
    from domain.value_objects import FusedHit

    settings = Settings()
    container = build_container(settings)
    if container.vector is None:
        print("向量端口不可用，退出")
        return 1

    # 构造重排器
    from infrastructure.search.bge_reranker_adapter import BgeRerankerAdapter

    reranker = BgeRerankerAdapter(
        model_dir=settings.rerank_model_dir,
        batch_size=settings.rerank_batch_size,
        max_length=settings.rerank_max_length,
    )

    queries = load_queries(args.limit)
    print(f"评测集: {len(queries)} 条 | 向量库: {settings.coll_training}")
    print(f"重排模型: {Path(settings.rerank_model_dir).name}\n")

    # ---- 先跑一次基线：收集每题「重排前」的候选 ----
    baselines: list[dict] = []
    t_recall: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        raw = container.vector.search(
            q["query"], collection=settings.coll_training, top_k=20,
            allowed_permissions=("public", "internal"),
        )
        t_recall.append(time.perf_counter() - t0)
        fused = [FusedHit(chunk_id=h.chunk.chunk_id, fused_score=h.score,
                          chunk=h.chunk) for h in raw]
        baselines.append({"q": q, "hits": fused})

    base_ranks = [rank_of_first_hit(b["hits"], b["q"]["labels"])
                  for b in baselines]
    print("=" * 68)
    print("【基线】纯向量召回（等于重排前的 RRF 单路近似）")
    print(f"  R@1 {recall_at(base_ranks, 1):.3f} | "
          f"R@3 {recall_at(base_ranks, 3):.3f} | "
          f"R@5 {recall_at(base_ranks, 5):.3f} | MRR {mrr(base_ranks):.3f}")
    print(f"  向量召回耗时中位 {statistics.median(t_recall)*1000:.0f}ms")

    # ---- 再跑各候选数下的重排 ----
    for n_cand in args.candidates:
        ranks: list[int | None] = []
        t_rerank: list[float] = []
        improved = worsened = same = 0

        for b in baselines:
            hits = b["hits"][:n_cand]
            if not hits:
                ranks.append(None)
                continue
            t0 = time.perf_counter()
            reranked = reranker.rerank(b["q"]["query"], hits,
                                       top_k=args.top_k)
            t_rerank.append(time.perf_counter() - t0)
            ranks.append(rank_of_first_hit(reranked, b["q"]["labels"]))

        # 逐题对比首命中排名的变化
        for base_r, new_r in zip(base_ranks, ranks):
            b_ok = base_r is not None and base_r <= args.top_k
            n_ok = new_r is not None and new_r <= args.top_k
            if n_ok and not b_ok:
                improved += 1
            elif b_ok and not n_ok:
                worsened += 1
            else:
                same += 1

        print("=" * 68)
        print(f"【重排】候选 {n_cand} 条 → 取前 {args.top_k} 条")
        print(f"  R@1 {recall_at(ranks, 1):.3f} | "
              f"R@3 {recall_at(ranks, 3):.3f} | "
              f"R@5 {recall_at(ranks, 5):.3f} | MRR {mrr(ranks):.3f}")
        if t_rerank:
            print(f"  重排耗时 中位 {statistics.median(t_rerank)*1000:.0f}ms | "
                  f"最大 {max(t_rerank)*1000:.0f}ms")
        print(f"  Top{args.top_k} 命中题数变化：改善 {improved} / "
              f"恶化 {worsened} / 不变 {same}")

        d1 = recall_at(ranks, 1) - recall_at(base_ranks, 1)
        print(f"  → R@1 变化 {d1:+.3f}")

    container.close() if hasattr(container, "close") else None
    return 0


if __name__ == "__main__":
    sys.exit(main())
