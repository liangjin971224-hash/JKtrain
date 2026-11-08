"""应用侧检索评测器：走**线上真实链路**（图谱 + 向量 + BM25 三路 RRF 融合）。

    python train/scripts/92_app_retrieval_eval.py --label app_current
    python train/scripts/92_app_retrieval_eval.py --label app_current \
        --compare train/eval_runs/p03_union_20261107_153451/detail.jsonl

────────────────────────────────────────────────────────────────────
和 81 号的分工
────────────────────────────────────────────────────────────────────
81 号只测 Milvus **单路**（dense / sparse / hybrid / route / union），
用来判断"检索算法本身"哪家强；
92 号实例化 `config.assembly.build_container`，走**线上同一条链路**：
    graph(Neo4j) + vector(Milvus bge-m3) + keyword(BM25, 从 Milvus 建索引)
    → RRF 融合（k=60，按意图调权）→ 可选 rerank
用来回答两个问题：
  ① 线上现在到底多少分？（P0 的所有数字都是单路测的，不能直接代表线上）
  ② P0 的 union 收益会被三路融合吃掉、保留还是放大？

输出与 81 号同构（summary.json / detail.jsonl），两者可直接用 81 号的
`--compare` 互相比对（detail 都有 qid / hit@5 / gold_rank）。

★ 只读脚本：不改应用代码、不改库、不写 Milvus。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAIN_ROOT.parent
PROJECT_CODE = PROJECT_ROOT / "项目代码"
DATA_DIR = TRAIN_ROOT / "data"
RUNS_DIR = TRAIN_ROOT / "eval_runs"

DEFAULT_EVAL = DATA_DIR / "retrieval_eval_v1_remap.jsonl"

#: 生产默认阈值（AnswerQuestionUseCase 的构造默认值，Settings 未覆盖这两项）
MIN_VECTOR_SCORE = 0.50
MIN_KEYWORD_SCORE = 14.0

#: 每路取多少、融合后留多少（与 81 号 TOP_K_SEARCH / 线上 fused_limit 对齐）
DEFAULT_TOP_K = 20
DEFAULT_FUSED_LIMIT = 10


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_81():
    """复用 81 号的指标与配对统计，保证两个评测器口径完全一致。"""
    path = Path(__file__).with_name("81_retrieval_eval.py")
    spec = importlib.util.spec_from_file_location("eval81", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="应用侧三路检索评测（线上链路）")
    parser.add_argument("--eval", default=None, help="评测集（默认 remap 205）")
    parser.add_argument("--label", default="app_current")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--fused-limit", type=int, default=DEFAULT_FUSED_LIMIT)
    parser.add_argument("--compare", default=None,
                        help="对比基线 detail.jsonl（可用 81 号的 run）")
    parser.add_argument("--gate", type=float, default=None, help="R@5 绝对门禁")
    parser.add_argument("--gate-delta", type=float, default=None,
                        help="配对增量门禁（需配合 --compare）")
    parser.add_argument("--limit", type=int, default=0,
                        help="只跑前 N 题（调试用，0=全部）")
    args = parser.parse_args()

    if args.gate_delta is not None and not args.compare:
        raise SystemExit("[FAIL] --gate-delta 需要同时给 --compare。")

    eval_path = Path(args.eval) if args.eval else DEFAULT_EVAL
    if not eval_path.exists():
        raise SystemExit(f"[FAIL] 评测集不存在：{eval_path}")
    queries = load_jsonl(eval_path)
    if args.limit:
        queries = queries[:args.limit]
    if not queries:
        raise SystemExit("[FAIL] 评测集为空")

    ev81 = _load_81()

    # ---------------- 装配线上容器 ----------------
    sys.path.insert(0, str(PROJECT_CODE))
    from application.use_cases.retrieve import RetrieveUseCase
    from config.assembly import build_container
    from config.settings import Settings
    from domain.services.intent_classifier import IntentClassifier

    print("=" * 72)
    print(f"  应用侧检索评测 label={args.label}")
    print(f"  评测集: {eval_path.name}（{len(queries)} 题）")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 72)

    t0 = time.perf_counter()
    settings = Settings()
    container = build_container(settings)
    print(f"  容器就绪 {time.perf_counter() - t0:.1f}s："
          f"{container.ready_ports()}；pending={list(container.pending) or '无'}")
    if container.vector is None or container.keyword is None:
        raise SystemExit(
            "[FAIL] 向量或关键词端口不可用，无法评测："
            f"{ {k: container.pending.get(k) for k in ('vector', 'keyword')} }")

    sparse_cols = frozenset(
        c.strip() for c in settings.sparse_recall_collections.split(",") if c.strip())
    path_head_cols = frozenset(
        c.strip() for c in settings.path_head_collections.split(",") if c.strip())
    retriever = RetrieveUseCase(
        graph=container.graph,
        vector=container.vector,
        keyword=container.keyword,
        sparse=getattr(container, "sparse", None),
        sparse_collections=sparse_cols,
        min_sparse_score=settings.min_sparse_score,
        path_head_collections=path_head_cols,
        rerank=container.rerank,
        rerank_candidate_k=settings.rerank_candidate_k,
        min_vector_score=MIN_VECTOR_SCORE,
        min_keyword_score=MIN_KEYWORD_SCORE,
    )
    classifier = IntentClassifier()          # 仅规则分类，与线上默认一致

    # ---------------- 逐题检索 ----------------
    details: list[dict] = []
    path_totals = {"graph": 0, "vector": 0, "keyword": 0, "sparse": 0, "fused": 0}
    degraded_counter: dict[str, int] = {}
    elapsed_sum = {"graph": 0.0, "vector": 0.0, "keyword": 0.0,
                   "sparse": 0.0, "rerank": 0.0}
    rerank_runs = 0

    for i, q in enumerate(queries, 1):
        intent = classifier.classify(q["question"]).label
        perms = q.get("allowed_permissions")
        res = retriever.execute(
            q["question"],
            intent=intent,
            product_ids=(),
            vector_collection=q["collection"],
            allowed_permissions=tuple(perms) if perms else None,
            top_k=args.top_k,
            fused_limit=args.fused_limit,
        )
        ranked = [h.chunk_id for h in res.fused]
        golds = q["gold_chunk_ids"]
        gold_set = set(golds)
        gold_rank = next((n for n, cid in enumerate(ranked, 1) if cid in gold_set), None)

        m = ev81.compute_metrics(ranked, golds)
        details.append({
            **m,
            "qid": q["qid"],
            "origin": q["origin"],
            "collection": q["collection"],
            "intent": intent,
            "gold_rank": gold_rank,
            "ranked_head": ranked[:10],
            "n_graph": len(res.graph_hits),
            "n_vector": len(res.vector_hits),
            "n_keyword": len(res.keyword_hits),
            "n_sparse": len(res.sparse_hits),
            "n_fused": len(res.fused),
            "degraded": sorted(res.degraded),
        })
        path_totals["graph"] += len(res.graph_hits)
        path_totals["vector"] += len(res.vector_hits)
        path_totals["keyword"] += len(res.keyword_hits)
        path_totals["sparse"] += len(res.sparse_hits)
        path_totals["fused"] += len(res.fused)
        for k in res.degraded:
            degraded_counter[k] = degraded_counter.get(k, 0) + 1
        for k, v in res.elapsed_ms.items():
            if k in elapsed_sum:
                elapsed_sum[k] += v
                if k == "rerank":
                    rerank_runs += 1
        if i % 25 == 0:
            print(f"    {i}/{len(queries)} ...")

    n = len(queries)
    r5 = ev81.mean([d["hit@5"] for d in details])
    r10 = ev81.mean([d["hit@10"] for d in details])
    mrr = ev81.mean([d["rr"] for d in details])

    def breakdown(key: str) -> dict:
        groups: dict[str, list[dict]] = {}
        for d in details:
            groups.setdefault(d[key], []).append(d)
        return {g: {"n": len(v),
                    "r@5": round(ev81.mean([x["hit@5"] for x in v]), 4),
                    "mrr": round(ev81.mean([x["rr"] for x in v]), 4)}
                for g, v in sorted(groups.items())}

    cmp_result = None
    delta_verdict = None
    if args.compare:
        cmp_path = Path(args.compare)
        if cmp_path.is_dir():
            cmp_path = cmp_path / "detail.jsonl"
        cmp_result = ev81.compare_with(cmp_path, details)
        if args.gate_delta is not None:
            delta_verdict = "PASS" if cmp_result["delta"] >= args.gate_delta else "FAIL"

    verdict = None
    if args.gate is not None:
        verdict = "PASS" if r5 >= args.gate else "FAIL"

    print("-" * 72)
    print(f"  R@5  = {r5:.4f} ({round(r5 * n)}/{n})")
    print(f"  R@10 = {r10:.4f}   MRR = {mrr:.4f}")
    print(f"  各路平均条数: graph {path_totals['graph'] / n:.2f} / "
          f"vector {path_totals['vector'] / n:.2f} / "
          f"keyword {path_totals['keyword'] / n:.2f} / "
          f"sparse {path_totals['sparse'] / n:.2f} → fused {path_totals['fused'] / n:.2f}")
    print(f"  降级统计: {degraded_counter or '无'}")
    print(f"  按 origin: {json.dumps(breakdown('origin'), ensure_ascii=False)}")
    print(f"  按库:      {json.dumps(breakdown('collection'), ensure_ascii=False)}")
    if verdict:
        print(f"  门禁(R@5≥{args.gate}): {verdict}")
    if cmp_result:
        print("-" * 72)
        print(f"  配对比较 vs {cmp_result['prev_detail']}")
        print(f"  共同题目 {cmp_result['n_common']}｜"
              f"翻转 miss→hit {cmp_result['flips_up']} / hit→miss {cmp_result['flips_down']}")
        print(f"  R@5 共同题口径: {cmp_result['prev_r5_on_common']:.4f} → "
              f"{cmp_result['cur_r5_on_common']:.4f}（{cmp_result['delta']:+.4f}）"
              f"｜McNemar p={cmp_result['mcnemar']['p']:.4f}")
        print(f"  按库: {json.dumps(cmp_result['by_collection'], ensure_ascii=False)}")
        if delta_verdict:
            print(f"  门禁(delta≥{args.gate_delta * 100:+.1f}pp): {delta_verdict}")

    # ---------------- 入档 ----------------
    run_dir = RUNS_DIR / f"{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "label": args.label,
        "eval_set": str(eval_path),
        "n_queries": n,
        "retrieval_mode": "app_three_path",
        "top_k": args.top_k,
        "fused_limit": args.fused_limit,
        "thresholds": {"min_vector_score": MIN_VECTOR_SCORE,
                       "min_keyword_score": MIN_KEYWORD_SCORE,
                       "min_sparse_score": settings.min_sparse_score,
                       "sparse_recall_enabled": container.sparse is not None,
                       "sparse_collections": sorted(sparse_cols),
                       "path_head_collections": sorted(path_head_cols),
                       "rerank_enabled": container.rerank is not None},
        "ready_ports": container.ready_ports(),
        "pending_ports": container.pending,
        "r@5": round(r5, 4), "r@10": round(r10, 4), "mrr": round(mrr, 4),
        "path_avg_hits": {k: round(v / n, 3) for k, v in path_totals.items()},
        "degraded_counts": degraded_counter,
        "elapsed_ms_avg": {k: round(v / n, 1) for k, v in elapsed_sum.items()},
        "breakdown": {"origin": breakdown("origin"),
                      "collection": breakdown("collection")},
        "gate": {"threshold": args.gate, "verdict": verdict},
        "generated_at": datetime.now(timezone.utc)
        .astimezone().isoformat(timespec="seconds"),
    }
    if cmp_result:
        summary["compare"] = cmp_result
        summary["gate_delta"] = {"threshold": args.gate_delta,
                                 "delta": cmp_result["delta"],
                                 "verdict": delta_verdict}
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "detail.jsonl").open("w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  [OK] 入档 {run_dir}")

    container.close()
    return 0 if (verdict != "FAIL" and delta_verdict != "FAIL") else 1


if __name__ == "__main__":
    sys.exit(main())
