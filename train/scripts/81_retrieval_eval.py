"""检索级评测执行器：R@5 / R@10 / MRR + 门禁判定。

    python scripts/81_retrieval_eval.py --label baseline_v0
    python scripts/81_retrieval_eval.py --label baseline_v0 --gate 0.0
    python scripts/81_retrieval_eval.py --label rerank_on --rerank-model-dir ../../models/bge-reranker-v2-m3

    # 与上一次运行做配对比较（McNemar）
    python scripts/81_retrieval_eval.py --label hybrid \
        --eval data/retrieval_eval_v1_remap.jsonl \
        --compare eval_runs/p03_dense_20261107_120000/detail.jsonl --gate-delta 0.03

    # P0-3d-1：案例库 dense top5 ∪ sparse top5 → ≤10 候选窗口（★ 口径已变）
    python scripts/81_retrieval_eval.py --label p03_union \
        --eval data/retrieval_eval_v1_remap.jsonl --retrieval union \
        --compare eval_runs/p03_dense_20261107_130202/detail.jsonl

输入：--eval 指定的评测集（默认 retrieval_eval_v1.jsonl，缺失时回退 v0 临时集）。

输出：train/eval_runs/<label>_<时间戳>/
    summary.json   指标 + 参数 + 环境信息（门禁结论在这里）
    detail.jsonl   每题的命中明细（排障用）

★ 评测口径：
  - 每条查询只在其所属 collection 内检索（分库报告，避免跨库混排干扰归因）；
    collection 内权限级别单一，故不再拼权限 filter（与线上行为不冲突，见 TDD §4.5）。
  - 向量路与线上同一配置：bge-m3、归一化、max_seq_length=4096。
  - --rerank-model-dir 给出时，对每题 Top-20 候选做 CrossEncoder 重排后再算指标
    （1.7 收益评估就是同一评测集跑两遍对比 summary）。

★ 为什么要配对比较（--compare）：
  55 题口径下 R@5 的标准误约 6pp，绝对率差 3~5pp 落在噪声里判不出来；
  同一批题的前后翻转（miss→hit vs hit→miss）方差小得多，用 McNemar 能判出显著。
  详见 docs/实施方案_P0检索质量三件套_2026-11-07.md §2.6。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = TRAIN_ROOT / "data"
RUNS_DIR = TRAIN_ROOT / "eval_runs"

DEFAULT_EVAL = DATA_DIR / "retrieval_eval_v1.jsonl"
PROVISIONAL_EVAL = DATA_DIR / "retrieval_eval_v0_provisional.jsonl"

TOP_K_SEARCH = 20          # 检索深度（≥rerank候选数与R@10窗口）
RERANK_CANDIDATE_K = 20

# P0-3c：按库路由表（写死为常量，便于归档复现与审计）。
# 依据：205 全量实测 案例库 sparse 25/34 > dense/hybrid 21/34；
#       training_docs hybrid 105/131 > dense 98/131 > sparse 94/131；
#       question_bank 三模式均 40/40，取 dense 保持默认行为。
# 注意：这里的 hybrid 仍是 unweighted RRFRanker(k=60)，未调 k、未用 WeightedRanker。
ROUTE_MAP = {
    "health_case_library": "sparse",
    "health_training_docs": "hybrid",
    "health_question_bank": "dense",
}

# P0-3d-1：union 模式常量（只对案例库生效，其余库沿用 ROUTE_MAP）。
# 案例库 holdout 上 sparse 5/9 < dense 6/9，方向未确认，故不定稿 case=sparse；
# 改为把两路 Top-5 合并成一个 ≤10 的候选窗口，以「窗口覆盖」作为命中判据。
UNION_CASE_COLLECTION = "health_case_library"
UNION_PER_PATH_TOPK = 5        # 每路取前 5
CASE_WINDOW = 10               # 合并去重后的候选窗口上限（5+5）


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics(ranked_ids: list[str], golds: list[str]) -> dict:
    """对单个查询算 R@5 / R@10 / MRR（MRR 用首个 gold 名次，全深度）。"""
    gold_set = set(golds)
    hit5 = any(cid in gold_set for cid in ranked_ids[:5])
    hit10 = any(cid in gold_set for cid in ranked_ids[:10])
    rr = 0.0
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in gold_set:
            rr = 1.0 / rank
            break
    return {"hit@5": hit5, "hit@10": hit10, "rr": rr}


def merge_union_window(dense_ids: list[str], sparse_ids: list[str]) -> list[str]:
    """union 合并：两路各取 Top-5，按名次轮转交错 d1,s1,d2,s2,… 后去重。

    ★ 不做跨路分数排序：dense 是 COSINE（0~1）、sparse 是 IP（无上界），
      两者尺度不可比，按分数混排没有意义。交错顺序只影响 gold_rank / MRR，
      不影响「gold 是否在窗口内」这一命中判据。
    """
    out: list[str] = []
    seen: set[str] = set()
    for k in range(UNION_PER_PATH_TOPK):
        for ids in (dense_ids, sparse_ids):
            if k >= len(ids):
                continue
            cid = ids[k]
            if cid and cid not in seen:
                seen.add(cid)
                out.append(cid)
    return out[:CASE_WINDOW]


def compute_window_metrics(window_ids: list[str], golds: list[str]) -> dict:
    """union 案例库口径：gold 是否落在合并候选窗口（≤10 条）内。

    ★ 这不是单列表 R@5。hit@5 在此处表示「10 候选窗口覆盖」，
      hit@10 与 hit@5 同值；rr 用 gold 在窗口内的名次（1~10）计算。
    """
    gold_set = set(golds)
    rank = next((r for r, cid in enumerate(window_ids, start=1)
                 if cid in gold_set), None)
    hit = rank is not None
    return {"hit@5": hit, "hit@10": hit,
            "rr": 1.0 / rank if rank else 0.0}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def resolve_detail_path(raw: str) -> Path:
    """--compare 允许给 detail.jsonl，也允许直接给 eval_runs 目录。"""
    path = Path(raw)
    if path.is_dir():
        return path / "detail.jsonl"
    return path


def mcnemar_p(b: int, c: int) -> tuple[float, str]:
    """McNemar 检验。b = miss→hit，c = hit→miss。

    b+c < 25 用精确二项检验（避免卡方在小样本下失真），否则用连续性校正卡方。
    """
    n = b + c
    if n == 0:
        return 1.0, "无翻转"
    if n < 25:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        return min(1.0, 2 * tail), "exact"
    diff = abs(b - c) - 1
    if diff <= 0:
        return 1.0, "chi2<=0"
    chi2 = diff * diff / n
    # df=1 的卡方生存函数：P(X>x) = erfc(sqrt(x/2))
    return min(1.0, math.erfc(math.sqrt(chi2 / 2.0))), f"chi2={chi2:.2f}"


def compare_with(prev_path: Path, details: list[dict]) -> dict:
    """与上一次运行逐题配对比较，输出翻转矩阵 + McNemar。"""
    prev_rows = load_jsonl(prev_path)
    prev = {d["qid"]: d for d in prev_rows}
    cur = {d["qid"]: d for d in details}

    common = sorted(set(prev) & set(cur))
    if not common:
        raise SystemExit(f"[FAIL] 与 {prev_path} 没有共同 qid，无法配对比较。")

    up = [q for q in common if not prev[q]["hit@5"] and cur[q]["hit@5"]]
    down = [q for q in common if prev[q]["hit@5"] and not cur[q]["hit@5"]]
    prev_r5 = mean([1.0 if prev[q]["hit@5"] else 0.0 for q in common])
    cur_r5 = mean([1.0 if cur[q]["hit@5"] else 0.0 for q in common])
    p, method = mcnemar_p(len(up), len(down))

    def cur_r5_of(qids: list[str]) -> float:
        return mean([1.0 if cur[q]["hit@5"] else 0.0 for q in qids])

    def prev_r5_of(qids: list[str]) -> float:
        return mean([1.0 if prev[q]["hit@5"] else 0.0 for q in qids])

    up_set, down_set = set(up), set(down)

    def group_flips(key: str) -> dict:
        groups: dict[str, list[str]] = {}
        for q in common:
            groups.setdefault(cur[q][key], []).append(q)
        return {
            g: {
                "n": len(qids),
                "delta": round(cur_r5_of(qids) - prev_r5_of(qids), 4),
                "up": sum(1 for q in qids if q in up_set),
                "down": sum(1 for q in qids if q in down_set),
            }
            for g, qids in sorted(groups.items())
        }

    result = {
        "prev_detail": str(prev_path),
        "n_prev": len(prev_rows),
        "n_cur": len(details),
        "n_common": len(common),
        "only_in_prev": len(set(prev) - set(cur)),
        "only_in_cur": len(set(cur) - set(prev)),
        "prev_r5_on_common": round(prev_r5, 4),
        "cur_r5_on_common": round(cur_r5, 4),
        "delta": round(cur_r5 - prev_r5, 4),
        "flips_up": len(up), "flips_down": len(down),
        "up_qids": up, "down_qids": down,
        "mcnemar": {"p": round(p, 4), "method": method},
        "by_origin": group_flips("origin"),
        "by_collection": group_flips("collection"),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="检索级评测（R@5/R@10/MRR）")
    parser.add_argument("--eval", default=None, help="评测集路径（默认自动回退）")
    parser.add_argument("--label", default="run", help="本次评测标签（入 eval_runs 目录名）")
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--embed-model-dir",
                        default=str(TRAIN_ROOT.parent / "models" / "bge-m3"))
    parser.add_argument("--rerank-model-dir", default=None,
                        help="给出则启用 CrossEncoder 重排（收益评估用）")
    parser.add_argument("--gate", type=float, default=None,
                        help="R@5 门禁阈值，如 0.60；给出则输出 PASS/FAIL")
    parser.add_argument("--compare", default=None,
                        help="上一次运行的 detail.jsonl（或 eval_runs 目录）；"
                             "给出则做逐题配对比较 + McNemar")
    parser.add_argument("--gate-delta", type=float, default=None,
                        help="配对比较的 R@5 增量门禁，如 0.03 表示 +3pp；需配合 --compare")
    parser.add_argument("--retrieval",
                        choices=("dense", "sparse", "hybrid", "route", "union"),
                        default="dense",
                        help="检索模式（默认 dense，保持历史基线可比）；"
                             "hybrid 用 hybrid_search + AnnSearchRequest×2 + RRFRanker(k=60)；"
                             "route 按 ROUTE_MAP 按库分派；"
                             "union 仅案例库生效（dense top5 ∪ sparse top5 → ≤10 候选窗口），"
                             "其余库与 route 一致。★ union 改变了案例库指标语义，"
                             "不可与历史 R@5 同口径比较")
    args = parser.parse_args()

    if args.gate_delta is not None and not args.compare:
        raise SystemExit("[FAIL] --gate-delta 需要同时给 --compare。")
    if args.compare and not resolve_detail_path(args.compare).exists():
        raise SystemExit(f"[FAIL] 找不到对比基线：{args.compare}")
    if args.retrieval in ("route", "union"):
        effective_map = (dict(ROUTE_MAP, **{UNION_CASE_COLLECTION: "union"})
                         if args.retrieval == "union" else ROUTE_MAP)
        print("  route_map: " + json.dumps(effective_map, ensure_ascii=False))
    if args.retrieval == "union":
        print(f"  ★ union 口径变更：案例库 {UNION_CASE_COLLECTION} 的 hit@5 "
              f"= dense top{UNION_PER_PATH_TOPK} ∪ sparse top{UNION_PER_PATH_TOPK} "
              f"（≤{CASE_WINDOW} 候选）覆盖，不是单列表 R@5；"
              "禁止与历史 R@5 同口径比较")

    eval_path = Path(args.eval) if args.eval else (
        DEFAULT_EVAL if DEFAULT_EVAL.exists() else PROVISIONAL_EVAL)
    if not eval_path.exists():
        raise SystemExit("[FAIL] 找不到评测集。先跑 85_merge_eval_set.py。")
    queries = load_jsonl(eval_path)
    if not queries:
        raise SystemExit(f"[FAIL] 评测集为空：{eval_path}")

    import sentence_transformers as st
    from pymilvus import MilvusClient

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    reranker = None
    if args.rerank_model_dir:
        from sentence_transformers import CrossEncoder
        print(f"  加载 reranker（CPU）: {args.rerank_model_dir}")
        reranker = CrossEncoder(args.rerank_model_dir, max_length=512,
                                device="cpu")

    print("=" * 68)
    print(f"  检索评测 label={args.label}"
          f"{' +rerank' if reranker else ''}"
          f"  retrieval={args.retrieval}")
    print(f"  评测集: {eval_path}（{len(queries)} 题）")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 68)

    print(f"  加载 bge-m3 dense（CPU）: {args.embed_model_dir}")
    t0 = time.perf_counter()
    model = st.SentenceTransformer(args.embed_model_dir, device="cpu")
    model.max_seq_length = 4096
    print(f"  模型就绪 {time.perf_counter() - t0:.1f}s")

    texts = [q["question"] for q in queries]
    t0 = time.perf_counter()
    vectors = model.encode(texts, batch_size=8, normalize_embeddings=True,
                           show_progress_bar=False)
    embed_ms = (time.perf_counter() - t0) * 1000 / len(queries)
    print(f"  dense 编码完成，均值 {embed_ms:.0f} ms/题")

    # P0-3：sparse / hybrid 需要查询侧稀疏向量。与入库共用 common.py 的实现，
    # 保证 token id、权重、max_length 完全一致。dense 模式不加载，行为不变。
    sparse_queries = None
    if args.retrieval in ("sparse", "hybrid", "route", "union"):
        sys.path.insert(0, str(TRAIN_ROOT.parent / "08_数据入库" / "milvus"))
        from common import load_bge_m3_sparse, encode_sparse_vectors
        print(f"  加载 bge-m3 sparse（FlagEmbedding，CPU）")
        sparse_model = load_bge_m3_sparse(args.embed_model_dir)
        t0 = time.perf_counter()
        sparse_queries = encode_sparse_vectors(sparse_model, texts, batch_size=8)
        if any(not s for s in sparse_queries):
            raise SystemExit("[FAIL] 出现空稀疏查询向量，拒绝继续评测。")
        print(f"  sparse 编码完成，均值 {(time.perf_counter()-t0)*1000/len(queries):.0f} ms/题"
              f" | 平均 {sum(len(s) for s in sparse_queries)/len(sparse_queries):.0f} token")

    client = MilvusClient(uri=args.milvus_uri)

    from pymilvus import AnnSearchRequest, RRFRanker

    # route / union 模式下按 collection 逐题分派；未覆盖的库直接报错，禁止静默兜底
    if args.retrieval in ("route", "union"):
        missing = sorted({q["collection"] for q in queries} - set(ROUTE_MAP))
        if missing:
            raise SystemExit(f"[FAIL] route_map 未覆盖这些 collection：{missing}")

    details = []
    search_ms_total, rerank_ms_total = 0.0, 0.0
    for i, (q, vec) in enumerate(zip(queries, vectors)):
        # union 只对案例库生效；其余库沿用 ROUTE_MAP（与 route 完全一致）
        union_case = (args.retrieval == "union"
                      and q["collection"] == UNION_CASE_COLLECTION)
        if args.retrieval in ("route", "union"):
            mode = "union" if union_case else ROUTE_MAP[q["collection"]]
        else:
            mode = args.retrieval
        dense_ids_top5: list[str] = []
        sparse_ids_top5: list[str] = []
        t0 = time.perf_counter()
        if union_case:
            # 两路各按原有 TOP_K_SEARCH 深度检索，再各取前 5；
            # 不改 TOP_K_SEARCH 常量，也不重建库 / 改 gold。
            dense_hits = client.search(
                collection_name=q["collection"],
                data=[vec.tolist()],
                anns_field="vector",
                limit=TOP_K_SEARCH,
                output_fields=["chunk_id", "text"],
            )[0]
            sparse_hits = client.search(
                collection_name=q["collection"],
                data=[sparse_queries[i]],
                anns_field="sparse_vector",
                search_params={"metric_type": "IP"},
                limit=TOP_K_SEARCH,
                output_fields=["chunk_id", "text"],
            )[0]
            dense_ids_top5 = [h["entity"].get("chunk_id", "")
                              for h in dense_hits][:UNION_PER_PATH_TOPK]
            sparse_ids_top5 = [h["entity"].get("chunk_id", "")
                               for h in sparse_hits][:UNION_PER_PATH_TOPK]
            hits = dense_hits + sparse_hits   # 供 reranker 取 text（如有）
            union_ranked = merge_union_window(dense_ids_top5, sparse_ids_top5)
        elif mode == "dense":
            hits = client.search(
                collection_name=q["collection"],
                data=[vec.tolist()],
                anns_field="vector",
                limit=TOP_K_SEARCH,
                output_fields=["chunk_id", "text"],
            )[0]
        elif mode == "sparse":
            hits = client.search(
                collection_name=q["collection"],
                data=[sparse_queries[i]],
                anns_field="sparse_vector",
                search_params={"metric_type": "IP"},
                limit=TOP_K_SEARCH,
                output_fields=["chunk_id", "text"],
            )[0]
        else:  # hybrid
            dense_req = AnnSearchRequest(
                data=[vec.tolist()], anns_field="vector",
                param={"metric_type": "COSINE"}, limit=TOP_K_SEARCH)
            sparse_req = AnnSearchRequest(
                data=[sparse_queries[i]], anns_field="sparse_vector",
                param={"metric_type": "IP"}, limit=TOP_K_SEARCH)
            hits = client.hybrid_search(
                collection_name=q["collection"],
                reqs=[dense_req, sparse_req],
                ranker=RRFRanker(k=60),
                limit=TOP_K_SEARCH,
                output_fields=["chunk_id", "text"],
            )[0]
        search_ms_total += (time.perf_counter() - t0) * 1000

        ranked = (union_ranked if union_case else
                  [h["entity"].get("chunk_id", "") for h in hits])

        if reranker is not None:
            cands = ranked[:RERANK_CANDIDATE_K]
            rest = ranked[RERANK_CANDIDATE_K:]
            texts_by_id = {h["entity"].get("chunk_id", ""):
                           h["entity"].get("text", "") for h in hits}
            t0 = time.perf_counter()
            scores = reranker.predict(
                [(q["question"], texts_by_id.get(cid, "")) for cid in cands],
                batch_size=16, show_progress_bar=False)
            rerank_ms_total += (time.perf_counter() - t0) * 1000
            order = sorted(range(len(cands)), key=lambda i: -float(scores[i]))
            # 重排只作用于候选窗口内；窗口外保留原序兜底
            ranked = [cands[i] for i in order] + rest

        # union 案例库：命中 = gold 落在 ≤10 的合并候选窗口内（非单列表 R@5）
        m = (compute_window_metrics(ranked, q["gold_chunk_ids"]) if union_case
             else compute_metrics(ranked, q["gold_chunk_ids"]))
        # P0-3 放行 3：gold 在候选里的首个名次；非 union 为 1~20，
        # union 案例库为 1~10（合并窗口内），未命中 null。
        gold_set = set(q["gold_chunk_ids"])
        gold_rank = next((r for r, cid in enumerate(ranked, start=1)
                          if cid in gold_set), None)
        row = {**m, "qid": q["qid"], "origin": q["origin"],
               "collection": q["collection"],
               "retrieval_mode": mode,
               "gold_rank": gold_rank,
               "ranked_head": ranked[:10]}
        if union_case:
            row["case_window"] = len(ranked)
            row["case_dense_top5"] = dense_ids_top5
            row["case_sparse_top5"] = sparse_ids_top5
        details.append(row)

    r5 = mean([d["hit@5"] for d in details])
    r10 = mean([d["hit@10"] for d in details])
    mrr = mean([d["rr"] for d in details])
    search_ms = search_ms_total / len(queries)
    rerank_ms = rerank_ms_total / len(queries) if reranker else None

    # 分维度报告
    def breakdown(key: str) -> dict:
        groups: dict[str, list[dict]] = {}
        for d in details:
            groups.setdefault(d[key], []).append(d)
        return {g: {"n": len(v), "r@5": round(mean([x["hit@5"] for x in v]), 4),
                    "mrr": round(mean([x["rr"] for x in v]), 4)}
                for g, v in sorted(groups.items())}

    def hit_counts(key: str) -> dict:
        groups: dict[str, list[dict]] = {}
        for d in details:
            groups.setdefault(d[key], []).append(d)
        return {g: {"n": len(v), "hits": sum(1 for x in v if x["hit@5"])}
                for g, v in sorted(groups.items())}

    union_semantics = None
    if args.retrieval == "union":
        union_semantics = {
            "changed": True,
            "case_collection": UNION_CASE_COLLECTION,
            "case_library_hit@5": (
                f"gold 是否落在 dense top{UNION_PER_PATH_TOPK} ∪ "
                f"sparse top{UNION_PER_PATH_TOPK} 去重后的候选窗口"
                f"（≤{CASE_WINDOW} 条）内；这是 10 候选窗口覆盖率，"
                "不是单列表前 5 的 R@5，hit@10 与 hit@5 同值。"),
            "gold_rank": "gold 在合并窗口内的名次（1~10），未命中 null；MRR 用该名次计算。",
            "merge_order": (
                "按名次轮转交错 d1,s1,d2,s2,… 后去重；dense(COSINE) 与 sparse(IP) "
                "分数尺度不可比，不做跨路分数排序。两路仍按 TOP_K_SEARCH=20 检索后各取前 5。"),
            "other_collections": (
                "health_training_docs=hybrid、health_question_bank=dense，"
                "与 p03_route 一致，仍为单列表 R@5。"),
            "comparability": (
                "★ 口径已变更，禁止把本 run 的案例库命中数与整体 R@5 "
                "与 p03_dense / p03_sparse / p03_hybrid / p03_route 等历史 R@5 "
                "放在同一口径里比较；本次整体 R@5 是混合口径，仅作归档记录，"
                "不得作为同口径门禁或 delta 结论的依据。"),
        }

    cmp_result = None
    delta_verdict = None
    if args.compare:
        cmp_result = compare_with(resolve_detail_path(args.compare), details)
        if args.gate_delta is not None:
            delta_verdict = "PASS" if cmp_result["delta"] >= args.gate_delta else "FAIL"

    verdict = None
    if args.gate is not None:
        verdict = "PASS" if r5 >= args.gate else "FAIL"

    print("-" * 68)
    print(f"  R@5  = {r5:.4f}")
    print(f"  R@10 = {r10:.4f}")
    print(f"  MRR  = {mrr:.4f}")
    print(f"  延迟均值: embed {embed_ms:.0f}ms / search {search_ms:.0f}ms"
          + (f" / rerank {rerank_ms:.0f}ms" if rerank_ms is not None else ""))
    print(f"  按 origin: {json.dumps(breakdown('origin'), ensure_ascii=False)}")
    print(f"  按库:      {json.dumps(breakdown('collection'), ensure_ascii=False)}")
    print(f"  命中数:    {json.dumps(hit_counts('collection'), ensure_ascii=False)}"
          f" | 合计 {sum(1 for d in details if d['hit@5'])}/{len(details)}")
    if verdict:
        print(f"  门禁(R@5≥{args.gate}): {verdict}")

    if cmp_result:
        print("-" * 68)
        print(f"  配对比较 vs {cmp_result['prev_detail']}")
        print(f"  共同题目 {cmp_result['n_common']}"
              f"（仅旧 {cmp_result['only_in_prev']} / 仅新 {cmp_result['only_in_cur']}）")
        if cmp_result["n_common"] < 0.9 * len(details):
            print(f"  ⚠️ 共同题目不足当前的 90%，配对结论代表性有限")
        print(f"  翻转: miss→hit {cmp_result['flips_up']}  |  "
              f"hit→miss {cmp_result['flips_down']}")
        print(f"  R@5（共同题口径）: {cmp_result['prev_r5_on_common']:.4f} → "
              f"{cmp_result['cur_r5_on_common']:.4f} "
              f"({cmp_result['delta']:+.4f} = {cmp_result['delta'] * 100:+.2f}pp）")
        mc = cmp_result["mcnemar"]
        print(f"  McNemar p={mc['p']:.4f}（{mc['method']}）"
              + ("  → 显著" if mc["p"] < 0.05 else "  → 不显著"))
        print(f"  按 origin: {json.dumps(cmp_result['by_origin'], ensure_ascii=False)}")
        print(f"  按库:      {json.dumps(cmp_result['by_collection'], ensure_ascii=False)}")
        for tag, ids in (("改善", cmp_result["up_qids"]),
                         ("恶化", cmp_result["down_qids"])):
            if ids:
                print(f"  {tag}题（前 10/{len(ids)}）: {', '.join(ids[:10])}")
        if delta_verdict:
            print(f"  门禁(delta≥{args.gate_delta * 100:+.1f}pp): {delta_verdict}")

    # 入档
    run_dir = RUNS_DIR / f"{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "label": args.label,
        "eval_set": str(eval_path),
        "n_queries": len(queries),
        "retrieval_mode": args.retrieval,
        "case_window": CASE_WINDOW if args.retrieval == "union" else None,
        "route_map": (dict(ROUTE_MAP, **{UNION_CASE_COLLECTION: "union"})
                      if args.retrieval == "union" else
                      ROUTE_MAP if args.retrieval == "route" else None),
        "metric_semantics": union_semantics,
        "hit_counts": {"total": sum(1 for d in details if d["hit@5"]),
                       "by_collection": hit_counts("collection")},
        "embed_model": args.embed_model_dir,
        "rerank_model": args.rerank_model_dir,
        "r@5": round(r5, 4), "r@10": round(r10, 4), "mrr": round(mrr, 4),
        "latency_ms": {"embed": round(embed_ms), "search": round(search_ms),
                       "rerank": round(rerank_ms) if rerank_ms is not None else None},
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
        if args.retrieval == "union":
            summary["compare_caveat"] = (
                "★ 逐题翻转 / McNemar 仍按 hit@5 字段计算，但案例库那部分口径已改为 "
                "≤10 候选窗口覆盖；本次 delta 与 p 值不得作为与历史 R@5 "
                "同口径的显著性结论，仅供逐题归因排障。")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "detail.jsonl").open("w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  [OK] 入档 {run_dir}")

    return 0 if (verdict != "FAIL" and delta_verdict != "FAIL") else 1


if __name__ == "__main__":
    sys.exit(main())
