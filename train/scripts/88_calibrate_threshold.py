"""MIN_VECTOR_SCORE 阈值重校准 / 分模式校准（TDD §4.4；P0-3d-2）。

    python scripts/88_calibrate_threshold.py --retrieval dense
    python scripts/88_calibrate_threshold.py --retrieval sparse
    python scripts/88_calibrate_threshold.py --retrieval dense \
        --eval data/retrieval_eval_v1_remap.jsonl \
        --embed-model-dir ../../models/bge-m3-ft-v1

原理：
    线上 MIN_VECTOR_SCORE=0.50 是按 **dense 余弦相似度** 标定的。
    sparse 走 bge-m3 lexical_weights + Milvus IP，分数尺度完全不同（无上界、
    与查询词权重相关），直接沿用 0.50 会使拒答防线失效。
    本脚本按模式各跑一次，产出两套独立的建议阈值与分数分布。

做法：
    · 相关分布：每题 Top-N 里 gold 块的最高分数（= 模型认为最相关的正确答案）
    · 噪声分布：Top1 不是 gold 的题，其 Top1 分数（= 模型误以为相关的噪声）
    · Top1 分布：所有题的 Top1 分数（= 线上 MIN_VECTOR_SCORE 实际卡的那一个数）
    推荐阈值 = max(相关 P5, 噪声 P95)：
        保证 ≥95% 的正确答案不被拦（recall 侧），
        且 ≥95% 的噪声被拦下（精度侧）。

★ 输出只是**建议**：改 retrieve.py 的常量与注释依据必须人工确认
  （PRD §6 门禁：这一步不可跳过）。本次 P0-3d-2 明确**不改 retrieve.py**。
"""

from __future__ import annotations

import argparse
import json
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
PROJECT_ROOT = TRAIN_ROOT.parent

#: 线上当前值。dense 模式用 COSINE(0~1)，sparse 模式用 IP（无上界）。
ONLINE_MIN_VECTOR_SCORE = 0.50


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
    return xs[k]


def dist(xs: list[float]) -> dict:
    return {"n": len(xs),
            "p5": round(percentile(xs, 0.05), 3) if xs else None,
            "p10": round(percentile(xs, 0.10), 3) if xs else None,
            "p50": round(percentile(xs, 0.50), 3) if xs else None,
            "p90": round(percentile(xs, 0.90), 3) if xs else None,
            "p95": round(percentile(xs, 0.95), 3) if xs else None,
            "min": round(min(xs), 3) if xs else None,
            "max": round(max(xs), 3) if xs else None}


def main() -> int:
    parser = argparse.ArgumentParser(description="MIN_VECTOR_SCORE 分模式重校准")
    parser.add_argument("--eval", default=None)
    parser.add_argument("--retrieval", choices=("dense", "sparse"),
                        default="dense",
                        help="dense=COSINE 向量路；sparse=IP 稀疏路。"
                             "两种分数尺度不可比，必须分别标定")
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--embed-model-dir",
                        default=str(TRAIN_ROOT.parent / "models" / "bge-m3"))
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    eval_path = (Path(args.eval) if args.eval else
                 (DATA_DIR / "retrieval_eval_v1.jsonl"
                  if (DATA_DIR / "retrieval_eval_v1.jsonl").exists()
                  else DATA_DIR / "retrieval_eval_v0_provisional.jsonl"))
    queries = load_jsonl(eval_path)
    if not queries:
        raise SystemExit(f"[FAIL] 评测集为空：{eval_path}")

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import sentence_transformers as st
    from pymilvus import MilvusClient

    metric = "COSINE" if args.retrieval == "dense" else "IP"
    print("=" * 68)
    print(f"  MIN_VECTOR_SCORE 分模式校准  retrieval={args.retrieval} "
          f"metric={metric}")
    print(f"  模型: {args.embed_model_dir}")
    print(f"  评测集: {eval_path}（{len(queries)} 题）")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 68)

    client = MilvusClient(uri=args.milvus_uri)
    texts = [q["question"] for q in queries]

    vectors = None
    sparse_queries = None
    if args.retrieval == "dense":
        model = st.SentenceTransformer(args.embed_model_dir, device="cpu")
        model.max_seq_length = 4096
        vectors = model.encode(texts, batch_size=8,
                               normalize_embeddings=True,
                               show_progress_bar=False)
    else:
        sys.path.insert(0, str(PROJECT_ROOT / "08_数据入库" / "milvus"))
        from common import load_bge_m3_sparse, encode_sparse_vectors
        print(f"  加载 bge-m3 sparse（FlagEmbedding，CPU）")
        sparse_model = load_bge_m3_sparse(args.embed_model_dir)
        sparse_queries = encode_sparse_vectors(sparse_model, texts, batch_size=8)
        if any(not s for s in sparse_queries):
            raise SystemExit("[FAIL] 出现空稀疏查询向量，拒绝继续校准。")

    related: list[float] = []      # gold 命中的最高分数
    noise: list[float] = []        # Top1 非 gold 的分数
    top1_all: list[float] = []     # 所有题 Top1（阈值实际作用对象）
    for i, q in enumerate(queries):
        if args.retrieval == "sparse":
            hits = client.search(
                collection_name=q["collection"],
                data=[sparse_queries[i]],
                anns_field="sparse_vector",
                search_params={"metric_type": "IP"},
                limit=args.top_n, output_fields=["chunk_id"])[0]
        else:
            # P0-3 加了 sparse_vector 后 collection 有多个 ANN 字段，
            # 不指定 anns_field 会报 "multiple anns_fields exist"。
            # 这里显式指定密集向量字段（与 81 的 dense 分支一致）。
            hits = client.search(
                collection_name=q["collection"], data=[vectors[i].tolist()],
                anns_field="vector",
                limit=args.top_n, output_fields=["chunk_id"])[0]
        if not hits:
            continue
        top1 = float(hits[0]["distance"])
        top1_all.append(top1)
        gold = set(q["gold_chunk_ids"])
        if hits[0]["entity"].get("chunk_id", "") not in gold:
            noise.append(top1)
        sims = [float(h["distance"]) for h in hits
                if h["entity"].get("chunk_id", "") in gold]
        if sims:
            related.append(max(sims))

    if not related:
        raise SystemExit("[FAIL] 没有任何 gold 命中，无法校准（检索或 gold id 空间可能已坏）")

    rel, noi, t1 = dist(related), dist(noise), dist(top1_all)
    suggested = max(rel["p5"], noi["p95"]) if noise else rel["p5"]

    thr = ONLINE_MIN_VECTOR_SCORE
    n_top1_blocked = sum(1 for s in top1_all if s < thr)
    n_related_blocked = sum(1 for s in related if s < thr)
    n_noise_passed = sum(1 for s in noise if s >= thr)
    blocked = {
        "threshold": thr,
        "top1_n": len(top1_all),
        "top1_below_threshold": n_top1_blocked,
        "top1_blocked_pct": round(100 * n_top1_blocked / len(top1_all), 2)
        if top1_all else None,
        "related_below_threshold": n_related_blocked,
        "related_blocked_pct": round(100 * n_related_blocked / len(related), 2),
        "noise_at_or_above_threshold": n_noise_passed,
        "noise_passed_pct": round(100 * n_noise_passed / len(noise), 2)
        if noise else None,
    }

    def line(name: str, d: dict) -> str:
        return (f"  {name:<12} n={d['n']:<4} P10={d['p10']} "
                f"P50={d['p50']}  P90={d['p90']}  "
                f"[min={d['min']} max={d['max']}]")

    print(line("相关分布", rel))
    print(line("噪声分布", noi) if noise else "  噪声分布: 无样本（所有题 Top1 都命中 gold）")
    print(line("Top1 分布", t1))
    print("-" * 68)
    print(f"  建议阈值 = max(相关P5={rel['p5']}, 噪声P95={noi['p95']}) "
          f"= **{suggested:.3f}**（{metric} 尺度）")
    print(f"  若沿用线上 {thr:.2f}：Top1 被卡 {n_top1_blocked}/{len(top1_all)} "
          f"({blocked['top1_blocked_pct']}%)；"
          f"gold 最高分被卡 {n_related_blocked}/{len(related)} "
          f"({blocked['related_blocked_pct']}%)；"
          f"非 gold Top1 放行 {n_noise_passed}/{len(noise)} "
          f"({blocked['noise_passed_pct'] if noise else 'n/a'}%)")
    print(f"  当前线上值: MIN_VECTOR_SCORE（application/use_cases/retrieve.py:55）")
    print("  ★ 本脚本只出建议；人工确认后更新常量**与注释里的分布依据数字**，禁止直接上线。")

    out = DATA_DIR / (f"threshold_calibration_{args.retrieval}_"
                      f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    payload = {
        "retrieval": args.retrieval,
        "metric_type": metric,
        "score_scale": ("dense 余弦相似度，有界 [0,1]" if args.retrieval == "dense"
                        else "bge-m3 lexical_weights 稀疏内积 IP，无上界、随查询词权重变化"),
        "embed_model": args.embed_model_dir,
        "eval_set": str(eval_path),
        "n_queries": len(queries),
        "top_n": args.top_n,
        "related": rel,
        "noise": noi,
        "top1": t1,
        "suggested_min_vector_score": round(suggested, 3),
        "suggestion_formula": "max(相关P5, 噪声P95)",
        "if_keep_online_0_50": blocked,
        "scores_raw": {"top1": [round(s, 4) for s in top1_all],
                       "related": [round(s, 4) for s in related],
                       "noise": [round(s, 4) for s in noise]},
        "note": "仅为建议报告；retrieve.py 未改动，阈值结论待人工一起定。",
        "generated_at": datetime.now(timezone.utc).astimezone()
        .isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"  [OK] 入档 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
