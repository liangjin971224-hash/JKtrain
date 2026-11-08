"""漏检归因：逐题导出三路原始名次，判定"该怪融合 / 阈值 / 排序 / 召回"。

    python train/scripts/96_miss_attribution.py \
        --run train/eval_runs/app_path_head_v1_20261107_194150

────────────────────────────────────────────────────────────────────
回答什么问题
────────────────────────────────────────────────────────────────────
对基线 run 里每一道 miss（gold 不在 fused 前 5）：
    ① 在 dense / sparse / BM25 三条路的 **top-50 原始结果**里，gold 排第几？
    ② 如果这一路按**线上阈值**剪枝，gold 还活着吗（kept）？
    ③ 反推下一步该做什么：

| 原始名次 | 阈值后 | 归因 | 下一步 |
|---|---|---|---|
| 1–5 | 活着 | 融合没用上 | 改窗口/权重（P1 已做一轮） |
| 1–5 | 被滤掉 | 阈值误杀 | 阈值分模式校准 |
| 6–20 | 任意 | 单路排序不够 | 微调/rerank 的可及范围 |
| >20 或找不到 | — | 召回不到 | 语料/切片/模型召回问题 |

★ 与 92 号的分工：92 号测线上融合后的结果；96 号只看**单路原始能力**，
  所以它跑的是 `container.vector/keyword/sparse` 的裸 search，
  不经 `RetrieveUseCase` 的阈值与窗口。

★ 只读脚本：不写库、不改文档，产物落在 train/eval_runs/ 下。
"""
from __future__ import annotations

import argparse
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

DEFAULT_REMAP = DATA_DIR / "retrieval_eval_v1_remap.jsonl"

#: 生产阈值（与 92 号 / RetrieveUseCase 默认一致）
MIN_VECTOR_SCORE = 0.50
MIN_KEYWORD_SCORE = 14.0
MIN_KEYWORD_RELATIVE = 0.35
MIN_SPARSE_SCORE = 0.02

DEFAULT_TOP_K = 50


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def first_gold_rank(hits, gold: set[str]) -> tuple[int | None, float | None, str]:
    """返回 (名次, 分数, chunk_id)；名次从 1 开始，找不到为 None。"""
    for rank, hit in enumerate(hits, 1):
        cid = hit.chunk.chunk_id if hit.chunk is not None else ""
        if cid in gold:
            return rank, float(hit.score), cid
    return None, None, ""


def path_record(name: str, hits, gold: set[str], *, kept) -> dict:
    """一路的诊断记录。

    Args:
        hits: 原始 top-50（未剪枝）。
        gold: gold chunk_id 集合。
        kept: 判定单条命中是否会被线上阈值保留的函数。
    """
    rank, score, cid = first_gold_rank(hits, gold)
    kept_hits = [h for h in hits if kept(h)]
    kept_rank, kept_score, _ = first_gold_rank(kept_hits, gold)
    return {
        "path": name,
        "rank": rank,
        "score": round(score, 4) if score is not None else None,
        "gold_chunk_id": cid,
        "kept_rank": kept_rank,
        "kept": kept_rank is not None,
        "kept_score": round(kept_score, 4) if kept_score is not None else None,
        "n_hits": len(hits),
    }


def bucket_of(rank: int | None) -> str:
    if rank is None:
        return "not_recalled"
    if rank <= 5:
        return "1-5"
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    return ">20"


def main() -> int:
    ap = argparse.ArgumentParser(description="漏检归因（三路原始名次）")
    ap.add_argument("--run", required=True,
                    help="基线 run 目录或 detail.jsonl（如 app_path_head_v1_*）")
    ap.add_argument("--remap", default=None,
                    help="评测集（默认 retrieval_eval_v1_remap.jsonl）")
    ap.add_argument("--label", default="miss_attribution")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args()

    run_path = Path(args.run)
    if run_path.is_dir():
        run_path = run_path / "detail.jsonl"
    if not run_path.exists():
        raise SystemExit(f"[FAIL] 找不到 detail：{run_path}")

    remap_path = Path(args.remap) if args.remap else DEFAULT_REMAP
    dataset = {r["qid"]: r for r in load_jsonl(remap_path)}
    detail = load_jsonl(run_path)

    misses = [d for d in detail if not d.get("hit@5")]
    if not misses:
        print("[OK] 基线没有漏检，无需归因。")
        return 0

    sys.path.insert(0, str(PROJECT_CODE))
    from config.assembly import build_container
    from config.settings import Settings

    print("=" * 72)
    print(f"  漏检归因 label={args.label}")
    print(f"  基线: {run_path}")
    print(f"  miss: {len(misses)}/{len(detail)}")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 72)

    settings = Settings()
    t0 = time.perf_counter()
    container = build_container(settings)
    print(f"  容器就绪 {time.perf_counter() - t0:.1f}s："
          f"{container.ready_ports()}；pending={list(container.pending) or '无'}")
    if container.vector is None or container.keyword is None:
        raise SystemExit("[FAIL] 向量或关键词端口不可用")

    sparse = getattr(container, "sparse", None)
    uses_sparse = sparse is not None

    rows: list[dict] = []
    for i, miss in enumerate(misses, 1):
        qid = miss["qid"]
        item = dataset.get(qid)
        if item is None:
            print(f"  ⚠️ {qid} 不在评测集里，跳过")
            continue
        question = item["question"]
        collection = item.get("collection") or ""
        perms = item.get("allowed_permissions")
        perms = tuple(perms) if perms else None
        gold = set(item.get("gold_chunk_ids") or ())

        # ---- dense（原始 top-50）----
        dense_hits = container.vector.search(
            question, collection=collection, top_k=args.top_k,
            allowed_permissions=perms)
        dense = path_record(
            "dense", dense_hits, gold,
            kept=lambda h: h.score >= MIN_VECTOR_SCORE)

        # ---- sparse（原始 top-50；只对启用稀疏的库有意义，但全库可查）----
        if uses_sparse:
            sparse_hits = sparse.search_sparse(
                question, collection=collection, top_k=args.top_k,
                allowed_permissions=perms)
            sparse_rec = path_record(
                "sparse", sparse_hits, gold,
                kept=lambda h: h.score >= MIN_SPARSE_SCORE)
        else:
            sparse_rec = {"path": "sparse", "rank": None, "score": None,
                          "gold_chunk_id": "", "kept_rank": None,
                          "kept": False, "kept_score": None, "n_hits": 0}

        # ---- BM25（原始 top-50；阈值是「Top1 绝对分 + 相对比例」）----
        kw_hits = container.keyword.search(
            question, top_k=args.top_k, allowed_permissions=perms)
        kw_top1 = kw_hits[0].score if kw_hits else 0.0
        kw_usable = kw_top1 >= MIN_KEYWORD_SCORE
        kw_floor = kw_top1 * MIN_KEYWORD_RELATIVE
        keyword = path_record(
            "keyword", kw_hits, gold,
            kept=lambda h: kw_usable and h.score >= kw_floor)

        paths = [dense, sparse_rec, keyword]
        raw_best = min((p["rank"] for p in paths if p["rank"] is not None),
                       default=None)
        kept_best = min((p["kept_rank"] for p in paths if p["kept_rank"] is not None),
                        default=None)

        # 归因：优先看「原始名次最近的那条路」是否被阈值挡掉
        if raw_best is None:
            cause = "not_recalled"
        elif raw_best <= 5 and kept_best is None:
            cause = "threshold_blocked"
        elif raw_best <= 5:
            cause = "fusion_missed"
        elif raw_best <= 20:
            cause = "ranking"
        else:
            cause = "far"

        rows.append({
            "qid": qid,
            "origin": miss.get("origin"),
            "collection": collection,
            "question": question,
            "gold_chunk_ids": sorted(gold),
            "fused_rank": miss.get("gold_rank"),
            "paths": {p["path"]: p for p in paths},
            "raw_best_rank": raw_best,
            "kept_best_rank": kept_best,
            "bucket": bucket_of(raw_best),
            "cause": cause,
        })

        rank_txt = "/".join(
            f"{p['path']}:{p['rank'] if p['rank'] is not None else '-'}"
            for p in paths)
        print(f"  {i:2d}/{len(misses)} {qid:28s} {collection:22s} "
              f"raw={raw_best or '-':>3} kept={kept_best or '-':>3} "
              f"({rank_txt}) → {cause}")

    # ---------------- 汇总 ----------------
    def count(key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            out[r[key]] = out.get(r[key], 0) + 1
        return dict(sorted(out.items()))

    by_bucket = count("bucket")
    by_cause = count("cause")
    by_collection: dict[str, dict[str, int]] = {}
    for r in rows:
        coll = by_collection.setdefault(r["collection"], {})
        coll[r["cause"]] = coll.get(r["cause"], 0) + 1

    print("\n" + "=" * 72)
    print("  汇总")
    print("=" * 72)
    print(f"  miss = {len(rows)}")
    print(f"  原始名次分桶: {by_bucket}")
    print(f"  归因: {by_cause}")
    for coll, causes in by_collection.items():
        print(f"    {coll}: {dict(sorted(causes.items()))}")

    # 给出「微调可及范围」的机械判读
    ranking = by_cause.get("ranking", 0)
    fusion = by_cause.get("fusion_missed", 0)
    threshold = by_cause.get("threshold_blocked", 0)
    print("\n  判读：")
    if fusion:
        print(f"    · {fusion} 道「单路前 5 但融合没用上」→ 先改融合，不是微调")
    if threshold:
        print(f"    · {threshold} 道「单路前 5 但被阈值滤掉」→ 阈值需分模式校准")
    print(f"    · {ranking} 道落在单路 6–20 名 → 微调/rerank 的可及范围（乐观上限）")
    print(f"    · {by_cause.get('far', 0)} 道 >20 名、"
          f"{by_cause.get('not_recalled', 0)} 道 top-50 都没有 → 召回问题")

    run_dir = RUNS_DIR / f"{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "detail.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "label": args.label,
        "baseline_run": str(run_path),
        "n_misses": len(rows),
        "buckets": by_bucket,
        "causes": by_cause,
        "by_collection": by_collection,
        "top_k": args.top_k,
        "thresholds": {
            "min_vector_score": MIN_VECTOR_SCORE,
            "min_keyword_score": MIN_KEYWORD_SCORE,
            "min_keyword_relative": MIN_KEYWORD_RELATIVE,
            "min_sparse_score": MIN_SPARSE_SCORE,
        },
        "generated_at": datetime.now(timezone.utc)
        .astimezone().isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  [OK] 入档 {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
