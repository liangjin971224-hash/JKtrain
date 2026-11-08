"""窗口构图离线模拟：用三路原始候选重放不同「头部配额」，找是否还有静态空间。

    python train/scripts/97_window_sim.py --collection health_training_docs

────────────────────────────────────────────────────────────────────
为什么需要
────────────────────────────────────────────────────────────────────
96 号归因显示：177 基线里仍有 12/28 道 miss 是「gold 在单路 top-5，
但融合窗口没装下」。P1-a 只取了每路头 1；本脚本回答
「换一种静态头部配额（v1+k2、v1+k3…）能不能多拿，且不掉」。
★ 只测文档库：案例库走稀疏并集，构图不同（另测会混口径）。

口径
    · 候选：vector 原始 top-20 按 0.50 过滤；keyword 原始 top-20 按
      「Top1≥14 且 ≥Top1×0.35」过滤（与线上 `_retrieve_*` 一致）。
    · 融合：`reciprocal_rank_fusion`，权重用基线 detail 里的 intent 对应值。
    · 命中：gold 是否落在窗口前 5（提示词当前只取 5 条）。
    · 本脚本只读；不写库、不改线上配置。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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
MIN_VECTOR_SCORE = 0.50
MIN_KEYWORD_SCORE = 14.0
MIN_KEYWORD_RELATIVE = 0.35
WINDOW = 5

#: 构图：按顺序列出要占住窗口头部的条目，如 ("v1","k1","k2")。
#: 余下位置由 RRF 序回填（跨源去重）。
COMPOSITIONS: dict[str, tuple[str, ...]] = {
    "rrf": (),
    "v1_k1（现状 P1-a）": ("v1", "k1"),
    "v1_k1_v2": ("v1", "k1", "v2"),
    "v1_k1_k2": ("v1", "k1", "k2"),
    "v1_k1_k2_k3": ("v1", "k1", "k2", "k3"),
    "v1_k1_k2_k3_k4": ("v1", "k1", "k2", "k3", "k4"),
    "v1_k1_v2_k2": ("v1", "k1", "v2", "k2"),
    "v1_v2_k1": ("v1", "v2", "k1"),
    "v1_k1_v2_v3": ("v1", "k1", "v2", "v3"),
    "v1_k1_k2_v2": ("v1", "k1", "k2", "v2"),
    "v1_v2_k1_k2": ("v1", "v2", "k1", "k2"),
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_rrf():
    """复用线上 RRF 实现，避免模拟与生产两套公式。"""
    sys.path.insert(0, str(PROJECT_CODE))
    from domain.services.rrf_fusion import (  # noqa: PLC0415
        reciprocal_rank_fusion, weights_for_intent)
    return reciprocal_rank_fusion, weights_for_intent


def compose_window(comp, vector_ids, keyword_ids, fused_ids, limit=WINDOW):
    """按构图占头部，再按 RRF 序回填（跨源去重）。"""
    window: list[str] = []
    seen: set[str] = set()
    for token in comp:
        path, rank_s = token[0], token[1:]
        try:
            rank = int(rank_s)
        except ValueError:
            continue
        pool = vector_ids if path == "v" else keyword_ids
        if 1 <= rank <= len(pool) and pool[rank - 1] not in seen:
            seen.add(pool[rank - 1])
            window.append(pool[rank - 1])
        if len(window) >= limit:
            return window
    for cid in fused_ids:
        if len(window) >= limit:
            break
        if cid in seen:
            continue
        seen.add(cid)
        window.append(cid)
    return window


def main() -> int:
    ap = argparse.ArgumentParser(description="窗口构图离线模拟")
    ap.add_argument("--collection", default="health_training_docs")
    ap.add_argument("--remap", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试）")
    args = ap.parse_args()

    dataset = load_jsonl(Path(args.remap) if args.remap else DEFAULT_REMAP)
    dataset = [q for q in dataset if q.get("collection") == args.collection]
    if args.limit:
        dataset = dataset[:args.limit]
    if not dataset:
        raise SystemExit("[FAIL] 该 collection 没有题目")

    reciprocal_rank_fusion, weights_for_intent = _load_rrf()

    from config.assembly import build_container
    from config.settings import Settings
    from domain.services.intent_classifier import IntentClassifier

    print("=" * 72)
    print(f"  窗口构图模拟 collection={args.collection} n={len(dataset)}")
    print(f"  评测集: {args.remap or DEFAULT_REMAP}")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 72)

    container = build_container(Settings())
    if container.vector is None or container.keyword is None:
        raise SystemExit("[FAIL] vector/keyword 端口不可用")
    classifier = IntentClassifier()          # 与 92 号/线上默认一致

    hits = {name: 0 for name in COMPOSITIONS}
    flips_up: dict[str, list[str]] = {name: [] for name in COMPOSITIONS}
    flips_down: dict[str, list[str]] = {name: [] for name in COMPOSITIONS}
    base_name = "v1_k1（现状 P1-a）"
    base_hit: dict[str, bool] = {}

    for i, q in enumerate(dataset, 1):
        qid = q["qid"]
        gold = set(q["gold_chunk_ids"])
        perms = q.get("allowed_permissions")
        perms = tuple(perms) if perms else None

        v_hits = container.vector.search(
            q["question"], collection=args.collection, top_k=20,
            allowed_permissions=perms)
        v_hits = [h for h in v_hits if h.score >= MIN_VECTOR_SCORE]
        v_ids = [h.chunk.chunk_id for h in v_hits]

        k_hits = container.keyword.search(
            q["question"], top_k=20, allowed_permissions=perms)
        if k_hits and k_hits[0].score >= MIN_KEYWORD_SCORE:
            floor = k_hits[0].score * MIN_KEYWORD_RELATIVE
            k_hits = [h for h in k_hits if h.score >= floor]
        else:
            k_hits = []
        k_ids = [h.chunk.chunk_id for h in k_hits]

        w = weights_for_intent(classifier.classify(q["question"]).label)
        fused = reciprocal_rank_fusion(
            [v_ids, k_ids], weights=[w[1], w[2]])
        fused_ids = [cid for cid, _ in fused]

        # 先算基线命中，再比较其他构图（避免字典顺序影响翻转口径）
        base_hit[qid] = bool(
            gold & set(compose_window(COMPOSITIONS[base_name],
                                      v_ids, k_ids, fused_ids)))

        for name, comp in COMPOSITIONS.items():
            window = compose_window(comp, v_ids, k_ids, fused_ids)
            hit = bool(gold & set(window))
            if hit:
                hits[name] += 1
            if name == base_name:
                continue
            if hit and not base_hit[qid]:
                flips_up[name].append(qid)
            elif not hit and base_hit[qid]:
                flips_down[name].append(qid)

        if i % 25 == 0:
            print(f"    {i}/{len(dataset)} ...")

    base = hits[base_name]
    print("\n" + "=" * 72)
    print("  结果（hit@5）")
    print("=" * 72)
    for name in COMPOSITIONS:
        delta = hits[name] - base
        print(f"  {name:22s} {hits[name]:3d}/{len(dataset)}  ({delta:+d})")
    print(f"\n  基线按题命中：{base}/{len(dataset)}")

    best = max(COMPOSITIONS, key=lambda n: hits[n])
    print(f"  最优：{best} = {hits[best]}/{len(dataset)}")
    if hits[best] > base:
        print(f"  相对基线翻转：+{len(flips_up[best])} / -{len(flips_down[best])}")
        print(f"    新增命中: {flips_up[best]}")
        print(f"    丢失命中: {flips_down[best]}")
    else:
        print("  没有构图能超过现状——静态配额已到顶，下一步应走 rerank/微调。")

    run_dir = RUNS_DIR / (
        f"window_sim_{args.collection}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "collection": args.collection,
        "n": len(dataset),
        "eval_set": str(Path(args.remap) if args.remap else DEFAULT_REMAP),
        "hits": hits,
        "baseline": base,
        "best": best,
        "flips_up": flips_up,
        "flips_down": flips_down,
        "window": WINDOW,
        "generated_at": datetime.now(timezone.utc)
        .astimezone().isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  [OK] 入档 {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
