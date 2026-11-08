"""P0-3 汇报取数：表 A（13 道案例库 miss 的 hybrid gold_rank）+ 表 B（附录块 top5）。

只读，不改任何评测产物。用法：

    python train/scripts/94_p03_report.py \
        --dense  train/eval_runs/p03_dense_xxx/detail.jsonl \
        --sparse train/eval_runs/p03_sparse_xxx/detail.jsonl \
        --hybrid train/eval_runs/p03_hybrid_xxx/detail.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]

TABLE_A = {
    "_m3": [
        "gen-C01_s06-paraphrase", "gen-C08_s40-scenario",
        "gen-C08_s41-paraphrase", "gen-C12_s68-scenario",
        "gen-C13_s75-scenario", "gen-C16_s97-paraphrase",
        "gen-C16_s97-scenario",
    ],
    "_m2": ["gen-C04_s11-scenario", "gen-C07_s31-scenario"],
    "_m1": [
        "gen-C06_s23-paraphrase", "gen-C07_s28-scenario",
        "gen-C13_s72-paraphrase", "gen-C16_s93-scenario",
    ],
}
APX = "C16_APX_m1"


def load(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["qid"]] = r
    return rows


def golds() -> dict[str, list[str]]:
    p = TRAIN_ROOT / "data" / "retrieval_eval_v1_remap.jsonl"
    return {r["qid"]: r["gold_chunk_ids"]
            for r in (json.loads(l) for l in p.open(encoding="utf-8") if l.strip())}


def rank_in_head(d: dict, gold: list[str]) -> str:
    """旧 detail 只有 ranked_head（前 10），用于 dense 基线对照。"""
    for i, cid in enumerate(d.get("ranked_head", [])[:10], start=1):
        if cid in gold:
            return str(i)
    return ">10"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True, type=Path)
    ap.add_argument("--sparse", required=True, type=Path)
    ap.add_argument("--hybrid", required=True, type=Path)
    a = ap.parse_args()

    dense, sparse, hybrid = load(a.dense), load(a.sparse), load(a.hybrid)
    gold = golds()

    print("=" * 96)
    print("表 A：13 道案例库 dense miss 在三模式下的 gold_rank（1~20，未命中 null）")
    print("=" * 96)
    hdr = (f'{"组":<5}{"qid":<34}{"gold":<14}'
           f'{"dense":>8}{"sparse":>8}{"hybrid":>8}  {"判定":<12}')
    print(hdr)
    print("-" * 96)
    saved = 0
    for grp, qids in TABLE_A.items():
        for q in qids:
            if q not in hybrid:
                print(f"{grp:<5}{q:<34}  ❌ hybrid detail 中不存在")
                continue
            g = gold[q]

            def fmt(d: dict | None) -> str:
                if d is None:
                    return "-"
                return str(d.get("gold_rank")) if d.get("gold_rank") is not None else "null"

            dh = fmt(dense.get(q))
            sh = fmt(sparse.get(q))
            hh = fmt(hybrid.get(q))
            hr = hybrid[q].get("gold_rank")
            hit = isinstance(hr, int) and hr <= 5
            saved += 1 if hit else 0
            verdict = "★救回 Top5" if hit else ("进前20" if hr is not None else "仍 miss")
            print(f"{grp:<5}{q:<34}{str(g):<14}"
                  f"{dh:>8}{sh:>8}{hh:>8}  {verdict:<12}")
    print("-" * 96)
    print(f"  hybrid 救回 Top5：{saved}/13")

    print()
    print("=" * 96)
    print(f"表 B：附录块 {APX} 进入 top5 / top1 的题数")
    print("=" * 96)
    for name, d in (("dense", dense), ("sparse", sparse), ("hybrid", hybrid)):
        t5 = [r["qid"] for r in d.values() if APX in r.get("ranked_head", [])[:5]]
        t1 = [r["qid"] for r in d.values()
              if r.get("ranked_head", [])[:1] == [APX]]
        print(f"  {name:<8} top5={len(t5):3d}   top1={len(t1):3d}")
        print(f"           top1 题: {', '.join(sorted(t1))}")
    d5 = {r["qid"] for r in dense.values() if APX in r.get("ranked_head", [])[:5]}
    h5 = {r["qid"] for r in hybrid.values() if APX in r.get("ranked_head", [])[:5]}
    print(f"  dense→hybrid：新增进 top5 {len(h5 - d5)} 题 / 退出 top5 {len(d5 - h5)} 题")
    print(f"           新增：{', '.join(sorted(h5 - d5)) or '无'}")
    print(f"           退出：{', '.join(sorted(d5 - h5)) or '无'}")

    print()
    print("=" * 96)
    print("分库明细（三模式）")
    print("=" * 96)
    for name, d in (("dense", dense), ("sparse", sparse), ("hybrid", hybrid)):
        parts = []
        for coll in ("health_case_library", "health_training_docs",
                     "health_question_bank"):
            rows = [r for r in d.values() if r["collection"] == coll]
            parts.append(f"{coll.split('_')[-2]} {sum(1 for r in rows if r['hit@5'])}/{len(rows)}")
        total = sum(1 for r in d.values() if r["hit@5"])
        print(f"  {name:<8} 整体 {total}/{len(d)}   " + "   ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
