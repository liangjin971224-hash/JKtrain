"""§3.2 C-1：把案例库 gold 扩展到「合并块」的同语义粒度。

切片从 7 节合并成 3 块后，新口径下「命中合并块里任一小节」就算对；
直接拿旧 gold 去比会高估收益。做法是在**旧库还没重建**时，把案例库每题的
gold 从「1 个旧小节 id」扩展为「该题所在合并块的全部旧小节 id」，
在旧库上跑一次得到公平的对照基线 case_expand_gold_baseline。

    python train/scripts/93_expand_gold.py

规则：
  · 由 case_chunk_map.json 建反向表 new_id → [old_ids]（正向映射的分组结果，
    并与 meta.old_ids 交叉校验）；
  · 只重写 collection == health_case_library 的行，其余行**逐字节原样拷贝**；
  · 任何 gold 在 map 中找不到 → 立即 FAIL 并列出 qid，禁止静默丢弃；
  · 输出改动行数（应为 34）与每题扩展后 gold 条数分布。

不含任何重建/写库动作。
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = ROOT / "train" / "data" / "retrieval_eval_v1.jsonl"
DEFAULT_MAP = ROOT / "08_数据入库" / "milvus" / "case_chunk_map.json"
DEFAULT_OUT = ROOT / "train" / "data" / "retrieval_eval_v1_expandedgold.jsonl"
CASE_COLLECTION = "health_case_library"
EXPECT_CHANGED = 34


def load_map(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """返回 (正向 old→new, 反向 new→[old_ids])。"""
    doc = json.loads(path.read_text(encoding="utf-8"))
    forward = {k: v for k, v in doc.items() if isinstance(v, str)}
    meta = {k: v for k, v in doc.items() if isinstance(v, dict)}

    if not forward or not meta:
        raise SystemExit(f"[FAIL] 映射文件结构异常：{path}")

    # 反向表：以 meta.old_ids（按文件顺序）为准，并与正向映射分组交叉校验
    reverse = {nid: list(m["old_ids"]) for nid, m in meta.items()}
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for oid, nid in sorted(forward.items()):
        grouped[nid].append(oid)
    for nid, oids in reverse.items():
        if sorted(oids) != sorted(grouped.get(nid, [])):
            raise SystemExit(f"[FAIL] 反向表与正向映射不一致：{nid}")
    if len(grouped) != len(reverse):
        raise SystemExit("[FAIL] 正向映射出现 meta 中不存在的新块 id")
    return forward, reverse


def main() -> int:
    ap = argparse.ArgumentParser(description="生成扩展 gold 评测集（§3.2 C-1）")
    ap.add_argument("--eval", default=str(DEFAULT_EVAL))
    ap.add_argument("--map", default=str(DEFAULT_MAP))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    eval_path, map_path, out_path = Path(args.eval), Path(args.map), Path(args.out)
    forward, reverse = load_map(map_path)

    raw_lines = [ln for ln in eval_path.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
    out_lines: list[str] = []
    changed = 0
    dist_before: collections.Counter[int] = collections.Counter()
    dist_after: collections.Counter[int] = collections.Counter()
    per_case_detail: list[tuple[str, str, list[str], list[str]]] = []
    missing: list[tuple[str, str]] = []

    for lineno, raw in enumerate(raw_lines, 1):
        row = json.loads(raw)
        if row.get("collection") != CASE_COLLECTION:
            out_lines.append(raw)          # 其余行逐字节原样
            continue

        golds = list(row["gold_chunk_ids"])
        dist_before[len(golds)] += 1
        expanded: list[str] = []
        for g in golds:
            if g not in forward:
                missing.append((row["qid"], g))
                continue
            for oid in reverse[forward[g]]:
                if oid not in expanded:
                    expanded.append(oid)
        if missing:
            continue
        if not expanded or any(g not in expanded for g in golds):
            raise SystemExit(f"[FAIL] 第 {lineno} 行 gold 扩展异常：{row['qid']}")

        new_row = dict(row)
        new_row["gold_chunk_ids"] = expanded
        out_lines.append(json.dumps(new_row, ensure_ascii=False))
        dist_after[len(expanded)] += 1
        changed += 1
        per_case_detail.append((row["qid"], forward[golds[0]], golds, expanded))

    if missing:
        raise SystemExit(f"[FAIL] {len(missing)} 条 gold 在 map 中找不到：{missing}")
    if changed != EXPECT_CHANGED:
        raise SystemExit(f"[FAIL] 改动行数 {changed} != {EXPECT_CHANGED}")
    if len(out_lines) != len(raw_lines):
        raise SystemExit("[FAIL] 输出行数与输入不一致")

    # 未改动行逐字节校验
    unchanged_ok = all(
        out_lines[i] == raw_lines[i]
        for i in range(len(raw_lines))
        if json.loads(raw_lines[i]).get("collection") != CASE_COLLECTION
    )
    if not unchanged_ok:
        raise SystemExit("[FAIL] 非案例库行被意外改写")

    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print("=" * 72)
    print(f"  映射:   {map_path}（{len(forward)} 条 old→new，{len(reverse)} 个新块）")
    print(f"  输入:   {eval_path}（{len(raw_lines)} 行）")
    print(f"  输出:   {out_path}")
    print(f"  总行数: {len(out_lines)}   改动行数: {changed}   "
          f"其余 {len(out_lines) - changed} 行逐字节原样")
    print("-" * 72)
    print(f"  改动前 gold 条数分布: {dict(sorted(dist_before.items()))}")
    print(f"  扩展后 gold 条数分布: {dict(sorted(dist_after.items()))}")
    print("-" * 72)
    print("  按合并块分组（题数 / 扩展条数 / 新块）:")
    by_block: dict[str, list[int]] = collections.defaultdict(list)
    for _qid, nid, _old, exp in per_case_detail:
        by_block[nid].append(len(exp))
    for nid in sorted(by_block):
        print(f"    {nid:<16} {len(by_block[nid]):>2} 题  "
              f"gold={sorted(set(by_block[nid]))}  含旧小节 {len(reverse[nid])} 条")
    print("-" * 72)
    for qid, nid, old, exp in per_case_detail:
        if "APX" in nid or qid == "gen-C16_s102-scenario":
            print(f"  {qid}: {old} → {nid} → {exp}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
