"""§3.2 C-2：把案例库 gold 从旧 _sNN 重映射到合并块 _mN / _APX_m1。

    python train/scripts/9x_remap_gold.py ^
        --eval train/data/retrieval_eval_v1.jsonl ^
        --map  08_数据入库/milvus/case_chunk_map.json ^
        --out  train/data/retrieval_eval_v1_remap.jsonl

规则（方案 §3.2 C-2）：
  · 只重写 collection == health_case_library 的 34 行；其余 171 行**逐字节原样**；
  · gold_new = {map[g] for g in gold_old}（去重，按首次出现顺序输出）；
  · 任何 gold 在 map 中找不到 → 立即 FAIL 并列出 qid，禁止静默丢弃；
    所有校验都在写文件之前完成，FAIL 时绝不产出半成品；
  · 打印：总行数 / 改动行数 / 案例库 gold 去重前后条数变化的题（逐条列出）；
  · 特别确认 gen-C16_s102-scenario 的 gold == ["C16_APX_m1"]。

去重诊断的两个口径（都打印，避免歧义）：
  A. C-2 输入 v1：每题 1 条旧 gold，天然无塌缩；
  B. C-1 同语义扩展 gold（--expanded，默认 retrieval_eval_v1_expandedgold.jsonl）：
     每题 2/3/5 条旧 id 映射到同一个新块，是「多条塌成一条」真正发生的地方。
     逐题复核：扩展 gold 映射并去重后必须与本次输出完全一致，否则 FAIL
     （这同时证明从 v1 或 expandedgold 出发得到的是同一份评测集）。

不含任何重建 / 写库动作。
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
DEFAULT_EXPANDED = ROOT / "train" / "data" / "retrieval_eval_v1_expandedgold.jsonl"
DEFAULT_MAP = ROOT / "08_数据入库" / "milvus" / "case_chunk_map.json"
DEFAULT_OUT = ROOT / "train" / "data" / "retrieval_eval_v1_remap.jsonl"

CASE_COLLECTION = "health_case_library"
EXPECT_LINES = 205
EXPECT_CHANGED = 34
EXPECT_MAP_ENTRIES = 117
EXPECT_NEW_BLOCKS = 49
SPOT_QID = "gen-C16_s102-scenario"
SPOT_EXPECT = ["C16_APX_m1"]


def read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def load_map(path: Path) -> tuple[dict[str, str], dict[str, dict]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    forward = {k: v for k, v in doc.items() if isinstance(v, str)}
    meta = {k: v for k, v in doc.items() if isinstance(v, dict)}
    if len(forward) != EXPECT_MAP_ENTRIES or len(meta) != EXPECT_NEW_BLOCKS:
        raise SystemExit(
            f"[FAIL] 映射结构异常：old→new {len(forward)} "
            f"(期望 {EXPECT_MAP_ENTRIES})，新块 {len(meta)} "
            f"(期望 {EXPECT_NEW_BLOCKS})：{path}")
    bad = sorted({v for v in forward.values() if v not in meta})
    if bad:
        raise SystemExit(f"[FAIL] 映射指向不存在的新块：{bad}")

    # 正反向交叉校验：meta.old_ids 必须与正向映射分组完全吻合
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for oid, nid in sorted(forward.items()):
        grouped[nid].append(oid)
    for nid, m in meta.items():
        if sorted(m["old_ids"]) != sorted(grouped.get(nid, [])):
            raise SystemExit(f"[FAIL] 反向表与正向映射不一致：{nid}")
    if len(grouped) != len(meta):
        raise SystemExit("[FAIL] 正向映射出现 meta 中不存在的新块 id")
    return forward, meta


def remap(golds: list[str], forward: dict[str, str],
          qid: str, missing: list[tuple[str, str]]) -> list[str]:
    """gold_new = {map[g] for g in gold_old}，按首次出现顺序输出。"""
    out: list[str] = []
    for g in golds:
        if g not in forward:
            missing.append((qid, g))
            continue
        nid = forward[g]
        if nid not in out:
            out.append(nid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 gold 重映射评测集（§3.2 C-2）")
    ap.add_argument("--eval", default=str(DEFAULT_EVAL))
    ap.add_argument("--map", default=str(DEFAULT_MAP))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--expanded", default=str(DEFAULT_EXPANDED),
                    help="C-1 扩展 gold 评测集，仅用于去重诊断与逐题交叉复核")
    args = ap.parse_args()

    eval_path, map_path = Path(args.eval), Path(args.map)
    out_path, expanded_path = Path(args.out), Path(args.expanded)
    forward, meta = load_map(map_path)

    raw_lines = read_lines(eval_path)
    out_lines: list[str] = []
    changed = 0
    missing: list[tuple[str, str]] = []
    per_case: list[tuple[str, list[str], list[str]]] = []

    for lineno, raw in enumerate(raw_lines, 1):
        row = json.loads(raw)
        if row.get("collection") != CASE_COLLECTION:
            out_lines.append(raw)                 # 逐字节原样
            continue
        qid = row["qid"]
        golds = list(row["gold_chunk_ids"])
        gold_new = remap(golds, forward, qid, missing)
        if not gold_new:
            raise SystemExit(f"[FAIL] 第 {lineno} 行 remap 后 gold 为空：{qid}")
        unknown = [nid for nid in gold_new if nid not in meta]
        if unknown:
            raise SystemExit(f"[FAIL] 第 {lineno} 行 gold 指向未知新块：{qid} {unknown}")

        new_row = dict(row)
        new_row["gold_chunk_ids"] = gold_new
        out_lines.append(json.dumps(new_row, ensure_ascii=False))
        changed += 1
        per_case.append((qid, golds, gold_new))

    # ---- 所有 FAIL 检查在写文件之前完成 ----
    if missing:
        qids = sorted({q for q, _ in missing})
        raise SystemExit(
            f"[FAIL] {len(missing)} 条 gold 在 map 中找不到（涉及 "
            f"{len(qids)} 个 qid，禁止静默丢弃）：\n"
            + "\n".join(f"    {q}  {g}" for q, g in missing)
            + f"\n  qid 清单：{qids}")
    if len(raw_lines) != EXPECT_LINES or len(out_lines) != EXPECT_LINES:
        raise SystemExit(f"[FAIL] 行数异常：输入 {len(raw_lines)} / 输出 {len(out_lines)} "
                         f"（期望 {EXPECT_LINES}）")
    if changed != EXPECT_CHANGED:
        raise SystemExit(f"[FAIL] 改动行数 {changed} != {EXPECT_CHANGED}")
    unchanged_ok = all(
        out_lines[i] == raw_lines[i]
        for i in range(len(raw_lines))
        if json.loads(raw_lines[i]).get("collection") != CASE_COLLECTION
    )
    if not unchanged_ok:
        raise SystemExit("[FAIL] 非案例库行被意外改写")

    by_qid = {qid: gold_new for qid, _, gold_new in per_case}
    spot = by_qid.get(SPOT_QID)
    if spot != SPOT_EXPECT:
        raise SystemExit(f"[FAIL] {SPOT_QID} gold 异常：{spot}，期望 {SPOT_EXPECT}")

    # ---- 与 C-1 扩展 gold 逐题交叉复核（同语义 → 同一输出）----
    exp_lines = read_lines(expanded_path)
    if len(exp_lines) != EXPECT_LINES:
        raise SystemExit(f"[FAIL] C-1 扩展集行数 {len(exp_lines)} != {EXPECT_LINES}")
    exp_by_qid: dict[str, list[str]] = {}
    for raw in exp_lines:
        r = json.loads(raw)
        if r.get("collection") == CASE_COLLECTION:
            exp_by_qid[r["qid"]] = list(r["gold_chunk_ids"])
    if len(exp_by_qid) != EXPECT_CHANGED:
        raise SystemExit(f"[FAIL] C-1 扩展集案例行数 {len(exp_by_qid)} != {EXPECT_CHANGED}")
    # 非案例库行在两版之间也必须逐字节一致
    for a, b in zip(raw_lines, exp_lines):
        ja = json.loads(a)
        if ja.get("collection") != CASE_COLLECTION and a != b:
            raise SystemExit(f"[FAIL] C-1 扩展集的非案例库行与 v1 不一致：{ja['qid']}")

    dedup_rows: list[tuple[str, int, int, int, list[str], list[str]]] = []
    exp_missing: list[tuple[str, str]] = []
    for qid, golds, gold_new in per_case:
        exp_golds = exp_by_qid.get(qid)
        if exp_golds is None:
            raise SystemExit(f"[FAIL] C-1 扩展集缺少 qid：{qid}")
        exp_new = remap(exp_golds, forward, qid, exp_missing)
        if exp_missing:
            raise SystemExit(
                f"[FAIL] C-1 扩展集 gold 在 map 中找不到：{exp_missing}")
        if exp_new != gold_new:
            raise SystemExit(
                f"[FAIL] 同语义交叉复核不一致 {qid}："
                f"v1→{gold_new}，expanded→{exp_new}")
        exp_distinct = len(set(forward[g] for g in exp_golds))
        dedup_rows.append((qid, len(exp_golds), exp_distinct, len(gold_new),
                           exp_golds, gold_new))

    # ---- 写文件（全部校验通过后才落盘）----
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # ---- 打印 ----
    before_total = sum(len(g) for _, g, _ in per_case)
    after_total = sum(len(n) for _, _, n in per_case)
    old_ids = {g for _, g, _ in per_case for g in g}
    new_ids = {n for _, _, n in per_case for n in n}
    print("=" * 78)
    print("  §3.2 C-2  gold 重映射")
    print(f"  映射:   {map_path}（{len(forward)} 条 old→new，{len(meta)} 个新块）")
    print(f"  输入:   {eval_path}（{len(raw_lines)} 行）")
    print(f"  输出:   {out_path}")
    print(f"  总行数: {len(out_lines)}   改动行数: {changed}   "
          f"其余 {len(out_lines) - changed} 行逐字节原样（已逐行校验）")
    print("-" * 78)
    print(f"  案例库 gold 条目数: 改前 {before_total}（{changed} 题，"
          f"不同旧 id {len(old_ids)} 个）→ 改后 {after_total}"
          f"（不同新 id {len(new_ids)} 个）")
    print(f"  gold 映射失败: 0 条（117 条旧 id 全覆盖，0 条静默丢弃）")
    print("-" * 78)
    a_changes = [(q, len(g), len(n)) for q, g, n in per_case if len(g) != len(n)]
    print(f"  [口径 A] 以 C-2 输入 v1 为去重前口径：条数变化 {len(a_changes)} 题"
          f"（每题 1 条旧 gold，1 条新 gold，天然无塌缩）")
    for q, b, a in a_changes:
        print(f"    {q}: {b} → {a}")
    b_changes = [r for r in dedup_rows if r[1] != r[3]]
    print(f"  [口径 B] 以 C-1 同语义扩展 gold 为去重前口径："
          f"条数变化 {len(b_changes)} 题（多条旧 id 塌成同一合并块，逐条列出）")
    for qid, n_exp, n_distinct, n_final, exp_golds, gold_new in dedup_rows:
        mark = "  ← 变化" if n_exp != n_final else ""
        print(f"    {qid:<32} 扩展 {n_exp} 条 → 映射去重 {n_distinct} 条 "
              f"→ 最终 {n_final} 条  {exp_golds} → {gold_new}{mark}")
    print("-" * 78)
    multi = [(q, n) for q, _, n in per_case if len(n) != 1]
    print(f"  恒真检查：最终 gold 多于 1 条的题 {len(multi)} 题"
          + (f" {multi}" if multi else "（每题恰好 1 个合并块，未出现多块恒真）"))
    print(f"  ★ {SPOT_QID}: "
          f"{[g for q, g, _ in per_case if q == SPOT_QID][0]} → {spot}  "
          f"{'OK' if spot == SPOT_EXPECT else 'FAIL'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
