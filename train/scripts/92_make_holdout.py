"""P0-1：按 (origin, collection) 严格分层切 holdout（正式划分只抽一次）。

    python train/scripts/92_make_holdout.py

产出：
    train/data/holdout_v1.json
    train/data/retrieval_eval_v1_holdout.jsonl       55 行，原样拷贝
    train/data/embed_pairs_v1_train150.jsonl         150 行，split 重算（val=15）
    train/eval_runs/p01_checks.json                  全部断言数字落盘

方案：docs/实施方案_P0检索质量三件套_2026-11-07.md §2。
★ 不改 retrieval_eval_v1.jsonl 原文件；seed 写入 holdout_v1.json 后不再修改。
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = TRAIN_ROOT / "data"
RUNS_DIR = TRAIN_ROOT / "eval_runs"

EVAL_PATH = DATA_DIR / "retrieval_eval_v1.jsonl"
PAIRS_PATH = DATA_DIR / "embed_pairs_v1.jsonl"

SEED = 20261107
#: (origin, collection) -> (层内总量, holdout 抽题数)。来自方案 §2.2 表，逐格硬断言。
STRATA_SPEC: dict[tuple[str, str], tuple[int, int]] = {
    ("auto_exact", "health_question_bank"): (40, 11),
    ("eval_v3", "health_training_docs"): (45, 12),
    ("gen_b", "health_case_library"): (34, 9),
    ("gen_b", "health_training_docs"): (86, 23),
}
EXPECT_HOLDOUT = 55
EXPECT_TRAIN = 150
EXPECT_VAL = 15
EXPECT_MIN_UNIQUE_GOLD = 45


def fail(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


def load_jsonl_with_raw(path: Path) -> tuple[list[dict], list[str]]:
    raw = [line.rstrip("\r\n") for line in path.open(encoding="utf-8") if line.strip()]
    return [json.loads(line) for line in raw], raw


def main() -> int:
    if not EVAL_PATH.exists() or not PAIRS_PATH.exists():
        fail("找不到 retrieval_eval_v1.jsonl / embed_pairs_v1.jsonl")

    eval_rows, eval_raw = load_jsonl_with_raw(EVAL_PATH)
    pair_rows, _ = load_jsonl_with_raw(PAIRS_PATH)
    qid_order = [r["qid"] for r in eval_rows]
    qid_set = set(qid_order)
    if len(qid_set) != len(qid_order):
        fail("评测集 qid 有重复")
    qid_to_raw = dict(zip(qid_order, eval_raw))
    qid_to_pair = {r["qid"]: r for r in pair_rows}
    if len(qid_to_pair) != len(pair_rows):
        fail("训练对 qid 有重复")
    if set(qid_to_pair) != qid_set:
        fail("训练对与评测集 qid 集合不一致")

    origin_of = {r["qid"]: r["origin"] for r in eval_rows}
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in eval_rows:
        groups[(r["origin"], r["collection"])].append(r["qid"])
    if set(groups) != set(STRATA_SPEC):
        fail(f"分层键不符：实际 {sorted(map('|'.join, groups))}")

    # ---- 正式划分：同一个 rng，只抽一次 ----
    rng = random.Random(SEED)
    strata_out: dict[str, dict] = {}
    holdout_set: set[str] = set()
    for key, (exp_total, exp_hold) in STRATA_SPEC.items():
        qs = sorted(groups[key])
        if len(qs) != exp_total:
            fail(f"层 {'|'.join(key)} 总量 {len(qs)} != 预期 {exp_total}")
        picked = set(rng.sample(qs, exp_hold))
        holdout_set |= picked
        strata_out["|".join(key)] = {
            "total": len(qs),
            "holdout": len(picked),
            "train": len(qs) - len(picked),
            "expected_total": exp_total,
            "expected_holdout": exp_hold,
            "holdout_qids": sorted(picked),
        }

    holdout_qids = [q for q in qid_order if q in holdout_set]
    train_qids = [q for q in qid_order if q not in holdout_set]
    train_set = set(train_qids)

    # ---- 断言（缺一即失败，失败时不写任何产物）----
    assertions: dict[str, bool] = {}
    assertions["strata_counts_exact"] = all(
        s["total"] == s["expected_total"] and s["holdout"] == s["expected_holdout"]
        for s in strata_out.values())
    assertions["holdout_55"] = len(holdout_qids) == EXPECT_HOLDOUT
    assertions["train_150"] = len(train_qids) == EXPECT_TRAIN
    assertions["disjoint"] = len(set(holdout_qids) & train_set) == 0
    assertions["union_eq_205"] = (set(holdout_qids) | train_set) == qid_set
    assertions["eval_rows_205"] = len(eval_rows) == 205
    assertions["pairs_rows_205"] = len(pair_rows) == 205

    holdout_golds: set[str] = set()
    for q in holdout_qids:
        holdout_golds.update(json.loads(qid_to_raw[q])["gold_chunk_ids"])
    assertions["holdout_unique_gold_ge_45"] = (
        len(holdout_golds) >= EXPECT_MIN_UNIQUE_GOLD)

    # ---- 在剩下 150 条里按 origin 分层重抽 15 条 val（不沿用旧 split）----
    by_origin: dict[str, list[str]] = defaultdict(list)
    for q in train_qids:
        by_origin[origin_of[q]].append(q)
    origins = sorted(by_origin)
    quota_raw = {o: len(by_origin[o]) * EXPECT_VAL / EXPECT_TRAIN for o in origins}
    quota = {o: int(quota_raw[o]) for o in origins}
    remainder = EXPECT_VAL - sum(quota.values())
    for o in sorted(origins, key=lambda x: (-(quota_raw[x] - int(quota_raw[x])), x))[:remainder]:
        quota[o] += 1
    val_set: set[str] = set()
    for o in origins:
        val_set |= set(rng.sample(sorted(by_origin[o]), quota[o]))

    out_pairs = []
    for q in train_qids:
        row = dict(qid_to_pair[q])
        row["split"] = "val" if q in val_set else "train"
        out_pairs.append(json.dumps(row, ensure_ascii=False))
    n_val = sum(1 for q in train_qids if q in val_set)
    assertions["pairs_lines_150"] = len(out_pairs) == EXPECT_TRAIN
    assertions["val_exactly_15"] = n_val == EXPECT_VAL

    all_ok = all(assertions.values())
    if not all_ok:
        bad = [k for k, v in assertions.items() if not v]
        fail(f"断言未通过：{bad}")

    # ---- 写产物 ----
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    holdout_doc = {
        "seed": SEED,
        "created_at": now,
        "strata": strata_out,
        "holdout_qids": holdout_qids,
        "train_qids": train_qids,
    }
    (DATA_DIR / "holdout_v1.json").write_text(
        json.dumps(holdout_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    with (DATA_DIR / "retrieval_eval_v1_holdout.jsonl").open(
            "w", encoding="utf-8", newline="\n") as f:
        for q in holdout_qids:
            f.write(qid_to_raw[q] + "\n")

    with (DATA_DIR / "embed_pairs_v1_train150.jsonl").open(
            "w", encoding="utf-8", newline="\n") as f:
        for line in out_pairs:
            f.write(line + "\n")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    checks = {
        "stage": "P0-1",
        "seed": SEED,
        "created_at": now,
        "source": {"eval": str(EVAL_PATH), "pairs": str(PAIRS_PATH)},
        "strata": {k: {kk: vv for kk, vv in v.items() if kk != "holdout_qids"}
                   for k, v in strata_out.items()},
        "counts": {
            "eval_rows": len(eval_rows),
            "pairs_rows": len(pair_rows),
            "holdout_qids": len(holdout_qids),
            "train_qids": len(train_qids),
            "holdout_unique_gold": len(holdout_golds),
            "pairs_lines": len(out_pairs),
            "split_val": n_val,
            "split_train": len(out_pairs) - n_val,
            "val_quota_by_origin": {o: quota[o] for o in origins},
        },
        "assertions": assertions,
        "all_pass": all_ok,
        "outputs": {
            "holdout_v1": str(DATA_DIR / "holdout_v1.json"),
            "eval_holdout": str(DATA_DIR / "retrieval_eval_v1_holdout.jsonl"),
            "pairs_train150": str(DATA_DIR / "embed_pairs_v1_train150.jsonl"),
        },
        "val_qids": sorted(val_set),
    }
    (RUNS_DIR / "p01_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 68)
    print("  P0-1 切分完成（正式划分，只抽一次）")
    print(f"  seed = {SEED}")
    for k, v in strata_out.items():
        print(f"  {k:<46} total {v['total']:>3} / holdout {v['holdout']:>2}"
              f" / train {v['train']:>3}")
    print(f"  holdout {len(holdout_qids)} / train {len(train_qids)} / "
          f"唯一 gold 块 {len(holdout_golds)}")
    print(f"  split: val {n_val} / train {len(out_pairs) - n_val}")
    print(f"  [OK] 断言 {len(assertions)}/{len(assertions)} 通过，落盘 {RUNS_DIR / 'p01_checks.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
