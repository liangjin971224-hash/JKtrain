"""质量复测：三次重复取中位 + `context_docs` 5/8 配对比较。

    python train/scripts/108_quality_recheck.py
    python train/scripts/108_quality_recheck.py --prefix a_ab_r --rounds 3

输入：`train/eval_runs/{prefix}{1..N}_*/`（由 95_answer_ab.py `--variants 5,8` 产出，
每个跑次目录里同时有 `summary.json`（含两个变体）与 `detail_5/8.jsonl`）。

输出：控制台表 + `train/eval_runs/quality_recheck_<ts>/{summary.json,report.md}`。

★ 为什么不用单次跑分：同一配置实测 69/72/76/77 都出现过（LLM 输出波动 ±5）。
  单次差异小于 5 时无法判断是「变体效果」还是「这一次模型心情」。
  所以：**中位数做结论**；配对检验只在同一轮次内做 —— 同一进程、同一模型状态、
  同一批检索结果，差异才归因于 `context_docs`。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
RUNS = TRAIN_ROOT / "eval_runs"
EVAL_SET = TRAIN_ROOT / "data" / "eval_questions_v4_100_revised.jsonl"


def _load_95():
    """复用 95 号里的精确 McNemar，避免另写一套统计。"""
    path = Path(__file__).with_name("95_answer_ab.py")
    spec = importlib.util.spec_from_file_location("ab95", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_detail(path: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["qid"]] = bool(row["passed"])
    return out


def find_round(prefix: str, i: int) -> Path:
    matches = sorted(RUNS.glob(f"{prefix}{i}_*"))
    if not matches:
        raise SystemExit(f"缺少跑次: {prefix}{i}_*")
    return matches[-1]


def med(xs: list[float]) -> float:
    return statistics.median(xs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="a_ab_r")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--variants", default="5,8")
    ap.add_argument("--case-origin", default="case")
    args = ap.parse_args()

    variants = [str(v) for v in args.variants.split(",") if v.strip()]
    case_qids = {
        r["qid"] for r in
        (json.loads(line) for line in EVAL_SET.open(encoding="utf-8") if line.strip())
        if r.get("data_origin") == args.case_origin
    }
    print(f"题集 {EVAL_SET.name}；案例题 {len(case_qids)} 道")

    runs = [find_round(args.prefix, i) for i in range(1, args.rounds + 1)]
    rows: dict[str, list[dict]] = {v: [] for v in variants}
    details: dict[str, list[dict[str, bool]]] = {v: [] for v in variants}
    for run in runs:
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        for v in variants:
            if v not in summary["results"]:
                raise SystemExit(f"{run.name} 里没有变体 {v}")
            rows[v].append(dict(summary["results"][v]))
            details[v].append(read_detail(run / f"detail_{v}.jsonl"))
        print(f"  {run.name}: " + "  ".join(
            f"ctx{v}={summary['results'][v]['passed']}" for v in variants))

    print("\n=== 中位数 ===")
    print(f"{'ctx':>4} {'通过':>8} {'区间':>9} {'禁用':>5} {'拒答':>5} {'字数':>6} {'P50':>6}")
    for v in variants:
        p = [r["passed"] for r in rows[v]]
        print(f"{v:>4} {med(p):>8.0f} {min(p)}~{max(p):>6} "
              f"{med([r['forbidden_questions'] for r in rows[v]]):>5.0f} "
              f"{med([r['refused'] for r in rows[v]]):>5.0f} "
              f"{med([r['avg_answer_chars'] for r in rows[v]]):>6.0f} "
              f"{med([r['p50_elapsed_s'] for r in rows[v]]):>6.2f}")

    pair: dict[str, list[dict]] = {"overall": [], "case": []}
    if len(variants) == 2:
        ab = _load_95()
        a, b = variants
        for rnd, (da, db) in enumerate(zip(details[a], details[b]), 1):
            common = set(da) & set(db)
            for scope, qids in (("overall", common),
                                ("case", common & case_qids)):
                only_a = sum(1 for q in qids if da[q] and not db[q])
                only_b = sum(1 for q in qids if db[q] and not da[q])
                pair[scope].append({
                    "round": rnd, "a_only": only_a, "b_only": only_b,
                    "p": round(ab.mcnemar_exact(only_a, only_b), 4)})
        print(f"\n=== 配对（ctx{a} vs ctx{b}，逐轮同进程） ===")
        for scope, items in pair.items():
            label = "全部" if scope == "overall" else f"案例 {len(case_qids)}"
            for it in items:
                print(f"  {label} 第{it['round']}轮: "
                      f"仅{a}对={it['a_only']} 仅{b}对={it['b_only']} "
                      f"p={it['p']:.3f}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RUNS / f"quality_recheck_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "eval_set": str(EVAL_SET.relative_to(TRAIN_ROOT.parent)),
        "rounds": args.rounds,
        "runs": [str(r) for r in runs],
        "case_questions": len(case_qids),
        "median": {
            v: {
                "passed": med([r["passed"] for r in rows[v]]),
                "min": min(r["passed"] for r in rows[v]),
                "max": max(r["passed"] for r in rows[v]),
                "forbidden": med([r["forbidden_questions"] for r in rows[v]]),
                "refused": med([r["refused"] for r in rows[v]]),
                "avg_answer_chars": med([r["avg_answer_chars"] for r in rows[v]]),
                "p50_elapsed_s": med([r["p50_elapsed_s"] for r in rows[v]]),
            } for v in variants
        },
        "paired": pair,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# 质量复测 {stamp}", "",
        f"题集 `{EVAL_SET.name}`，{args.rounds} 轮同进程配对。", "",
        "| ctx | 通过（中位） | 区间 | 禁用 | 拒答 | P50 |",
        "|---|---|---|---|---|---|",
    ]
    for v in variants:
        s = summary["median"][v]
        lines.append(f"| {v} | {s['passed']:.0f}/{rows[v][0]['n']} | "
                     f"{s['min']}~{s['max']} | {s['forbidden']:.0f} | "
                     f"{s['refused']:.0f} | {s['p50_elapsed_s']:.2f}s |")
    if pair["overall"]:
        lines += ["", "## 配对（同轮次）", "",
                  "| 轮 | 全部: 仅5对 | 仅8对 | p | 案例: 仅5对 | 仅8对 | p |",
                  "|---|---|---|---|---|---|---|"]
        for o, c in zip(pair["overall"], pair["case"]):
            lines.append(f"| {o['round']} | {o['a_only']} | {o['b_only']} | "
                         f"{o['p']:.3f} | {c['a_only']} | {c['b_only']} | {c['p']:.3f} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[OK] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
