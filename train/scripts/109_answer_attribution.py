"""把 100 题答案评测的失分逐题归因，并对比新旧评分器/尺子。

用法：
    python train/scripts/109_answer_attribution.py \
        --runs a_ab_r1_20261108_200158 a_ab_r2_20261108_202017 a_ab_r3_20261108_204000

产物写到 train/eval_runs/answer_attribution_<时间戳>/：
    summary.json       轮次得分、分类计数、常挂题、评分器翻转
    attribution.jsonl  每题一行的合并归因
    attribution.md     人读版报告

口径：
- 新尺子 = train/data/eval_questions_v4_100_revised2.jsonl
- 旧尺子 = train/data/eval_questions_v4_100_revised.jsonl
- 新评分器 = 60_score_eval.score_one（否定/引述/列表变体）
- 旧评分器 = score_one(..., legacy=True)（纯子串）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
SCORE_PATH = Path(__file__).resolve().parent / "60_score_eval.py"
DEFAULT_EVAL_OLD = ROOT / "train/data/eval_questions_v4_100_revised.jsonl"
DEFAULT_EVAL_NEW = ROOT / "train/data/eval_questions_v4_100_revised2.jsonl"
DEFAULT_RUNS = [
    "a_ab_r1_20261108_200158",
    "a_ab_r2_20261108_202017",
    "a_ab_r3_20261108_204000",
]


def _load_score_module():
    spec = importlib.util.spec_from_file_location("score60", SCORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_by_qid(path: Path) -> dict[str, dict]:
    return {row["qid"]: row for row in load_jsonl(path)}


def resolve_run(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        return path
    return ROOT / "train/eval_runs" / value


def prediction_file(run_dir: Path) -> Path:
    for name in ("detail_5.jsonl", "pred_5.jsonl"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{run_dir} 下没有 detail_5.jsonl / pred_5.jsonl")


def primary_cause(record: dict) -> str:
    if record["refused"]:
        return "refused"
    if not record["gold_in_context"]:
        return "gold_not_in_context"
    if record["forbidden_hit"]:
        return "forbidden"
    if record["missed"]:
        return "missed_points"
    return "other"


def score_run(
    rows: list[dict],
    items_old: dict[str, dict],
    items_new: dict[str, dict],
    scorer,
) -> tuple[dict, dict[str, dict]]:
    baseline = 0
    legacy_new = 0
    calibrated = 0
    per_qid: dict[str, dict] = {}
    for row in rows:
        qid = row["qid"]
        answer = row.get("answer") or row.get("output") or ""
        old_item = items_old[qid]
        new_item = items_new[qid]
        base = scorer.score_one(old_item, answer, legacy=True)
        legacy = scorer.score_one(new_item, answer, legacy=True)
        new = scorer.score_one(new_item, answer)
        baseline += base["passed"]
        legacy_new += legacy["passed"]
        calibrated += new["passed"]
        per_qid[qid] = {
            "qid": qid,
            "question": new_item.get("question", ""),
            "passed": new["passed"],
            "missed": new["missed"],
            "forbidden_hit": new["forbidden_hit"],
            "refused": bool(row.get("refused")),
            "refusal_reason": row.get("refusal_reason", ""),
            "gold_in_context": bool(row.get("gold_in_context", True)),
            "baseline_old_ruler": base["passed"],
            "rubric_only": legacy["passed"],
            "calibrated": new["passed"],
        }
    return {
        "baseline_old_ruler": baseline,
        "rubric_fixes_only": legacy_new,
        "calibrated": calibrated,
    }, per_qid


def fmt_list(values: list[str], limit: int = 3) -> str:
    if not values:
        return ""
    shown = "；".join(v.replace("\n", " ")[:80] for v in values[:limit])
    return shown + ("…" if len(values) > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--eval-old", type=Path, default=DEFAULT_EVAL_OLD)
    parser.add_argument("--eval-new", type=Path, default=DEFAULT_EVAL_NEW)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    scorer = _load_score_module()
    items_old = load_by_qid(args.eval_old)
    items_new = load_by_qid(args.eval_new)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or ROOT / f"train/eval_runs/answer_attribution_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_summaries = []
    per_qid_runs: dict[str, list[dict]] = defaultdict(list)
    for run_value in args.runs:
        run_dir = resolve_run(run_value)
        rows = [row for row in load_jsonl(prediction_file(run_dir)) if row["qid"] in items_new]
        scores, per_qid = score_run(rows, items_old, items_new, scorer)
        scores["run"] = run_dir.name
        run_summaries.append(scores)
        for qid, record in per_qid.items():
            per_qid_runs[qid].append(record)

    category_counter = Counter()
    for records in per_qid_runs.values():
        for record in records:
            if not record["calibrated"]:
                category_counter[primary_cause(record)] += 1

    attribution_rows = []
    for qid, records in per_qid_runs.items():
        fail_records = [r for r in records if not r["calibrated"]]
        causes = Counter(primary_cause(r) for r in fail_records)
        missed = Counter(m for r in fail_records for m in r["missed"])
        forbidden = Counter(m for r in fail_records for m in r["forbidden_hit"])
        attribution_rows.append(
            {
                "qid": qid,
                "question": records[0]["question"],
                "runs": len(records),
                "pass_runs": len(records) - len(fail_records),
                "fail_runs": len(fail_records),
                "causes": dict(causes),
                "missed": [{"phrase": p, "runs": c} for p, c in missed.most_common()],
                "forbidden": [{"phrase": p, "runs": c} for p, c in forbidden.most_common()],
                "gold_missing_runs": sum(1 for r in fail_records if not r["gold_in_context"]),
                "refused_runs": sum(1 for r in fail_records if r["refused"]),
                "flipped_by_scorer": sum(
                    1 for r in records if not r["rubric_only"] and r["calibrated"]
                ),
                "rubric_fix_candidates": [
                    r["qid"] for r in records if r["baseline_old_ruler"] != r["rubric_only"]
                ],
            }
        )
    attribution_rows.sort(key=lambda row: (-row["fail_runs"], -row["runs"], row["qid"]))

    calibrated_scores = [r["calibrated"] for r in run_summaries]
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_old": str(args.eval_old),
        "eval_new": str(args.eval_new),
        "runs": run_summaries,
        "median": {
            "baseline_old_ruler": statistics.median(
                r["baseline_old_ruler"] for r in run_summaries
            ),
            "rubric_fixes_only": statistics.median(
                r["rubric_fixes_only"] for r in run_summaries
            ),
            "calibrated": statistics.median(calibrated_scores),
        },
        "categories": dict(category_counter),
        "persistent_failures": [
            row["qid"] for row in attribution_rows if row["fail_runs"] == row["runs"]
        ],
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "attribution.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for row in attribution_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# 100 题答案评测归因（自动生成）",
        "",
        f"> 生成时间：{summary['generated_at']}",
        f"> 尺子：`{args.eval_new.name}`；旧尺子：`{args.eval_old.name}`",
        "",
        "## 1. 轮次得分",
        "",
        "| 跑次 | 旧尺子+旧评分器 | 新尺子+旧评分器 | 新尺子+新评分器 |",
        "|---|---:|---:|---:|",
    ]
    for row in run_summaries:
        lines.append(
            f"| {row['run']} | {row['baseline_old_ruler']} | "
            f"{row['rubric_fixes_only']} | {row['calibrated']} |"
        )
    lines += [
        f"| **中位** | **{summary['median']['baseline_old_ruler']:.0f}** | "
        f"**{summary['median']['rubric_fixes_only']:.0f}** | "
        f"**{summary['median']['calibrated']:.0f}** |",
        "",
        "## 2. 失败分类（按轮次计）",
        "",
        "| 原因 | 次数 |",
        "|---|---:|",
    ]
    for cause, count in category_counter.most_common():
        lines.append(f"| {cause} | {count} |")
    lines += [
        "",
        f"## 3. 三轮全挂（{len(summary['persistent_failures'])} 道）",
        "",
        "| qid | 主因 | 缺要点 | 禁用命中 | gold 不在上下文 |",
        "|---|---|---|---|---|",
    ]
    persistent = [row for row in attribution_rows if row["fail_runs"] == row["runs"]]
    for row in persistent:
        cause = "／".join(f"{k}×{v}" for k, v in row["causes"].items())
        missed = "；".join(item["phrase"] for item in row["missed"][:3])
        forbidden = "；".join(item["phrase"] for item in row["forbidden"][:3])
        lines.append(
            f"| `{row['qid']}` | {cause} | {fmt_list([missed])} | "
            f"{fmt_list([forbidden])} | {row['gold_missing_runs']} |"
        )
    lines += [
        "",
        "## 4. 评分器/尺子翻转",
        "",
    ]
    flips = [
        row
        for row in attribution_rows
        if row["flipped_by_scorer"] or row["rubric_fix_candidates"]
    ]
    if flips:
        lines += ["| qid | 旧尺子旧评分器 | 新尺子旧评分器 | 新尺子新评分器 |", "|---|---:|---:|---:|"]
        for row in flips:
            records = per_qid_runs[row["qid"]]
            old = sum(r["baseline_old_ruler"] for r in records)
            mid = sum(r["rubric_only"] for r in records)
            new = sum(r["calibrated"] for r in records)
            lines.append(f"| `{row['qid']}` | {old}/{row['runs']} | {mid}/{row['runs']} | {new}/{row['runs']} |")
    else:
        lines.append("（无）")
    lines += [
        "",
        "## 5. 判读",
        "",
        "- `gold_not_in_context` 是检索构图问题，先看该题 gold 是否真的在库/是否 gold 本身错。",
        "- `missed_points` 是生成侧问题，优先看数字、单位、警示语、遵医嘱/就医等必答项。",
        "- `forbidden` 若来自否定/引述语境，应继续修评分器；若是真实宣称，属产品护栏问题。",
        "- `refused` 是产品护栏拒答，需区分真实违规与引述误拒。",
    ]
    (out_dir / "attribution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"得分中位：旧尺子旧评分器 {summary['median']['baseline_old_ruler']:.0f} → "
          f"新尺子旧评分器 {summary['median']['rubric_fixes_only']:.0f} → "
          f"新尺子新评分器 {summary['median']['calibrated']:.0f}")
    print(f"分类：{dict(category_counter)}")
    print(f"三轮全挂：{len(summary['persistent_failures'])} 道")
    print(f"产物 -> {out_dir}")


if __name__ == "__main__":
    main()
