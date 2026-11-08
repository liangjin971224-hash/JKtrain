"""校验修订后的案例 rubric（103 号修订包的输出）。

    python train/scripts/104_validate_revised_rubrics.py --wp-dir train/annotation_wp_v3

硬校验：schema / 31 行 / must_include 1~5 / any_of 2~5 / 总必答点 2~5 /
        must_include 与 any_of 至少一个变体能在 gold 找到 / must_not 否定误伤。
软警告：总必答点比原 rubric 少 ≥2（放宽过度）/ any_of 组内变体过多。
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WP = TRAIN_ROOT / "annotation_wp_v3"
MIN_PHRASE, MAX_PHRASE = 2, 20


def normalize(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in s if not ch.isspace())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="校验修订 rubric")
    ap.add_argument("--wp-dir", default=str(DEFAULT_WP))
    args = ap.parse_args()
    wp = Path(args.wp_dir) / "revision"

    targets = load_jsonl(wp / "targets.jsonl")
    current = {r["qid"]: r["expect"]
               for r in load_jsonl(wp / "current_rubrics.jsonl")}
    pred_path = wp / "outputs" / "answer_rubrics_revised.jsonl"
    preds = load_jsonl(pred_path) if pred_path.exists() else []
    if not targets:
        print(f"[FAIL] 找不到 {wp}/targets.jsonl")
        return 1
    if not preds:
        print(f"[FAIL] 找不到产出：{pred_path}")
        return 1

    by_qid = {t["qid"]: t for t in targets}
    hard: list[str] = []
    warn: list[str] = []
    seen: set[str] = set()

    for r in preds:
        qid = r.get("qid")
        if qid not in by_qid:
            hard.append(f"未知 qid: {qid}")
            continue
        if qid in seen:
            hard.append(f"qid 重复: {qid}")
        seen.add(qid)
        expect = r.get("expect")
        if not isinstance(expect, dict):
            hard.append(f"{qid} expect 不是对象")
            continue
        extra = set(expect) - {"must_include", "any_of", "must_not_include"}
        if extra:
            hard.append(f"{qid} expect 多字段: {sorted(extra)}")
        inc = expect.get("must_include") or []
        any_of = expect.get("any_of") or []
        exc = expect.get("must_not_include") or []
        if not (1 <= len(inc) <= 5):
            hard.append(f"{qid} must_include 数量 {len(inc)} 不在 1~5")
        total_points = len(inc) + len(any_of)
        if not (2 <= total_points <= 5):
            hard.append(f"{qid} 总必答点 {total_points} 不在 2~5")
        gold_text = normalize("".join(c["text"] for c in by_qid[qid]["gold_chunks"]))
        for p in inc:
            if not isinstance(p, str) or not (MIN_PHRASE <= len(p) <= MAX_PHRASE):
                hard.append(f"{qid} must_include 长度非法: {p!r}")
            elif normalize(p) not in gold_text:
                hard.append(f"{qid} must_include 在 gold 找不到: {p!r}")
        for group in any_of:
            if not isinstance(group, list) or not (2 <= len(group) <= 5):
                hard.append(f"{qid} any_of 组非法: {group!r}")
                continue
            grounded = [g for g in group
                        if isinstance(g, str) and normalize(g) in gold_text]
            if not grounded:
                hard.append(f"{qid} any_of 组在 gold 找不到任一变体: {group!r}")
        if not isinstance(exc, list) or len(exc) > 3:
            hard.append(f"{qid} must_not 数量非法")
        for p in exc:
            if not isinstance(p, str) or not (MIN_PHRASE <= len(p) <= MAX_PHRASE):
                hard.append(f"{qid} must_not 长度非法: {p!r}")
                continue
            if any(p in i for i in inc):
                hard.append(f"{qid} must_not 是 must_include 子串: {p!r}")
            if ("不" + normalize(p)) in gold_text:
                hard.append(f"{qid} must_not 会被正确否定句误伤: {p!r}")
        old = current.get(qid, {})
        old_points = len(old.get("must_include") or []) + len(old.get("any_of") or [])
        if old_points and total_points < old_points - 2:
            warn.append(f"{qid} 必答点从 {old_points} 降到 {total_points}，放宽过度？")

    missing = set(by_qid) - seen
    if missing:
        hard.append(f"缺 {len(missing)} 题: {sorted(missing)[:5]}")

    report = {
        "wp_dir": str(wp),
        "n_targets": len(targets),
        "n_pred": len(preds),
        "hard": hard,
        "warn": warn,
        "verdict": "PASS" if not hard else "FAIL",
    }
    out = wp / "outputs" / "validation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 68)
    print(f"  目标 {len(targets)} / 实收 {len(preds)}｜硬违规 {len(hard)}｜"
          f"软警告 {len(warn)}｜判定 {report['verdict']}")
    for msg in hard[:20]:
        print(f"  [HARD] {msg}")
    for msg in warn[:10]:
        print(f"  [WARN] {msg}")
    print(f"  报告 -> {out}")
    print("=" * 68)
    return 0 if not hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
