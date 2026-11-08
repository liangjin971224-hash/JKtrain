"""校验案例库答案 rubric（100 号工作包的 outputs/answer_rubrics.jsonl）。

    python train/scripts/101_validate_answer_rubrics.py --wp-dir train/annotation_wp_v3

硬校验（任一失败 exit 1）：
    · 合法 JSON、字段恰好 {qid, expect}
    · qid 与 targets 一一对应、31 行
    · must_include 2~5 条，每条 2~20 字，且能在 gold 正文里找到（归一化子串）
    · any_of 每组 2~5 条；must_not_include 0~3 条
    · must_not_include 不得是某个 gold 否定句的子串（「可以X」vs「不可以X」，v3 踩过）
    · must_not_include 不得是 must_include 的子串

软校验（WARN）：
    · must_include 超过 5 条 / any_of 组过大 / 短语与 gold 仅部分匹配
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
MIN_INCLUDE, MAX_INCLUDE = 2, 5
MIN_PHRASE, MAX_PHRASE = 2, 20
MAX_ANY_OF_GROUP = 5
MAX_EXCLUDE = 3


def normalize(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in s if not ch.isspace())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="校验答案 rubric")
    ap.add_argument("--wp-dir", default=str(DEFAULT_WP))
    args = ap.parse_args()
    wp = Path(args.wp_dir)

    targets = load_jsonl(wp / "targets.jsonl")
    pred_path = wp / "outputs" / "answer_rubrics.jsonl"
    preds = load_jsonl(pred_path) if pred_path.exists() else []
    if not targets:
        print(f"[FAIL] 找不到 targets.jsonl：{wp}")
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
        if not isinstance(inc, list) or not (MIN_INCLUDE <= len(inc) <= MAX_INCLUDE):
            hard.append(f"{qid} must_include 数量 {len(inc)} 不在 {MIN_INCLUDE}~{MAX_INCLUDE}")
        gold_text = normalize("".join(c["text"] for c in by_qid[qid]["gold_chunks"]))
        for p in inc:
            if not isinstance(p, str) or not (MIN_PHRASE <= len(p) <= MAX_PHRASE):
                hard.append(f"{qid} must_include 长度非法: {p!r}")
                continue
            if normalize(p) not in gold_text:
                hard.append(f"{qid} must_include 在 gold 正文找不到: {p!r}")
        for group in any_of:
            if not isinstance(group, list) or not (2 <= len(group) <= MAX_ANY_OF_GROUP):
                hard.append(f"{qid} any_of 组非法: {group!r}")
        if not isinstance(exc, list) or len(exc) > MAX_EXCLUDE:
            hard.append(f"{qid} must_not_include 数量 {len(exc)} > {MAX_EXCLUDE}")
        for p in exc:
            if not isinstance(p, str) or not (MIN_PHRASE <= len(p) <= MAX_PHRASE):
                hard.append(f"{qid} must_not_include 长度非法: {p!r}")
                continue
            if any(p in i for i in inc):
                hard.append(f"{qid} must_not 是 must_include 子串: {p!r}")
            # v3 踩过的坑：「可以单独退」会命中「不可以单独退」
            if ("不" + normalize(p)) in gold_text:
                hard.append(
                    f"{qid} must_not 会被正确否定句误伤（gold 含「不{p}」）: {p!r}")

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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 68)
    print(f"  目标 {len(targets)} / 实收 {len(preds)}｜硬违规 {len(hard)}｜"
          f"软警告 {len(warn)}｜判定 {report['verdict']}")
    for msg in hard[:20]:
        print(f"  [HARD] {msg}")
    if len(hard) > 20:
        print(f"  ... 其余 {len(hard) - 20} 条见报告")
    print(f"  报告 -> {out}")
    print("=" * 68)
    return 0 if not hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
