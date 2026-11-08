"""校验多轮评测 rubric（105 号工作包的输出）。"""
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
DEFAULT_WP = TRAIN_ROOT / "annotation_wp_v4"
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
    ap = argparse.ArgumentParser(description="校验多轮 rubric")
    ap.add_argument("--wp-dir", default=str(DEFAULT_WP))
    args = ap.parse_args()
    wp = Path(args.wp_dir)
    targets = load_jsonl(wp / "targets.jsonl")
    preds = load_jsonl(wp / "outputs" / "multi_turn_rubrics.jsonl")
    if not targets:
        print(f"[FAIL] 找不到 targets.jsonl：{wp}")
        return 1
    if not preds:
        print(f"[FAIL] 找不到产出：{wp / 'outputs' / 'multi_turn_rubrics.jsonl'}")
        return 1

    by_id = {t["group_id"]: t for t in targets}
    hard: list[str] = []
    warn: list[str] = []
    seen: set[str] = set()

    for r in preds:
        gid = r.get("group_id")
        if gid not in by_id:
            hard.append(f"未知 group_id: {gid}")
            continue
        if gid in seen:
            hard.append(f"group_id 重复: {gid}")
        seen.add(gid)
        expect = r.get("expect")
        if not isinstance(expect, dict):
            hard.append(f"{gid} expect 不是对象")
            continue
        extra = set(expect) - {"must_include", "any_of", "must_not_include"}
        if extra:
            hard.append(f"{gid} expect 多字段: {sorted(extra)}")
        inc = expect.get("must_include") or []
        any_of = expect.get("any_of") or []
        exc = expect.get("must_not_include") or []
        if not (2 <= len(inc) <= 5):
            hard.append(f"{gid} must_include 数量 {len(inc)} 不在 2~5")
        gold = normalize("".join(c["text"] for c in by_id[gid]["reference_chunks"]))
        for p in inc:
            if not isinstance(p, str) or not (MIN_PHRASE <= len(p) <= MAX_PHRASE):
                hard.append(f"{gid} must_include 长度非法: {p!r}")
            elif normalize(p) not in gold:
                hard.append(f"{gid} must_include 在参考块找不到: {p!r}")
        for group in any_of:
            if not isinstance(group, list) or not (2 <= len(group) <= 5):
                hard.append(f"{gid} any_of 组非法: {group!r}")
                continue
            if not any(isinstance(g, str) and normalize(g) in gold for g in group):
                hard.append(f"{gid} any_of 组在参考块找不到任一变体: {group!r}")
        if not isinstance(exc, list) or len(exc) > 3:
            hard.append(f"{gid} must_not 数量非法")
        for p in exc:
            if not isinstance(p, str) or not (MIN_PHRASE <= len(p) <= MAX_PHRASE):
                hard.append(f"{gid} must_not 长度非法: {p!r}")
                continue
            if any(p in i for i in inc):
                hard.append(f"{gid} must_not 是 must_include 子串: {p!r}")
            if ("不" + normalize(p)) in gold:
                hard.append(f"{gid} must_not 会被否定句误伤: {p!r}")

    missing = set(by_id) - seen
    if missing:
        hard.append(f"缺 {len(missing)} 组: {sorted(missing)}")

    report = {"n_targets": len(targets), "n_pred": len(preds),
              "hard": hard, "warn": warn,
              "verdict": "PASS" if not hard else "FAIL"}
    out = wp / "outputs" / "validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 68)
    print(f"  目标 {len(targets)} / 实收 {len(preds)}｜硬违规 {len(hard)}｜"
          f"判定 {report['verdict']}")
    for msg in hard[:20]:
        print(f"  [HARD] {msg}")
    print(f"  报告 -> {out}")
    print("=" * 68)
    return 0 if not hard else 1


if __name__ == "__main__":
    raise SystemExit(main())
