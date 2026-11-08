"""按 expect 给评测结果打分。

    python scripts/60_score_eval.py \
        --eval data/eval_questions_v3.jsonl \
        --pred /mnt/workspace/output/adapter_eval_v4.jsonl

pred 就是 20_baseline.py / 50_compare.py 的输出（含 qid 与 answer）。

★ 判定口径（与审核报告一致）：
  must_include      —— 归一化后做**子串**匹配。归一化含：NFKC、去空白、
                       统一破折号、克→g、％→%、全角括号转半角。
  must_not_include  —— 同样归一化后子串匹配。**必须写成短语**，
                       不要写「护心」这类裸词，否则会误伤
                       「资料没有批准保护心脏的功能」这种正确表述。

★ 这是个粗筛工具，不是语义判定器：
  它能稳定抓住「该说的没说」「不该出现的编造出现了」，
  但抓不住「说反了」「数字配错了对象」。最终仍需人工判读。
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

_DASHES = "–—−－~〜"


def normalize(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    for d in _DASHES:
        s = s.replace(d, "-")
    s = s.replace("克", "g")
    s = "".join(ch for ch in s if not ch.isspace())
    return s


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def score_one(item: dict, answer: str) -> dict:
    expect = item.get("expect") or {}
    text = normalize(answer)
    missed = [m for m in expect.get("must_include", []) if normalize(m) not in text]
    # any_of: 每组至少命中一个；用于「结论由医生出具」这类同义表达
    for group in expect.get("any_of", []):
        if not any(normalize(g) in text for g in group):
            missed.append("／".join(group))
    hit_bad = [m for m in expect.get("must_not_include", []) if normalize(m) in text]
    return {
        "qid": item.get("qid"),
        "passed": not missed and not hit_bad,
        "missed": missed,
        "forbidden_hit": hit_bad,
        "answer_chars": len(answer or ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="逐题明细写这里（可选）")
    ap.add_argument("--show-fail", action="store_true", default=True)
    args = ap.parse_args()

    items = {i["qid"]: i for i in load_jsonl(args.eval)}
    preds = {p["qid"]: p for p in load_jsonl(args.pred)}

    results: list[dict] = []
    missing = [q for q in items if q not in preds]
    for qid, item in items.items():
        pred = preds.get(qid)
        if pred is None:
            continue
        rec = score_one(item, pred.get("answer") or pred.get("output") or "")
        rec["question"] = item["question"]
        results.append(rec)

    passed = sum(r["passed"] for r in results)
    print(f"评测 {len(results)} 题 | 通过 {passed} | 未通过 {len(results) - passed}")
    if missing:
        print(f"⚠️ 预测里缺这些题：{missing}")

    if args.show_fail:
        for r in results:
            if r["passed"]:
                continue
            print(f"\n--- {r['qid']}  {r['question']}")
            if r["missed"]:
                print(f"    缺少要点: {r['missed']}")
            if r["forbidden_hit"]:
                print(f"    命中禁用: {r['forbidden_hit']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="\n") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n明细 -> {args.out}")


if __name__ == "__main__":
    main()
