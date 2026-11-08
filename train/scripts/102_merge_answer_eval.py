"""合并答案评测集 v4（45 v3 + 24 heldout + 31 案例 = 100 题）。

    python train/scripts/102_merge_answer_eval.py

前置：101 号校验 PASS（annotation_wp_v3/outputs/answer_rubrics.jsonl）。
输出：data/eval_questions_v4_100.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = TRAIN_ROOT / "data"
DEFAULT_WP = TRAIN_ROOT / "annotation_wp_v3"
DEFAULT_OUT = DATA_DIR / "eval_questions_v4_100.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="合并答案评测集 v4")
    ap.add_argument("--wp-dir", default=str(DEFAULT_WP))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--rubrics", default=None,
                    help="rubric 文件（默认 wp/outputs/answer_rubrics.jsonl；"
                         "A1b 修订后用 revision/outputs/answer_rubrics_revised.jsonl）")
    args = ap.parse_args()
    wp = Path(args.wp_dir)
    rubrics_path = (Path(args.rubrics) if args.rubrics
                    else wp / "outputs" / "answer_rubrics.jsonl")

    v3 = load_jsonl(DATA_DIR / "eval_questions_v3.jsonl")
    held = (load_jsonl(DATA_DIR / "heldout_questions.jsonl")
            + load_jsonl(DATA_DIR / "heldout2_questions.jsonl"))
    rubrics = {r["qid"]: r["expect"] for r in load_jsonl(rubrics_path)}
    targets = {t["qid"]: t for t in load_jsonl(wp / "targets.jsonl")}

    rows: list[dict] = []
    seen: set[str] = set()
    for r in v3 + held:
        rows.append({**r, "data_origin": r.get("data_origin", "eval")})
        seen.add(r["qid"])
    for qid, t in targets.items():
        if qid not in rubrics:
            raise SystemExit(f"[FAIL] {qid} 缺 rubric，先跑 101 号校验")
        rows.append({
            "qid": qid,
            "data_origin": "case",
            "question": t["question"],
            "facts": None,
            "fact_hints": None,
            "references": [],
            "expect": rubrics[qid],
        })
        seen.add(qid)

    if len(rows) != len(seen):
        raise SystemExit(f"[FAIL] qid 重复：{len(rows)} 行 / {len(seen)} 唯一")
    if len(rows) != 100:
        raise SystemExit(f"[FAIL] 期望 100 题，实际 {len(rows)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] {out}  {len(rows)} 题")
    print(f"     来源：v3 {len(v3)} / heldout {len(held)} / case {len(targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
