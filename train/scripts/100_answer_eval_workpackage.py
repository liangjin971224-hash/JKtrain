"""生成答案级评测工作包 v3：31 道案例题补 rubric（凑 100 题）。

    python train/scripts/100_answer_eval_workpackage.py

输入：
    data/eval_questions_v3.jsonl        45 题（已有 rubric）
    data/heldout_questions.jsonl        12 题（已有 rubric）
    data/heldout2_questions.jsonl       12 题（已有 rubric）
    data/retrieval_eval_v1_remap.jsonl  案例库 34 题（gold chunk）
    Milvus health_case_library          取 gold 块正文

输出（annotation_wp_v3/）：
    targets.jsonl                       31 题 + gold 块正文
    ANNOTATION_BRIEF_ANSWER_RUBRIC.md   给 Codex 的任务书
    manifest.json                       数量/来源/生成时间

★ 只读脚本：不写库、不改应用代码。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAIN_ROOT.parent
PROJECT_CODE = PROJECT_ROOT / "项目代码"
DATA_DIR = TRAIN_ROOT / "data"
DEFAULT_OUT = TRAIN_ROOT / "annotation_wp_v3"

EVAL_V3 = DATA_DIR / "eval_questions_v3.jsonl"
HELDOUT = (DATA_DIR / "heldout_questions.jsonl", DATA_DIR / "heldout2_questions.jsonl")
REMAP = DATA_DIR / "retrieval_eval_v1_remap.jsonl"
CASE_COLLECTION = "health_case_library"
TARGET_CASE_QUESTIONS = 31

BRIEF = """# 答案评测 rubric 标注任务书（案例库，v3）

> 由 `100_answer_eval_workpackage.py` 于 {generated_at} 生成。
> 执行者：Codex。产出写入本目录 `outputs/answer_rubrics.jsonl`，逐行合法 JSON，UTF-8。

## 任务

输入 `targets.jsonl`（31 行）：每行一道案例题 + 它的 gold 知识块正文。
为**每道题**写一份机器可判的评分要点，写入：

    {{"qid": "...", "expect": {{"must_include": [...], "any_of": [...], "must_not_include": [...]}}}}

字段说明：

- `must_include`：回答**必须包含**的短语（2~5 个）。必须是 gold 正文里能核实的事实
  （编号、数值、术语、步骤名），不要写整句。
- `any_of`：同义表达分组；每组至少命中一个即算通过。用于「遵医嘱／听医生／由医生决定」
  这类等价说法。没有同义表达就给空数组。
- `must_not_include`：回答**不得出现**的短语（0~3 个）。只写明确的错误表述。

## 硬性规则

1. 每个 `must_include` 必须是 **gold 正文可核实的短语**，长度 2~20 字；不要写
   「要回答得准确」这种不可判的句子。
2. `must_not_include` **不能是某个正确否定句的子串**。
   反例（v3 踩过）：「可以单独退」会命中正确回答「**不**可以单独退」。
   如果必须写，写成完整错误句（如「赠品可以单独退换，无需按促销规则」），
   或者改用 `must_include` 约束正确说法。
3. 不要写跨产品对比、不要编造 gold 里没有的数字/结论。
4. 每题 `must_include` 2~5 条；`any_of` 每组 2~5 个同义短语；`must_not_include` 0~3 条。
5. 输出 31 行，`qid` 与 targets 一一对应，不多不少。
6. 不要修改 targets.jsonl、任务书或任何脚本。

## 交付后自检

```bash
项目代码/.venv/Scripts/python.exe train/scripts/101_validate_answer_rubrics.py --wp-dir train/annotation_wp_v3
```

要求硬违规 0、判定 PASS；有 HARD 按提示逐条修，改完重跑。
"""


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="生成答案评测工作包 v3")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    v3 = load_jsonl(EVAL_V3)
    held = [r for p in HELDOUT for r in load_jsonl(p)]
    existing_qids = {r["qid"] for r in v3} | {r["qid"] for r in held}
    print(f"已有 rubric：v3 {len(v3)} + heldout {len(held)} = "
          f"{len(existing_qids)}")

    remap = {r["qid"]: r for r in load_jsonl(REMAP)}
    case_rows = [r for r in remap.values()
                 if r.get("collection") == CASE_COLLECTION]
    case_rows.sort(key=lambda r: r["qid"])
    selected = case_rows[:TARGET_CASE_QUESTIONS]
    if len(selected) < TARGET_CASE_QUESTIONS:
        raise SystemExit(
            f"[FAIL] 案例题不足：{len(selected)} < {TARGET_CASE_QUESTIONS}")

    # 取 gold 块正文（与线上一致）
    sys.path.insert(0, str(PROJECT_CODE))
    from config.assembly import build_container
    from config.settings import Settings

    container = build_container(Settings())
    if container.vector is None:
        raise SystemExit("[FAIL] 向量端口不可用")

    all_ids = sorted({cid for r in selected for cid in r["gold_chunk_ids"]})
    chunks = container.vector.get_chunks_by_ids(
        all_ids, collection=CASE_COLLECTION)
    by_id = {c.chunk_id: c for c in chunks}
    missing = [cid for cid in all_ids if cid not in by_id]
    if missing:
        raise SystemExit(f"[FAIL] 取不到 gold 块: {missing}")

    targets = []
    for r in selected:
        targets.append({
            "qid": r["qid"],
            "question": r["question"],
            "collection": r["collection"],
            "allowed_permissions": r.get("allowed_permissions") or [],
            "gold_chunks": [
                {
                    "chunk_id": cid,
                    "title": by_id[cid].title,
                    "section": by_id[cid].section,
                    "text": by_id[cid].text,
                }
                for cid in r["gold_chunk_ids"]
            ],
        })

    with (out / "targets.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for t in targets:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    (out / "ANNOTATION_BRIEF_ANSWER_RUBRIC.md").write_text(
        BRIEF.format(generated_at=datetime.now(timezone.utc).astimezone()
                     .isoformat(timespec="seconds")),
        encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "purpose": "answer-level rubric annotation for case library",
        "n_targets": len(targets),
        "n_existing_rubrics": len(existing_qids),
        "n_total_after_merge": len(existing_qids) + len(targets),
        "sources": {
            "eval_v3": str(EVAL_V3),
            "heldout": [str(p) for p in HELDOUT],
            "remap": str(REMAP),
        },
        "generated_at": datetime.now(timezone.utc).astimezone()
        .isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 68)
    print(f"  工作包 -> {out}")
    print(f"  案例题 {len(targets)} 道；合并后答案评测 {len(existing_qids) + len(targets)} 题")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
