"""构造 embedding 微调数据：正例对 + 难负例 → embed_pairs_v1.jsonl。

    python scripts/86_build_embed_pairs.py
    python scripts/86_build_embed_pairs.py --hard-negatives data/hard_negatives_v1.jsonl

输入：
    retrieval_eval_v1.jsonl（或 v0 临时集）   —— query ↔ gold_chunk
    --hard-negatives（87 号产出，可选）        —— {qid, chunk_id}
    chunks/*.jsonl                             —— chunk_id → text

输出：data/embed_pairs_v1.jsonl
    {"qid", "query", "pos": [chunk text...], "neg": [chunk text...], "split"}
    split 按 qid 确定性哈希 8:2（val 与 train 永不重叠，重跑不变）。

★ 正例文本直接用 chunk 原文（与线上被检索的正文一致），
  不做任何改写——embedding 学的正是「问法 ↔ 线上真实正文」的映射。
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
DATA_DIR = TRAIN_ROOT / "data"
WP_DIR = TRAIN_ROOT / "annotation_wp_v1"

OUT_PATH = DATA_DIR / "embed_pairs_v1.jsonl"
MIN_VAL_ROWS = 10


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_split(qid: str) -> str:
    """确定性 8:2 切分（与随机种子无关，重跑绝对稳定）。"""
    return "train" if (hash(qid) % 10) < 8 else "val"


def main() -> int:
    parser = argparse.ArgumentParser(description="构造 embedding 微调对")
    parser.add_argument("--eval", default=None,
                        help="评测集（默认 v1，缺则回退 v0 临时集）")
    parser.add_argument("--hard-negatives", default=None)
    args = parser.parse_args()

    eval_path = (Path(args.eval) if args.eval else
                 (DATA_DIR / "retrieval_eval_v1.jsonl"
                  if (DATA_DIR / "retrieval_eval_v1.jsonl").exists()
                  else DATA_DIR / "retrieval_eval_v0_provisional.jsonl"))
    queries = load_jsonl(eval_path)
    if not queries:
        raise SystemExit(f"[FAIL] 评测集为空：{eval_path}")

    chunks_by_id: dict[str, dict] = {}
    for coll in ("health_training_docs", "health_case_library",
                 "health_question_bank", "health_self_test"):
        for c in load_jsonl(WP_DIR / "chunks" / f"{coll}.jsonl"):
            chunks_by_id[c["chunk_id"]] = c

    negs_by_qid: dict[str, list[str]] = {}
    if args.hard_negatives:
        for n in load_jsonl(Path(args.hard_negatives)):
            negs_by_qid.setdefault(n["qid"], []).append(n["chunk_id"])

    print("=" * 68)
    print(f"  构造 embed pairs → {OUT_PATH}")
    print(f"  评测集: {eval_path}（{len(queries)} 题）")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 68)

    rows, skipped = [], 0
    for q in queries:
        pos_texts = [chunks_by_id[c]["text"] for c in q["gold_chunk_ids"]
                     if c in chunks_by_id and chunks_by_id[c]["text"].strip()]
        if not pos_texts:
            skipped += 1
            continue
        neg_ids = negs_by_qid.get(q["qid"], [])
        neg_texts = [chunks_by_id[c]["text"] for c in neg_ids
                     if c in chunks_by_id
                     and c not in set(q["gold_chunk_ids"])
                     and chunks_by_id[c]["text"].strip()]
        rows.append({
            "qid": q["qid"], "query": q["question"],
            "pos": pos_texts, "neg": neg_texts,
            "split": stable_split(q["qid"]),
        })

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = len(rows) - n_train
    n_with_neg = sum(1 for r in rows if r["neg"])
    OUT_PATH.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")

    print(f"  [OK] {len(rows)} 对（train {n_train} / val {n_val}），"
          f"带难负例 {n_with_neg}，跳过(无有效gold) {skipped}")
    if n_val < MIN_VAL_ROWS:
        print(f"  [WARN] val 仅 {n_val} 条（<{MIN_VAL_ROWS}），"
              "标注合入后重跑本脚本扩充。")
    print(f"  云端训练配置见 train/configs/embed_lora_v1.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
