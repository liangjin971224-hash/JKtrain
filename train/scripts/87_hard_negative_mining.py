"""难负例挖掘：用当前模型检索，把「排名靠前但不是 gold」的块记为难负例。

    python scripts/87_hard_negative_mining.py
    python scripts/87_hard_negative_mining.py --top-n 10 --limit 100   # 试跑

输入：评测集（v1 或 v0）+ chunks（gold 归属）
输出：data/hard_negatives_v1.jsonl   {"qid", "chunk_id", "rank", "score"}

★ 为什么难负例比随机负例有效一个量级：
  随机负例模型一眼就能认出来，梯度≈0；
  「当前模型以为相关」的负例才逼着模型学出真正的边界。
  这里 Top-N 里非 gold 的块，就是当前模型最容易犯的错——
  微调后它们必须被压下去（87 号在微调前后各跑一次可对比）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = TRAIN_ROOT / "data"
WP_DIR = TRAIN_ROOT / "annotation_wp_v1"

OUT_PATH = DATA_DIR / "hard_negatives_v1.jsonl"


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


def main() -> int:
    parser = argparse.ArgumentParser(description="难负例挖掘")
    parser.add_argument("--eval", default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    parser.add_argument("--embed-model-dir",
                        default=str(TRAIN_ROOT.parent / "models" / "bge-m3"))
    args = parser.parse_args()

    eval_path = (Path(args.eval) if args.eval else
                 (DATA_DIR / "retrieval_eval_v1.jsonl"
                  if (DATA_DIR / "retrieval_eval_v1.jsonl").exists()
                  else DATA_DIR / "retrieval_eval_v0_provisional.jsonl"))
    queries = load_jsonl(eval_path)
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        raise SystemExit(f"[FAIL] 评测集为空：{eval_path}")

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import sentence_transformers as st
    from pymilvus import MilvusClient

    print("=" * 68)
    print(f"  难负例挖掘 → {OUT_PATH}（Top-{args.top_n} 非gold）")
    print(f"  评测集: {eval_path}（{len(queries)} 题）")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 68)

    print(f"  加载 bge-m3（CPU）: {args.embed_model_dir}")
    model = st.SentenceTransformer(args.embed_model_dir, device="cpu")
    model.max_seq_length = 4096
    client = MilvusClient(uri=args.milvus_uri)

    texts = [q["question"] for q in queries]
    t0 = time.perf_counter()
    vectors = model.encode(texts, batch_size=8, normalize_embeddings=True,
                           show_progress_bar=False)
    print(f"  编码完成 {time.perf_counter() - t0:.1f}s")

    rows, total_neg = [], 0
    for q, vec in zip(queries, vectors):
        gold = set(q["gold_chunk_ids"])
        hits = client.search(
            collection_name=q["collection"],
            data=[vec.tolist()],
            limit=args.top_n,
            output_fields=["chunk_id"],
        )[0]
        for rank, h in enumerate(hits, 1):
            cid = h["entity"].get("chunk_id", "")
            if cid and cid not in gold:
                rows.append({"qid": q["qid"], "chunk_id": cid,
                             "rank": rank, "score": float(h["distance"])})
                total_neg += 1

    OUT_PATH.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8")
    print(f"  [OK] {len(queries)} 题 → 难负例 {total_neg} 条"
          f"（均值 {total_neg / max(1, len(queries)):.1f}/题）")
    print("  下一步：python scripts/86_build_embed_pairs.py "
          "--hard-negatives data/hard_negatives_v1.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
