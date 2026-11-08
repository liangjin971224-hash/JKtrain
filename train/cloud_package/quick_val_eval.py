"""云端 sanity 评测（非正式门禁）：val 集上 pos 是否排进 Top-5。

    python quick_val_eval.py --model ./merged --data data/val.jsonl
★ 参考水位：未训练的 bge-m3 本地实测 ≈0.78（val 集难负例密集，2026-11-07
  用 B=0 恒等合并干跑标定）。<0.50 才说明合并/数据大概率出了问题；
  0.50–0.90 之间不阻塞打包，正式结论以本地 81 号门禁为准。
"""
import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    st = SentenceTransformer(a.model, device="cpu")

    hit5, n = 0, 0
    for r in rows:
        cand_texts = list(r["pos"]) + list(r["neg"])
        gold_set = set(range(len(r["pos"])))
        qv = st.encode([r["query"]], normalize_embeddings=True)
        cv = st.encode(cand_texts, normalize_embeddings=True)
        order = np.argsort(-(qv @ cv.T)[0])
        n += 1
        if any(int(i) in gold_set for i in order[:5]):
            hit5 += 1

    r5 = hit5 / n
    flag = "OK" if r5 >= 0.50 else "疑似异常——检查合并与数据，暂缓打包"
    print(f"sanity R@5 = {r5:.4f}（{n} 条）[{flag}]，"
          f"参考水位 0.78（未训练），正式门禁在本地 81 号")


if __name__ == "__main__":
    main()
