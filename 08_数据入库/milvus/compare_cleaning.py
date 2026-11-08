"""
文本清洗 A/B 实验
三档清洗强度 + 同一切片方式（标题切 + 案例分段），对比召回指标

C0_none 不清洗
C1_light 去格式符（emoji/加粗/分隔线/引用元信息）
C2_deep  L1 + 表格转自然语言
C3_deep_noqa C2 + 剔除自测题/参考答案段
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, ".")

import re
from chunking_variants import (
    BASE_DIR_STR, TRAINING_PERMISSION,
    chunk_case_split, chunk_case_whole, chunk_heading,
)
from common import Embedder, get_client
from compare_chunking import create_collection, evaluate, EVAL_PATH
from text_clean import clean_light, clean_deep, strip_self_test, convert_tables

BASE = Path(BASE_DIR_STR)

LEVELS = {
    "C0_none": lambda t: t,
    "C1_light": clean_light,
    "C2_deep": clean_deep,
    "C3_deep_noqa": lambda t: strip_self_test(clean_deep(t)),
}


def build(cleaner, include_question: bool = False) -> list[dict]:
    """用指定清洗器构建 units。

    include_question=False 时不含题库——题库按 knowledge_point 字段过滤，
    不参与语义检索评估，纳入会稀释文档/案例的召回指标。
    """
    units: list[dict] = []

    # 培训文档：标题切
    for f in sorted((BASE / "01_培训文档").glob("*.md")):
        doc_id = f.name[:4]
        perm = TRAINING_PERMISSION.get(doc_id, "internal")
        md = f.read_text(encoding="utf-8")
        title = md.splitlines()[0].lstrip("# ").strip() or f.stem
        for i, (sec, body) in enumerate(chunk_heading(md)):
            if len(body) < 30:
                continue
            cleaned = cleaner(body)
            if len(cleaned) < 30:
                continue
            units.append(dict(
                chunk_id=f"{doc_id}_c{i:03d}", text=cleaned[:11000],
                doc_id=doc_id, title=title, section=sec,
                source_file=f.name, permission_level=perm,
                category="training", parent_text="",
            ))

    # 情景案例：分段
    for f in sorted((BASE / "02_情景案例").glob("*.md")):
        md = f.read_text(encoding="utf-8")
        for idx, (cid, ctitle, sec, body) in enumerate(chunk_case_split(md)):
            cleaned = cleaner(body)
            if len(cleaned) < 30:
                continue
            units.append(dict(
                chunk_id=f"{cid}_s{idx:02d}", text=cleaned[:11000],
                doc_id=cid, title=ctitle, section=f"{ctitle} · {sec}",
                source_file=f.name, permission_level="restricted",
                category="case", parent_text="",
            ))

    if not include_question:
        return units

    # 题库：每题
    q_re = re.compile(r"^#{2,3}\s+([QS]\d{3})\s*([★☆]{1,3})?\s*[（(]?[^）)\n]*[）)]?\s*$", re.M)
    for f in sorted((BASE / "03_题库").glob("*.md")):
        text = f.read_text(encoding="utf-8")
        ms = list(q_re.finditer(text))
        stage = "S1" if "S1" in f.name or "产品知识" in f.name else \
                "S2" if "S2" in f.name or "业务流程" in f.name else "MIX"
        for idx, m in enumerate(ms):
            qno = m.group(1)
            start = m.end()
            end = ms[idx + 1].start() if idx + 1 < len(ms) else len(text)
            body = text[start:end].strip()
            cleaned = cleaner(body)
            if len(cleaned) < 20:
                continue
            q_stage = stage
            if f.name.startswith("题库_补充"):
                q_stage = "S1" if int(qno[1:]) <= 80 else "S2"
            units.append(dict(
                chunk_id=qno, text=cleaned[:11000],
                doc_id=qno, title=f"{qno} {q_stage}", section=q_stage,
                source_file=f.name, permission_level="restricted",
                category="question", parent_text="",
            ))
    return units


def main() -> None:
    ev = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    client = get_client()
    print("加载 bge-m3 ...")
    emb = Embedder()
    print(f"  维度 {emb.dim}，max_seq={emb._model.max_seq_length}")
    print(f"\n评测 {len(ev['queries'])} 条\n")

    print(f"{'清洗档位':<16}{'块数':>6}{'平均字数':>9}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}")
    print("-" * 66)

    results: dict[str, dict] = {}
    for name, cleaner in LEVELS.items():
        coll = f"eval_{name.lower()}"
        units = build(cleaner)
        avg_len = sum(len(u["text"]) for u in units) / len(units)

        vecs = emb.encode([u["text"] for u in units])
        rows = []
        for u, v in zip(units, vecs):
            r = dict(u)
            r["vector"] = v
            rows.append(r)

        create_collection(client, coll)
        for i in range(0, len(rows), 100):
            client.insert(collection_name=coll, data=rows[i:i + 100])
        client.flush(coll)

        r = evaluate(client, emb, coll, ev, extra_filter='category in ["training","case"]')
        r["chunks"] = len(units)
        r["avg_len"] = avg_len
        results[name] = r

        print(f"{name:<16}{len(units):>6}{avg_len:>9.0f}"
              f"{r['recall@1']:>8.3f}{r['recall@3']:>8.3f}"
              f"{r['recall@5']:>8.3f}{r['mrr']:>8.3f}")

    # ---------- 逐条对比 ----------
    print("\n" + "=" * 78)
    print("逐条命中位置（0=未命中）")
    print("-" * 78)
    header = f"{'ID':<6}{'查询':<26}"
    for k in LEVELS:
        header += f"{k.split('_')[0]:>6}"
    print(header)
    print("-" * 78)

    base_key = "C0_none"
    for q in ev["queries"]:
        line = f"{q['id']:<6}{q['query'][:24]:<26}"
        for k in LEVELS:
            d = next((x for x in results[k]["details"] if x["id"] == q["id"]), None)
            line += f"{d['rank'] if d else 0:>6}"
        print(line)

    # ---------- 变化分析 ----------
    print("\n" + "=" * 78)
    print("相对「不清洗」的变化")
    print("-" * 78)
    b = results[base_key]
    for k, r in results.items():
        if k == base_key:
            continue
        print(f"\n{k}:")
        print(f"  R@1  {b['recall@1']:.3f} → {r['recall@1']:.3f}  "
              f"({(r['recall@1']-b['recall@1'])*100:+.1f}%)")
        print(f"  R@3  {b['recall@3']:.3f} → {r['recall@3']:.3f}  "
              f"({(r['recall@3']-b['recall@3'])*100:+.1f}%)")
        print(f"  R@5  {b['recall@5']:.3f} → {r['recall@5']:.3f}  "
              f"({(r['recall@5']-b['recall@5'])*100:+.1f}%)")
        print(f"  MRR  {b['mrr']:.3f} → {r['mrr']:.3f}  "
              f"({(r['mrr']-b['mrr'])*100:+.1f}%)")

        # 改善/恶化的条目
        better = []
        worse = []
        for q in ev["queries"]:
            d0 = next((x for x in b["details"] if x["id"] == q["id"]), None)
            d1 = next((x for x in r["details"] if x["id"] == q["id"]), None)
            if not d0 or not d1:
                continue
            r0, r1 = d0["rank"], d1["rank"]
            # rank 0 视为无穷大
            r0v = r0 if r0 > 0 else 99
            r1v = r1 if r1 > 0 else 99
            if r1v < r0v:
                better.append(f"{q['id']}({r0}→{r1})")
            elif r1v > r0v:
                worse.append(f"{q['id']}({r0}→{r1})")
        if better:
            print(f"  ↑ 改善: {', '.join(better[:8])}")
        if worse:
            print(f"  ↓ 恶化: {', '.join(worse[:8])}")

    Path("clean_result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n结果已保存: clean_result.json")


if __name__ == "__main__":
    main()
