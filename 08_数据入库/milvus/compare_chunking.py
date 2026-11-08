"""
四种切片方案对比实验

A. heading      标题切分（当前方案）
B. recursive    递归字符切分 500/80
C. parentchild  父子块 300/60（子块检索，父块返回）
D. casemulti    案例分段 + 文档标题切

评测：Recall@1/3/5、MRR、命中准确率
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, ".")
from chunking_variants import (
    BASE_DIR_STR, TRAINING_PERMISSION,
    chunk_case_split, chunk_case_whole, chunk_heading, chunk_parent_child,
    chunk_recursive, is_noise_section,
)
from common import Embedder, get_client
from pymilvus import DataType

BASE = Path(BASE_DIR_STR)
EVAL_PATH = Path("eval_set.json")

# 方案配置：每个方案产出 (collection_name, units)
# unit = dict(chunk_id, text, doc_id, title, section, source_file,
#             permission_level, category, parent_text)


def build_heading() -> list[dict]:
    units = []
    for f in sorted((BASE / "01_培训文档").glob("*.md")):
        doc_id = f.name[:4]
        perm = TRAINING_PERMISSION.get(doc_id, "internal")
        md = f.read_text(encoding="utf-8")
        title = md.splitlines()[0].lstrip("# ").strip() or f.stem
        for i, (sec, body) in enumerate(chunk_heading(md)):
            if len(body) < 30 or is_noise_section(sec):
                continue
            units.append(dict(
                chunk_id=f"{doc_id}_c{i:03d}", text=body[:11000],
                doc_id=doc_id, title=title, section=sec,
                source_file=f.name, permission_level=perm,
                category="training", parent_text="",
            ))
    for f in sorted((BASE / "02_情景案例").glob("*.md")):
        md = f.read_text(encoding="utf-8")
        for cid, ctitle, body in chunk_case_whole(md):
            units.append(dict(
                chunk_id=f"{cid}_full", text=body[:11000],
                doc_id=cid, title=ctitle, section=ctitle,
                source_file=f.name, permission_level="restricted",
                category="case", parent_text="",
            ))
    return units


def build_recursive() -> list[dict]:
    units = []
    for f in sorted((BASE / "01_培训文档").glob("*.md")):
        doc_id = f.name[:4]
        perm = TRAINING_PERMISSION.get(doc_id, "internal")
        md = f.read_text(encoding="utf-8")
        title = md.splitlines()[0].lstrip("# ").strip() or f.stem
        for i, (sec, body) in enumerate(chunk_recursive(md, 500, 80)):
            if len(body) < 30 or is_noise_section(sec):
                continue
            units.append(dict(
                chunk_id=f"{doc_id}_r{i:03d}", text=body[:11000],
                doc_id=doc_id, title=title, section=sec,
                source_file=f.name, permission_level=perm,
                category="training", parent_text="",
            ))
    for f in sorted((BASE / "02_情景案例").glob("*.md")):
        md = f.read_text(encoding="utf-8")
        for cid, ctitle, body in chunk_case_whole(md):
            units.append(dict(
                chunk_id=f"{cid}_full", text=body[:11000],
                doc_id=cid, title=ctitle, section=ctitle,
                source_file=f.name, permission_level="restricted",
                category="case", parent_text="",
            ))
    return units


def build_parentchild() -> list[dict]:
    units = []
    for f in sorted((BASE / "01_培训文档").glob("*.md")):
        doc_id = f.name[:4]
        perm = TRAINING_PERMISSION.get(doc_id, "internal")
        md = f.read_text(encoding="utf-8")
        title = md.splitlines()[0].lstrip("# ").strip() or f.stem
        for i, (sec, child, pid, parent) in enumerate(chunk_parent_child(md, 300, 60)):
            if len(child) < 30 or is_noise_section(sec):
                continue
            units.append(dict(
                chunk_id=f"{doc_id}_{pid}_c{i:03d}", text=child[:11000],
                doc_id=doc_id, title=title, section=sec,
                source_file=f.name, permission_level=perm,
                category="training", parent_text=parent[:11000],
            ))
    for f in sorted((BASE / "02_情景案例").glob("*.md")):
        md = f.read_text(encoding="utf-8")
        for cid, ctitle, body in chunk_case_whole(md):
            units.append(dict(
                chunk_id=f"{cid}_full", text=body[:11000],
                doc_id=cid, title=ctitle, section=ctitle,
                source_file=f.name, permission_level="restricted",
                category="case", parent_text="",
            ))
    return units


def build_casemulti() -> list[dict]:
    """案例分段 + 文档标题切"""
    units = []
    # 文档：同 A
    for f in sorted((BASE / "01_培训文档").glob("*.md")):
        doc_id = f.name[:4]
        perm = TRAINING_PERMISSION.get(doc_id, "internal")
        md = f.read_text(encoding="utf-8")
        title = md.splitlines()[0].lstrip("# ").strip() or f.stem
        for i, (sec, body) in enumerate(chunk_heading(md)):
            if len(body) < 30 or is_noise_section(sec):
                continue
            units.append(dict(
                chunk_id=f"{doc_id}_c{i:03d}", text=body[:11000],
                doc_id=doc_id, title=title, section=sec,
                source_file=f.name, permission_level=perm,
                category="training", parent_text="",
            ))
    # 案例：分段
    for f in sorted((BASE / "02_情景案例").glob("*.md")):
        md = f.read_text(encoding="utf-8")
        for idx, (cid, ctitle, sec, body) in enumerate(chunk_case_split(md)):
            units.append(dict(
                chunk_id=f"{cid}_s{idx:02d}", text=body[:11000],
                doc_id=cid, title=ctitle, section=f"{ctitle} · {sec}",
                source_file=f.name, permission_level="restricted",
                category="case", parent_text="",
            ))
    return units


SCHEMES = {
    "A_heading": ("eval_a_heading", build_heading),
    "B_recursive": ("eval_b_recursive", build_recursive),
    "C_parentchild": ("eval_c_parentchild", build_parentchild),
    "D_casemulti": ("eval_d_casemulti", build_casemulti),
}


# ---------------------------------------------------------------
def create_collection(client, name: str) -> None:
    if client.has_collection(name):
        client.drop_collection(name)
    s = client.create_schema(auto_id=True, enable_dynamic_field=False)
    s.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    s.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)
    s.add_field("chunk_id", DataType.VARCHAR, max_length=64)
    s.add_field("text", DataType.VARCHAR, max_length=12000)
    s.add_field("parent_text", DataType.VARCHAR, max_length=12000)
    s.add_field("doc_id", DataType.VARCHAR, max_length=32)
    s.add_field("title", DataType.VARCHAR, max_length=256)
    s.add_field("section", DataType.VARCHAR, max_length=256)
    s.add_field("source_file", DataType.VARCHAR, max_length=128)
    s.add_field("permission_level", DataType.VARCHAR, max_length=32)
    s.add_field("category", DataType.VARCHAR, max_length=32)

    ip = client.prepare_index_params()
    ip.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE",
                 params={"M": 16, "efConstruction": 200})
    client.create_collection(collection_name=name, schema=s, index_params=ip)


def ingest_scheme(client, emb, name: str, builder) -> int:
    units = builder()
    vecs = emb.encode([u["text"] for u in units])
    rows = []
    for u, v in zip(units, vecs):
        r = dict(u)
        r["vector"] = v
        rows.append(r)
    create_collection(client, name)
    for i in range(0, len(rows), 100):
        client.insert(collection_name=name, data=rows[i:i + 100])
    client.flush(name)
    return len(rows)


# ---------------------------------------------------------------
def evaluate(client, emb, collection: str, eval_data: dict,
             include_restricted: bool = True,
             extra_filter: str = "") -> dict:
    """计算 Recall@1/3/5 与 MRR。"""
    queries = eval_data["queries"]
    k_max = 5

    filters = 'permission_level in ["public", "internal", "restricted"]' if include_restricted \
        else 'permission_level in ["public", "internal"]'
    if extra_filter:
        filters = f"({filters}) and ({extra_filter})"

    r1 = r3 = r5 = 0
    mrr_sum = 0.0
    details = []

    for q in queries:
        if not include_restricted and q["intent"] == "case":
            continue

        vec = emb.encode([q["query"]])[0]
        res = client.search(
            collection_name=collection,
            data=[vec],
            limit=k_max,
            filter=filters,
            output_fields=["chunk_id", "doc_id", "title", "section", "category"],
        )
        hits = res[0] if res else []

        # 命中判定：doc_id 或 section 或 title 命中任一 label
        rank = 0
        for i, h in enumerate(hits, 1):
            e = h["entity"]
            haystack = f"{e['doc_id']} {e['title']} {e['section']}"
            if any(lab in haystack for lab in q["labels"]):
                rank = i
                break

        if rank == 1:
            r1 += 1
        if 0 < rank <= 3:
            r3 += 1
        if 0 < rank <= 5:
            r5 += 1
        if rank > 0:
            mrr_sum += 1.0 / rank

        details.append({
            "id": q["id"], "query": q["query"], "rank": rank,
            "top1": f"{hits[0]['entity']['doc_id']}|{hits[0]['entity']['section'][:20]}" if hits else "",
            "top1_score": round(hits[0]["distance"], 4) if hits else 0,
        })

    n = len(details)
    return {
        "n": n,
        "recall@1": r1 / n,
        "recall@3": r3 / n,
        "recall@5": r5 / n,
        "mrr": mrr_sum / n,
        "details": details,
    }


def main() -> None:
    eval_data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    print("=" * 78)
    print(f"四种切片方案对比 · 评测集 {len(eval_data['queries'])} 条")
    print("=" * 78)

    client = get_client()
    print("\n加载 bge-m3 ...")
    emb = Embedder()          # 用默认 max_seq_length（本次已在 common 中改为 4096）
    print(f"  完成，维度 {emb.dim}，max_seq_length={emb._model.max_seq_length}")

    results = {}
    for key, (coll, builder) in SCHEMES.items():
        print(f"\n{'=' * 78}")
        print(f"[{key}] 建库 → {coll}")
        t0 = time.perf_counter()
        n = ingest_scheme(client, emb, coll, builder)
        build_s = time.perf_counter() - t0
        print(f"  {n} 块，建库耗时 {build_s:.1f}s")

        ev = evaluate(client, emb, coll, eval_data)
        ev["chunks"] = n
        ev["build_s"] = build_s
        results[key] = ev
        print(f"  Recall@1={ev['recall@1']:.3f}  @3={ev['recall@3']:.3f}  "
              f"@5={ev['recall@5']:.3f}  MRR={ev['mrr']:.3f}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 78)
    print("汇总对比")
    print("-" * 78)
    print(f"{'方案':<16}{'块数':>6}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}{'建库(s)':>9}")
    print("-" * 78)
    for k, r in results.items():
        print(f"{k:<16}{r['chunks']:>6}{r['recall@1']:>8.3f}{r['recall@3']:>8.3f}"
              f"{r['recall@5']:>8.3f}{r['mrr']:>8.3f}{r['build_s']:>9.1f}")

    # ---------- 逐条对比 ----------
    print("\n" + "=" * 78)
    print("逐条命中位置对比（数字=排名，0=未命中）")
    print("-" * 78)
    header = f"{'ID':<6}{'查询':<30}"
    for k in results:
        header += f"{k.split('_')[0]:>6}"
    print(header)
    print("-" * 78)

    ids = [d["id"] for d in results["A_heading"]["details"]]
    for i, qid in enumerate(ids):
        q = next(x for x in eval_data["queries"] if x["id"] == qid)
        line = f"{qid:<6}{q['query'][:28]:<30}"
        for k in results:
            d = next(x for x in results[k]["details"] if x["id"] == qid)
            line += f"{d['rank']:>6}"
        print(line)

    # ---------- 保存 ----------
    out = {
        "eval_size": len(eval_data["queries"]),
        "model": "bge-m3",
        "max_seq_length": emb._model.max_seq_length,
        "results": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
    }
    Path("eval_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已保存: eval_result.json")


if __name__ == "__main__":
    main()
