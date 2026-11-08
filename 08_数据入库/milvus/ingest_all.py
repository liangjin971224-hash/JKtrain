"""
三个文件夹分别写入三个 Milvus collection
  01_培训文档 → health_training_docs
  02_情景案例 → health_case_library
  03_题库     → health_question_bank

模型：bge-m3（1024 维）
用法：
  python ingest_all.py              # 全部重建
  python ingest_all.py training     # 只重建培训文档
  python ingest_all.py case question
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from common import (
    BASE_DIR, Chunk, Embedder, clean_text, get_client,
    insert_chunks, print_collection_stat, recreate_collection, split_by_heading,
)

# ---------------------------------------------------------------
# 三个 collection 的配置
# ---------------------------------------------------------------
COLLECTIONS = {
    "training": {
        "name": "health_training_docs",
        "source_dir": "01_培训文档",
        "category": "training",
        "desc": "培训文档 15 篇",
    },
    "case": {
        "name": "health_case_library",
        "source_dir": "02_情景案例",
        "category": "case",
        "desc": "情景演练案例 16 个",
    },
    "question": {
        "name": "health_question_bank",
        "source_dir": "03_题库",
        "category": "question",
        "desc": "题库 140 道",
    },
}

# 文档权限映射（依据 D001-D015 的文档信息块）
TRAINING_PERMISSION = {
    "D001": "public",
    "D002": "internal",
    "D003": "public",
    "D004": "public",
    "D005": "internal",     # 禁用词清单，受限
    "D006": "public",
    "D007": "public",
    "D008": "public",
    "D009": "internal",
    "D010": "internal",
    "D011": "internal",
    "D012": "internal",
    "D013": "internal",
    "D014": "internal",
    "D015": "internal",
}


# ---------------------------------------------------------------
# 三个 folder 各自的解析逻辑
# ---------------------------------------------------------------
def parse_training() -> list[Chunk]:
    """培训文档：按标题切块，doc_id 从文件名提取 D0xx。"""
    folder = BASE_DIR / COLLECTIONS["training"]["source_dir"]
    chunks: list[Chunk] = []

    for f in sorted(folder.glob("*.md")):
        m = re.match(r"(D\d{3})_", f.name)
        doc_id = m.group(1) if m else f.stem[:4]
        perm = TRAINING_PERMISSION.get(doc_id, "internal")

        text = f.read_text(encoding="utf-8")
        # 文档主标题
        first_line = text.splitlines()[0] if text else ""
        title = first_line.lstrip("# ").strip() or f.stem

        sections = split_by_heading(text)
        for i, (section, body) in enumerate(sections):
            body = clean_text(body)
            if len(body) < 30:      # 太短的片段（如只有一行）跳过
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_c{i:03d}",
                text=body[:11000],
                source_dir="01_培训文档",
                source_file=f.name,
                doc_id=doc_id,
                title=title,
                category="training",
                permission_level=perm,
                review_status="approved",
                data_origin="real",
                section=section,
                chunk_index=i,
            ))
    return chunks


def parse_cases() -> list[Chunk]:
    """情景案例：每个案例一块（含设定/脚本/要素/答案）。"""
    folder = BASE_DIR / COLLECTIONS["case"]["source_dir"]
    chunks: list[Chunk] = []

    for f in sorted(folder.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        # 按 "# C01" / "## 案例 C01" 切分单个案例
        pattern = re.compile(r"^(?:#\s+|\#{2}\s+案例\s*)(C\d{2})[：:\s]*(.*)$", re.M)
        matches = list(pattern.finditer(text))

        if not matches:
            # 无案例编号的文件整体切成块
            for i, (section, body) in enumerate(split_by_heading(text, max_chars=900)):
                body = clean_text(body)
                if len(body) < 50:
                    continue
                chunks.append(Chunk(
                    chunk_id=f"CASEX_c{i:03d}",
                    text=body[:11000],
                    source_dir="02_情景案例",
                    source_file=f.name,
                    doc_id="CASEX",
                    title=f.stem,
                    category="case",
                    permission_level="internal",
                    review_status="approved",
                    data_origin="synthetic",
                    section=section,
                    chunk_index=i,
                ))
            continue

        for idx, m in enumerate(matches):
            case_id = m.group(1)
            case_title = m.group(2).strip().rstrip("：:")
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = clean_text(text[start:end])

            # 案例含参考答案 → restricted（学员不可检索）
            chunks.append(Chunk(
                chunk_id=f"{case_id}_full",
                text=body[:11000],
                source_dir="02_情景案例",
                source_file=f.name,
                doc_id=case_id,
                title=case_title or case_id,
                category="case",
                permission_level="restricted",
                review_status="approved",
                data_origin="synthetic",
                section=case_title,
                chunk_index=0,
            ))
    return chunks


def parse_questions() -> list[Chunk]:
    """题库：每题一块，extra 带题型/难度/答案/知识点。"""
    folder = BASE_DIR / COLLECTIONS["question"]["source_dir"]
    chunks: list[Chunk] = []

    # 题号 + 难度标记 + 可选括号注释
    # 兼容：## Q001 ★☆☆  /  ## Q003 ★★☆（关键题）  /  ## S001 ★★☆（多选）
    q_pattern = re.compile(
        r"^#{2,3}\s+([QS]\d{3})\s*([★☆]{1,3})?\s*[（(]?[^）)\n]*[）)]?\s*$",
        re.M,
    )

    for f in sorted(folder.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        matches = list(q_pattern.finditer(text))

        # 文件级阶段判断
        stage = "S1" if "S1" in f.name or "产品知识" in f.name else \
                "S2" if "S2" in f.name or "业务流程" in f.name else "MIX"

        for idx, m in enumerate(matches):
            q_no = m.group(1)
            stars = m.group(2) or ""
            difficulty = stars.count("★")
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = clean_text(text[start:end])

            if len(body) < 20:
                continue

            # 从正文里抽答案
            ans_m = re.search(r"\*\*答案[：:]\s*([A-F]+)\*\*", body)
            answer = ans_m.group(1) if ans_m else ""

            # 判断题目归属阶段（补充题按题号段）
            q_stage = stage
            if f.name.startswith("题库_补充"):
                n = int(q_no[1:])
                q_stage = "S1" if n <= 80 else "S2"

            chunks.append(Chunk(
                chunk_id=f"{q_no}",
                text=body[:11000],
                source_dir="03_题库",
                source_file=f.name,
                doc_id=q_no,
                title=f"{q_no} {q_stage}",
                category="question",
                permission_level="restricted",      # 题库学员不可见
                review_status="approved",
                data_origin="synthetic",
                section=q_stage,
                chunk_index=idx,
                extra=json.dumps({
                    "stage": q_stage,
                    "difficulty": difficulty,
                    "answer": answer,
                }, ensure_ascii=False),
            ))
    return chunks


PARSERS = {
    "training": parse_training,
    "case": parse_cases,
    "question": parse_questions,
}


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------
def ingest(keys: list[str]) -> None:
    print("=" * 74)
    print("Milvus 入库：三个文件夹 → 三个 collection")
    print("=" * 74)

    # 1. 解析
    all_chunks: dict[str, list[Chunk]] = {}
    for k in keys:
        chunks = PARSERS[k]()
        all_chunks[k] = chunks
        print(f"\n[{k}] 解析出 {len(chunks)} 个 chunk")

        by_file: dict[str, int] = {}
        for c in chunks:
            by_file[c.source_file] = by_file.get(c.source_file, 0) + 1
        for fn, cnt in sorted(by_file.items())[:20]:
            print(f"    {fn[:46]:<48} {cnt:>3}")

    total_chunks = sum(len(v) for v in all_chunks.values())
    print(f"\n合计 {total_chunks} 个 chunk")

    # 2. 加载模型
    print("\n加载 bge-m3 ...")
    t0 = time.perf_counter()
    emb = Embedder()
    print(f"  加载完成 {emb.load_seconds:.2f}s，维度 {emb.dim}")

    # 3. 向量化
    print("\n向量化 ...")
    t1 = time.perf_counter()
    for k, chunks in all_chunks.items():
        if not chunks:
            continue
        vecs = emb.encode([c.text for c in chunks])
        for c, v in zip(chunks, vecs):
            c.embedding = v
        print(f"  [{k}] {len(vecs)} 条完成")
    print(f"  向量化总耗时 {time.perf_counter() - t1:.1f}s")

    # 4. 写入 Milvus
    print("\n写入 Milvus ...")
    client = get_client()
    for k in keys:
        cfg = COLLECTIONS[k]
        chunks = all_chunks[k]
        print(f"\n[{k}] → {cfg['name']}")
        recreate_collection(client, cfg["name"], cfg["desc"])
        n = insert_chunks(client, cfg["name"], chunks)
        print(f"  写入 {n} 条")

    # 5. 校验
    print("\n" + "=" * 74)
    print("校验")
    print("-" * 74)
    for k in keys:
        print_collection_stat(client, COLLECTIONS[k]["name"])

    print(f"\n总耗时 {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        keys = list(COLLECTIONS.keys())
    else:
        keys = [a for a in args if a in COLLECTIONS]
        if not keys:
            print(f"无效参数。可选: {list(COLLECTIONS.keys())}")
            sys.exit(1)
    ingest(keys)
