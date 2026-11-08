"""Milvus 定稿重建（V2）。

按两轮实验的最终结论重建 4 个 collection：

  health_training_docs   培训正文   C3 清洗 + 标题切 + 剔「学习目标」
  health_case_library    情景案例   L1 清洗 + 按 一、二、三… 分段
  health_question_bank   题库       L1 清洗 + 每题一块 + 知识点
  health_self_test       自测题/答案 从培训文档分流，restricted（仅主管）

──────────────────────────────────────────────────────────────────────
与 V1 的差异（每一条都有实测依据，见 重建说明_V2.md）
──────────────────────────────────────────────────────────────────────
1. 案例分段正则从写死 `###` 改为 `#{2,4}`
   V1 只有 C01/C04 分段，其余 14 个案例退化成整块 → 实际 16 块。
   V2 正确分成 117 块。演练定位 Top1 从 7/8 → 8/8。

2. 自测题/参考答案块从培训库剥离，进 health_self_test
   V1 有 12 个答案块标成 public，学员能直接搜到答案（违反 FR-050）。
   V2 全部 restricted，且培训库 R@3/R@5/MRR 同时提升。

3. 题库补 knowledge_point / kp_name 字段
   源文件本来就有 `# 一、KP_PROD_CATEGORY 产品类别辨识` 标注，
   V1 脚本没提取。V2 提取，140/140 覆盖。

4. 新增 question_type / is_multi 字段
   组卷需要区分单选/多选/判断。

用法：
  python build_collections.py            # 重建全部
  python build_collections.py training case
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, ".")
from chunking_variants import (
    BASE_DIR_STR, TRAINING_PERMISSION,
    chunk_case_split, chunk_heading,
    is_meta_section, is_qa_section,
)
from common import Embedder, get_client
from text_clean import clean_deep, clean_light

BASE = Path(BASE_DIR_STR)
EMBED_DIM = 1024

COLLECTIONS = {
    "training": "health_training_docs",
    "case": "health_case_library",
    "question": "health_question_bank",
    "self_test": "health_self_test",
}


# ===================================================================
# Schema
# ===================================================================
def create_collection(client, name: str, with_kp: bool = False) -> None:
    """建 collection。with_kp=True 时追加题库专用字段。"""
    from pymilvus import DataType

    if client.has_collection(name):
        client.drop_collection(name)

    s = client.create_schema(auto_id=True, enable_dynamic_field=False)
    s.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    s.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBED_DIM)

    s.add_field("chunk_id", DataType.VARCHAR, max_length=64)
    s.add_field("text", DataType.VARCHAR, max_length=12000)
    s.add_field("source_dir", DataType.VARCHAR, max_length=64)
    s.add_field("source_file", DataType.VARCHAR, max_length=128)
    s.add_field("doc_id", DataType.VARCHAR, max_length=32)
    s.add_field("title", DataType.VARCHAR, max_length=256)
    s.add_field("category", DataType.VARCHAR, max_length=32)
    s.add_field("permission_level", DataType.VARCHAR, max_length=32)
    s.add_field("review_status", DataType.VARCHAR, max_length=32)
    s.add_field("data_origin", DataType.VARCHAR, max_length=32)
    s.add_field("section", DataType.VARCHAR, max_length=256)
    s.add_field("chunk_index", DataType.INT32)
    s.add_field("extra", DataType.VARCHAR, max_length=2000)

    if with_kp:
        s.add_field("knowledge_point", DataType.VARCHAR, max_length=64)
        s.add_field("kp_name", DataType.VARCHAR, max_length=128)
        s.add_field("question_type", DataType.VARCHAR, max_length=16)
        s.add_field("difficulty", DataType.INT32)

    ip = client.prepare_index_params()
    ip.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE",
                 params={"M": 16, "efConstruction": 200})
    # 权限过滤 / 知识点过滤是高频 exact-match，建标量索引
    scl_fields = ["permission_level", "category", "doc_id"]
    if with_kp:
        scl_fields += ["knowledge_point"]
    for f in scl_fields:
        ip.add_index(field_name=f, index_type="INVERTED")

    client.create_collection(collection_name=name, schema=s, index_params=ip)


# ===================================================================
# 解析
# ===================================================================
def parse_training() -> tuple[list[dict], list[dict]]:
    """培训文档 → (正文块, 自测题/答案块)。

    正文块：C3 清洗（去格式 + 表格转自然语言 + 去文档头元信息），
            剔掉「学习目标」这类纯元信息章节。
    答案块：「自测题 / 参考答案」章节分离出来，进 health_self_test。
    """
    body_units: list[dict] = []
    qa_units: list[dict] = []

    for f in sorted((BASE / "01_培训文档").glob("*.md")):
        doc_id = f.name[:4]
        perm = TRAINING_PERMISSION.get(doc_id, "internal")
        md = f.read_text(encoding="utf-8")
        title = md.splitlines()[0].lstrip("# ").strip() or f.stem

        for i, (sec, raw) in enumerate(chunk_heading(md)):
            if is_meta_section(sec):
                continue

            is_qa = is_qa_section(sec)
            # 答案块保留原文结构（学员练题要看选项和解析），只做 L1
            text = clean_light(raw) if is_qa else clean_deep(raw)
            if len(text) < 30:
                continue

            unit = dict(
                chunk_id=f"{doc_id}_{'qa' if is_qa else 'c'}{i:03d}",
                text=text[:11000],
                source_dir="01_培训文档",
                source_file=f.name,
                doc_id=doc_id,
                title=title,
                category="self_test" if is_qa else "training",
                # ★ 关键：答案块一律 restricted，绝不继承文档的 public
                permission_level="restricted" if is_qa else perm,
                review_status="approved",
                data_origin="real",
                section=sec,
                chunk_index=i,
                extra=json.dumps({"doc_permission": perm}, ensure_ascii=False),
            )
            (qa_units if is_qa else body_units).append(unit)

    return body_units, qa_units


def parse_cases() -> list[dict]:
    """情景案例 → 按 一、二、三… 分段。

    参考答案作为独立块保留（category=case，section 含「参考答案」），
    P5 演练评阅需要它。共 16 个案例 → 约 117 块。
    """
    units: list[dict] = []
    for f in sorted((BASE / "02_情景案例").glob("*.md")):
        md = f.read_text(encoding="utf-8")
        for idx, (cid, ctitle, sec, body) in enumerate(chunk_case_split(md)):
            text = clean_light(body)
            if len(text) < 30:
                continue
            kind = "answer" if "参考答案" in sec else \
                   "rubric" if ("评分" in sec or "违禁" in sec) else "scene"
            units.append(dict(
                chunk_id=f"{cid}_s{idx:02d}",
                text=text[:11000],
                source_dir="02_情景案例",
                source_file=f.name,
                doc_id=cid,
                title=ctitle,
                category="case",
                permission_level="restricted",
                review_status="approved",
                data_origin="synthetic",
                section=f"{ctitle} · {sec}",
                chunk_index=idx,
                extra=json.dumps({"section_type": kind}, ensure_ascii=False),
            ))
    return units


# KP 标题：`# 一、KP_PROD_CATEGORY 产品类别辨识（12 道）`
_KP_RE = re.compile(
    r"^#\s+[一二三四五六七八九十]+、\s*(KP_[A-Z_]+)\s*(.+?)\s*(?:（\d+\s*道）)?\s*$",
    re.M,
)
_Q_RE = re.compile(
    r"^#{2,3}\s+([QS]\d{3})\s*([★☆]{1,3})?\s*[（(]?[^）)\n]*[）)]?\s*$",
    re.M,
)


def parse_questions() -> list[dict]:
    """题库 → 每题一块，带 knowledge_point / 题型 / 难度。"""
    units: list[dict] = []

    for f in sorted((BASE / "03_题库").glob("*.md")):
        md = f.read_text(encoding="utf-8")
        stage = ("S1" if ("S1" in f.name or "产品知识" in f.name) else
                 "S2" if ("S2" in f.name or "业务流程" in f.name) else "MIX")

        kps = [(m.start(), m.group(1), m.group(2).strip()) for m in _KP_RE.finditer(md)]
        qs = list(_Q_RE.finditer(md))

        for idx, m in enumerate(qs):
            qno, stars = m.group(1), m.group(2) or ""
            title_line = m.group(0)          # 含「（多选）」等标记
            start = m.end()
            end = qs[idx + 1].start() if idx + 1 < len(qs) else len(md)
            text = clean_light(md[start:end])
            if len(text) < 20:
                continue

            # 归属知识点：取该题之前最近的一个 KP 标题
            kp_code, kp_name = "", ""
            for pos, code, name in kps:
                if pos < m.start():
                    kp_code, kp_name = code, name
                else:
                    break

            q_stage = stage
            if f.name.startswith("题库_补充"):
                q_stage = "S1" if int(qno[1:]) <= 80 else "S2"

            # 答案提取：单选 1 个字母，多选可能连写（ABC）或顿号分隔（A、B、D）
            ans = re.search(r"答案[：:]\s*\**\s*([A-F][A-F,，、\s]*)", text)
            answer = re.sub(r"[^A-F]", "", ans.group(1)) if ans else ""

            # 题型判定，两条依据：
            # ① 标题行标「（多选）」（源文件 14 题）
            # ② 答案本身是多字母 —— 源文件有 2 题漏标（Q043/S023 答案是 A、B、C、D），
            #    按答案形态补判，否则会被当成单选题且答案校验逻辑出错。
            is_multi = ("多选" in title_line) or (len(answer) > 1)
            qtype = "多选" if is_multi else "单选"

            units.append(dict(
                chunk_id=qno,
                text=text[:11000],
                source_dir="03_题库",
                source_file=f.name,
                doc_id=qno,
                title=f"{qno} {kp_name or q_stage}",
                category="question",
                permission_level="restricted",
                review_status="approved",
                data_origin="synthetic",
                section=q_stage,
                chunk_index=idx,
                extra=json.dumps({
                    "stage": q_stage,
                    "answer": answer,
                    "knowledge_point": kp_code,
                }, ensure_ascii=False),
                knowledge_point=kp_code or "KP_UNKNOWN",
                kp_name=kp_name,
                question_type=qtype,
                difficulty=stars.count("★"),
            ))
    return units


# ===================================================================
# 写入
# ===================================================================
def ingest(client, emb, key: str, units: list[dict], with_kp: bool = False) -> int:
    name = COLLECTIONS[key]
    print(f"\n[{key}] → {name}  ({len(units)} 块)")
    if not units:
        print("  空，跳过")
        return 0

    t0 = time.perf_counter()
    vecs = emb.encode([u["text"] for u in units])
    create_collection(client, name, with_kp=with_kp)

    rows = [dict(u, vector=v) for u, v in zip(units, vecs)]
    for i in range(0, len(rows), 100):
        client.insert(collection_name=name, data=rows[i:i + 100])
    client.flush(name)

    lens = [len(u["text"]) for u in units]
    print(f"  写入 {len(rows)} 条 | 平均 {sum(lens)/len(lens):.0f} 字 | "
          f"最长 {max(lens)} 字 | 耗时 {time.perf_counter()-t0:.1f}s")
    return len(rows)


def main() -> None:
    keys = [a for a in sys.argv[1:] if a in COLLECTIONS] or list(COLLECTIONS)

    print("=" * 74)
    print("Milvus 定稿重建 V2")
    print("=" * 74)

    t_all = time.perf_counter()

    # 1. 解析（不加载模型，先出报告）
    print("\n--- 解析 ---")
    parsed: dict[str, list[dict]] = {}
    with_kp: dict[str, bool] = {}

    body, qa = parse_training()
    parsed["training"], parsed["self_test"] = body, qa
    with_kp["training"] = with_kp["self_test"] = False
    print(f"  training  正文块 {len(body):>4}")
    print(f"  self_test 问答块 {len(qa):>4}  (restricted)")

    parsed["case"] = parse_cases()
    with_kp["case"] = False
    cases = sorted(set(u["doc_id"] for u in parsed["case"]))
    print(f"  case      {len(parsed['case']):>4} 块 / {len(cases)} 个案例")

    parsed["question"] = parse_questions()
    with_kp["question"] = True
    kps = sorted(set(u["knowledge_point"] for u in parsed["question"]))
    miss_kp = sum(1 for u in parsed["question"] if u["knowledge_point"] == "KP_UNKNOWN")
    print(f"  question  {len(parsed['question']):>4} 题 / {len(kps)} 个知识点 / "
          f"未归属 {miss_kp}")

    total = sum(len(v) for v in parsed.values())
    print(f"\n  合计 {total} 块")

    # 2. 入库
    print("\n--- 加载 bge-m3 ---")
    emb = Embedder()
    print(f"  {emb.load_seconds:.2f}s, dim={emb.dim}, max_seq={emb._model.max_seq_length}")

    client = get_client()
    written = {}
    for k in keys:
        written[k] = ingest(client, emb, k, parsed[k], with_kp=with_kp[k])

    # 3. 汇总
    print("\n" + "=" * 74)
    print("校验")
    print("-" * 74)
    for k in keys:
        name = COLLECTIONS[k]
        n = client.get_collection_stats(name).get("row_count", 0)
        print(f"  {name:<24} {n:>5} 条")

    print(f"\n总耗时 {time.perf_counter()-t_all:.1f}s")


if __name__ == "__main__":
    main()
