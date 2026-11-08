"""
通用工具：规则切块 + bge-m3 向量化 + Milvus 写入
三个文件夹共用，各自指定 collection 与元数据规则
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MILVUS_URI = "http://localhost:19530"
MODEL_DIR = r"D:\ai260803\xiangmudaima\pytorch\model\bge-m3"
EMBED_DIM = 1024
BASE_DIR = Path(r"D:\项目\某健康公司培训系统\06_正式内容")


# ---------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------
@dataclass
class Chunk:
    """一个知识分块。所有字段都会写入 Milvus。"""

    chunk_id: str
    text: str
    source_dir: str          # 01_培训文档 / 02_情景案例 / 03_题库
    source_file: str         # 文件名
    doc_id: str              # D001 / C01 / Q001
    title: str               # 文档或案例标题
    category: str            # training / case / question
    permission_level: str    # public / internal / restricted
    review_status: str       # approved
    data_origin: str         # real / synthetic
    section: str = ""        # 章节标题
    chunk_index: int = 0
    extra: str = ""          # 额外 JSON 字符串（如题型、难度）
    embedding: list[float] = field(default_factory=list)

    def to_milvus_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        d["vector"] = self.embedding
        return d


# ---------------------------------------------------------------
# 规则切块（按 Markdown 标题层级，不用模型）
# ---------------------------------------------------------------
def split_by_heading(md_text: str, max_chars: int = 700) -> list[tuple[str, str]]:
    """按 Markdown 标题切块。

    Returns:
        [(section_title, text), ...]
    """
    lines = md_text.splitlines()
    sections: list[tuple[str, str]] = []
    cur_title = ""
    cur_lines: list[str] = []

    def flush() -> None:
        if not cur_lines:
            return
        body = "\n".join(cur_lines).strip()
        if not body:
            return
        # 超长段落再按空行二次切分
        if len(body) > max_chars * 2:
            parts = re.split(r"\n\s*\n", body)
            buf: list[str] = []
            for p in parts:
                buf.append(p)
                if sum(len(x) for x in buf) >= max_chars:
                    sections.append((cur_title, "\n\n".join(buf).strip()))
                    buf = []
            if buf:
                sections.append((cur_title, "\n\n".join(buf).strip()))
        else:
            sections.append((cur_title, body))

    for line in lines:
        if re.match(r"^#{1,3}\s", line):
            flush()
            cur_lines = []
            cur_title = line.lstrip("#").strip()
        else:
            cur_lines.append(line)
    flush()
    return [(t, b) for t, b in sections if b]


def clean_text(t: str) -> str:
    """去掉代码围栏与纯分隔线，保留可检索正文。"""
    t = re.sub(r"^```[a-zA-Z]*\s*$", "", t, flags=re.M)
    t = re.sub(r"^-{3,}\s*$", "", t, flags=re.M)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ---------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------
class Embedder:
    """bge-m3 向量化。加载一次复用。"""

    def __init__(self, model_dir: str = MODEL_DIR, device: str = "cuda") -> None:
        from sentence_transformers import SentenceTransformer

        t0 = time.perf_counter()
        self._model = SentenceTransformer(model_dir, device=device)
        self._model.max_seq_length = 4096      # 案例最长 2821 token，512 会截断
        self.load_seconds = time.perf_counter() - t0
        self.dim = self._model.get_sentence_embedding_dimension()
        if self.dim != EMBED_DIM:
            raise RuntimeError(f"维度不符: 实际 {self.dim}, 期望 {EMBED_DIM}")

    def encode(self, texts: list[str], batch_size: int = 8) -> list[list[float]]:
        """按长度分桶编码，避免长文本与短文本混批导致显存峰值。

        4096 序列长度下注意力矩阵是 O(L^2)，长文本必须单独小批。
        """
        import torch

        if not texts:
            return []

        tok = self._model.tokenizer
        lens = [len(tok.encode(t, add_special_tokens=True)) for t in texts]

        # 分桶：短(<400) / 中(400-1200) / 长(>1200)
        buckets: dict[str, list[int]] = {"short": [], "mid": [], "long": []}
        for i, n in enumerate(lens):
            if n < 400:
                buckets["short"].append(i)
            elif n < 1200:
                buckets["mid"].append(i)
            else:
                buckets["long"].append(i)

        out: list[list[float]] = [[] for _ in texts]

        # 每桶用不同 batch_size
        plan = [("short", 16), ("mid", 4), ("long", 1)]
        for name, bs in plan:
            idxs = buckets[name]
            if not idxs:
                continue
            for s in range(0, len(idxs), bs):
                sub = idxs[s:s + bs]
                vecs = self._model.encode(
                    [texts[i] for i in sub],
                    batch_size=bs,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                for i, v in zip(sub, vecs):
                    out[i] = v.tolist()
                if name == "long":
                    torch.cuda.empty_cache()

        return out


# ---------------------------------------------------------------
# Milvus 操作
# ---------------------------------------------------------------
def get_client():
    from pymilvus import MilvusClient

    return MilvusClient(uri=MILVUS_URI)


def recreate_collection(client, name: str, desc: str = "") -> None:
    """幂等重建 collection。已存在则先 drop。"""
    if client.has_collection(name):
        client.drop_collection(name)
        print(f"  已删除旧 collection: {name}")

    from pymilvus import DataType

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBED_DIM)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=12000)
    schema.add_field("source_dir", DataType.VARCHAR, max_length=64)
    schema.add_field("source_file", DataType.VARCHAR, max_length=128)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=32)
    schema.add_field("title", DataType.VARCHAR, max_length=256)
    schema.add_field("category", DataType.VARCHAR, max_length=32)
    schema.add_field("permission_level", DataType.VARCHAR, max_length=32)
    schema.add_field("review_status", DataType.VARCHAR, max_length=32)
    schema.add_field("data_origin", DataType.VARCHAR, max_length=32)
    schema.add_field("section", DataType.VARCHAR, max_length=256)
    schema.add_field("chunk_index", DataType.INT32)
    schema.add_field("extra", DataType.VARCHAR, max_length=2000)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )

    client.create_collection(
        collection_name=name,
        schema=schema,
        index_params=index_params,
    )
    print(f"  已创建 collection: {name}  ({desc})")


def insert_chunks(client, collection: str, chunks: list[Chunk], batch: int = 100) -> int:
    total = 0
    for i in range(0, len(chunks), batch):
        rows = [c.to_milvus_dict() for c in chunks[i:i + batch]]
        client.insert(collection_name=collection, data=rows)
        total += len(rows)
    client.flush(collection)
    return total


def print_collection_stat(client, name: str) -> None:
    stats = client.get_collection_stats(name)
    print(f"  {name}: {stats.get('row_count', 0)} 条")
