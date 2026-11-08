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

# ★ 路径改为相对本文件推导 + 环境变量覆盖（2026-10-23 迁移时改）。
#   原先是硬编码 `D:\项目\某健康公司培训系统\06_正式内容` 与
#   `D:\ai260803\...\bge-m3`，换一台电脑必然报错。
_HERE = Path(__file__).resolve().parent          # …/08_数据入库/milvus
_PROJECT_ROOT = _HERE.parents[1]                 # 项目根目录

#: Milvus 地址。可用环境变量 MILVUS_URI 覆盖。
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")

#: bge-m3 模型目录。
#:
#: ★ 默认值指向「项目根目录/models/bge-m3」。
#:   模型放在别处就用环境变量 EMBED_MODEL_DIR 指定，不必改代码。
MODEL_DIR = os.getenv(
    "EMBED_MODEL_DIR", str(_PROJECT_ROOT / "models" / "bge-m3"))

EMBED_DIM = 1024

#: 数据源目录（06_正式内容）。可用环境变量 CONTENT_DIR 覆盖。
BASE_DIR = Path(os.getenv(
    "CONTENT_DIR", str(_PROJECT_ROOT / "06_正式内容")))


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
# Sparse Embedding（P0-3）
# ---------------------------------------------------------------
def load_bge_m3_sparse(model_dir: str = MODEL_DIR, device: str = "cpu"):
    """加载 BGEM3FlagModel 用于 sparse 编码，并强制校验官方权重已读到。

    bge-m3 的 sparse/colbert 线性层权重不在 pytorch_model.bin 里，而在
    sparse_linear.pt / colbert_linear.pt。缺这两个文件时 FlagEmbedding 会
    静默随机初始化（日志 "new initialize"），产出的 lexical_weights 近乎全空，
    导致稀疏向量全是噪声。这里直接把「静默」变成「报错」。
    """
    import io
    import logging
    from pathlib import Path
    from FlagEmbedding import BGEM3FlagModel

    for f in ("sparse_linear.pt", "colbert_linear.pt"):
        if not (Path(model_dir) / f).exists():
            raise RuntimeError(
                f"[P0-3] 缺少 {f}，FlagEmbedding 会随机初始化稀疏层。"
                f"请补到 {model_dir} 后再入库。")

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    try:
        model = BGEM3FlagModel(model_dir, device=device, use_fp16=False)
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    log = buf.getvalue()
    if "new initialize" in log:
        raise RuntimeError(
            "[P0-3] FlagEmbedding 仍在随机初始化 sparse/colbert 层，"
            "官方权重未被读取；拒绝继续（会产生空稀疏向量）。")
    return model


def to_sparse_rows(lexical_weights) -> list[dict[int, float]]:
    """FlagEmbedding 的 lexical_weights（str token id）→ Milvus 稀疏向量格式。"""
    return [{int(k): float(v) for k, v in lw.items()}
            for lw in lexical_weights]


def encode_sparse_vectors(sparse_model, texts: list[str],
                          batch_size: int = 8,
                          max_length: int = 4096) -> list[dict[int, float]]:
    """与 dense 侧对齐：max_length=4096（案例合并块最长 2110 字，512 会截断）。"""
    if not texts:
        return []
    out = sparse_model.encode(
        texts, batch_size=batch_size, max_length=max_length,
        return_dense=False, return_sparse=True, return_colbert_vecs=False)
    return to_sparse_rows(out["lexical_weights"])


# ---------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------
class Embedder:
    """bge-m3 向量化。加载一次复用。"""

    def __init__(self, model_dir: str = MODEL_DIR, device: str | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        # 迁移到无 NVIDIA 显卡的机器时使用 CPU，仍允许调用方指定设备。
        device = device or os.getenv("EMBED_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.perf_counter()
        self._model = SentenceTransformer(model_dir, device=device)
        self._model.max_seq_length = 4096      # 案例最长 2821 token，512 会截断
        self.load_seconds = time.perf_counter() - t0
        self.dim = self._model.get_sentence_embedding_dimension()
        if self.dim != EMBED_DIM:
            raise RuntimeError(f"维度不符: 实际 {self.dim}, 期望 {EMBED_DIM}")
        self.model_dir = model_dir
        self._sparse_model = None            # 懒加载，dense-only 时不占内存

    def encode_sparse(self, texts: list[str], batch_size: int = 8) -> list[dict[int, float]]:
        """P0-3 4.2-B：产出 bge-m3 lexical_weights，供 sparse_vector 字段写入。"""
        if self._sparse_model is None:
            self._sparse_model = load_bge_m3_sparse(self.model_dir)
        return encode_sparse_vectors(self._sparse_model, texts, batch_size=batch_size)

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
                if name == "long" and self._model.device.type == "cuda":
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
