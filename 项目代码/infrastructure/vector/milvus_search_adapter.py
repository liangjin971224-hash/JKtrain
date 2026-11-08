"""Milvus 向量检索适配器。

实现 `VectorSearchPort` 的 3 个方法。

数据源：Milvus 4 个 collection（V2 重建，2026-10-22）
    health_training_docs  293（public 113 / internal 180）
    health_case_library   117（全 restricted）
    health_question_bank  140（全 restricted，带 knowledge_point）
    health_self_test       30（全 restricted）

────────────────────────────────────────────────────────────────────
设计约束（违反即架构失效）
────────────────────────────────────────────────────────────────────
1. **权限过滤必须在检索层**（拼进 filter_expr），不能检索后再过滤。
   原因：top-20 里若有 18 条受限，过滤后只剩 2 条，召回质量崩溃。
   这是 P2 的红灯项，也是 FR-050 的技术实现点。
2. **模型惰性加载**：`import sentence_transformers` 占启动时间 87%（11.5s），
   不能在模块导入时就加载。首次 `embed()` 才初始化。
3. 返回 `VectorHit`（含 `ChunkMeta` + score），**不含向量**——
   1024 维向量会给领域对象撑出几百 KB，且领域层不该关心向量。
4. 空查询返回空列表，不抛异常；向量检索本身永远有 Top-N，靠分数判断相关性。
"""
from __future__ import annotations

import logging
import os
from typing import Any

from pymilvus import MilvusClient

from config.settings import Settings
from domain.exceptions import EmbeddingDimensionError
from domain.ports import VectorSearchPort
from domain.value_objects import ChunkMeta, VectorHit
from infrastructure.exceptions import VectorSearchError

logger = logging.getLogger(__name__)

# 离线模式：避免 sentence_transformers 尝试联网检查模型
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

#: 所有 collection 共有的标量字段。★ 不含 vector —— 它会让返回体体积暴涨且领域层不需要
_COMMON_FIELDS = (
    "chunk_id", "text", "doc_id", "title", "section",
    "category", "permission_level", "review_status", "data_origin",
    "source_file", "chunk_index",
)

#: 只有题库 collection 才有的字段。
#: ★ 实测坑：在 health_training_docs 上指定 `knowledge_point` 会直接报
#:   `MilvusException: field knowledge_point not exist`（不是静默忽略）。
#:   所以输出字段必须按 collection 的 schema 动态取，不能写死一份全字段列表。
_QUESTION_ONLY_FIELDS = ("knowledge_point", "question_type", "difficulty")

#: 允许的权限级别白名单（防止拼进 filter 的字符串被污染）
_VALID_PERMISSIONS = frozenset({"public", "internal", "restricted", "confidential"})


class MilvusSearchAdapter(VectorSearchPort):
    """Milvus 语义检索实现。"""

    def __init__(self, client: MilvusClient, *, settings: Settings) -> None:
        """
        Args:
            client: Milvus 客户端（由装配根创建并负责关闭）。
            settings: 配置（模型路径、维度、collection 名）。
        """
        self._client = client
        self._settings = settings
        self._model: Any = None          # 惰性加载，见 _ensure_model
        # collection → 该库实际存在的输出字段（惰性探测并缓存）
        self._field_cache: dict[str, list[str]] = {}

    # ---------------- 内部 ----------------
    def _output_fields(self, collection: str) -> list[str]:
        """按 collection 的真实 schema 取输出字段。

        为什么必须动态取：
            `health_question_bank` 有 knowledge_point / question_type / difficulty，
            其余三个 collection **没有**。在 training 库上指定这些字段会让
            Milvus 直接抛 `field xxx not exist`，而不是忽略。

        缓存理由：schema 在运行期不变，一次探测够用；
        但 collection 若被重建（如 08_数据入库 重跑），需重启进程刷新。
        """
        if collection in self._field_cache:
            return self._field_cache[collection]

        desc = self._client.describe_collection(collection)
        existing = {f["name"] for f in desc["fields"]} - {"id", "vector"}

        fields = [f for f in _COMMON_FIELDS if f in existing]
        fields += [f for f in _QUESTION_ONLY_FIELDS if f in existing]

        missing = set(_COMMON_FIELDS) - existing
        if missing:
            logger.warning(
                "collection %s 缺少通用字段 %s，检索结果可能不完整",
                collection, sorted(missing),
            )

        self._field_cache[collection] = fields
        return fields

    def _ensure_model(self) -> Any:
        """惰性加载 bge-m3。

        为什么不在 __init__ 里加载：
            `import sentence_transformers` 约 11.5s（占总启动 87%），
            模型加载再 5.5s，还要占 8GB 显存的一部分。
            P0/P1 阶段完全不需要向量化，却要为它付启动代价。
        为什么不在模块级加载：
            import 该模块就会触发，等于把代价转嫁给所有 import 方。
        """
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        model_dir = self._settings.embed_model_dir
        logger.info("首次加载 bge-m3: %s", model_dir)

        device = "cuda"
        try:
            model = SentenceTransformer(model_dir, device=device)
        except Exception as exc:              # noqa: BLE001 显存不足时降级 CPU
            logger.warning(
                "GPU 加载失败（%s），回退 CPU。CPU 编码会慢 10 倍以上。", exc)
            model = SentenceTransformer(model_dir, device="cpu")

        # 4096 序列长度：案例最长 2821 token，512 会截断（V1 踩过的坑）
        model.max_seq_length = self._settings.embed_max_seq_length

        dim = model.get_sentence_embedding_dimension()
        self._check_dim(dim)

        self._model = model
        logger.info("bge-m3 加载完成: dim=%d, max_seq=%d",
                    dim, model.max_seq_length)
        return model

    def _check_dim(self, actual: int) -> None:
        """校验向量维度与配置一致。

        Raises:
            EmbeddingDimensionError: 不符。维度不符会让 Milvus 检索直接失败，
                且原生报错很晦涩，所以在适配器层就拦下并给可执行提示。
        """
        expected = self._settings.expected_embed_dim
        if actual != expected:
            raise EmbeddingDimensionError(actual, expected)

    def _build_filter(
        self,
        allowed_permissions: tuple[str, ...] | None,
        filter_expr: str | None,
        *,
        review_only: bool = True,
    ) -> str | None:
        """组装 Milvus 标量过滤表达式。

        ★ 权限条件在这里拼进 filter_expr，由 Milvus 在**检索时**过滤。
          绝不能拿到结果后再 `[h for h in hits if ...]`。

        Args:
            allowed_permissions: 允许的权限级别。None 表示不限制（仅内部调用）。
            filter_expr: 额外的过滤表达式，与权限条件用 and 组合。
            review_only: 是否只检索已审核内容。

        Returns:
            Milvus filter 表达式；无条件时返回 None。

        Raises:
            VectorSearchError: allowed_permissions 含非法值。
        """
        conditions: list[str] = []

        if allowed_permissions is not None:
            perms = [str(p) for p in allowed_permissions]
            invalid = set(perms) - _VALID_PERMISSIONS
            if invalid:
                # 不静默忽略：非法权限值可能是调用方拼错了，放行会造成越权
                raise VectorSearchError(
                    f"非法权限级别: {sorted(invalid)}", collection="")
            if not perms:
                # 空列表意味着「一个都不允许」→ 用不可能成立的条件
                return 'permission_level in ["__none__"]'
            quoted = ", ".join(f'"{p}"' for p in perms)
            conditions.append(f"permission_level in [{quoted}]")

        if review_only:
            conditions.append('review_status == "approved"')

        if filter_expr:
            conditions.append(f"({filter_expr})")

        return " and ".join(conditions) if conditions else None

    # ---------------- 端口实现 ----------------
    def search(
        self,
        query: str,
        *,
        collection: str,
        top_k: int = 20,
        filter_expr: str | None = None,
        allowed_permissions: tuple[str, ...] | None = None,
    ) -> list[VectorHit]:
        text = (query or "").strip()
        if not text:
            return []
        if top_k <= 0:
            return []

        # 先建 filter 再编码：参数非法时不必浪费一次编码
        expr = self._build_filter(allowed_permissions, filter_expr)

        vector = self.embed([text])[0]

        try:
            raw = self._client.search(
                collection_name=collection,
                data=[vector],
                limit=top_k,
                filter=expr,
                output_fields=self._output_fields(collection),
            )
        except Exception as exc:              # noqa: BLE001
            raise VectorSearchError(
                f"Milvus 检索失败: {type(exc).__name__}: {exc}",
                collection=collection,
            ) from exc

        hits = raw[0] if raw else []
        out = [
            VectorHit(chunk=_to_chunk_meta(h["entity"]), score=float(h["distance"]))
            for h in hits
        ]

        logger.debug(
            "向量检索: collection=%s, top_k=%d, 返回=%d, filter=%s",
            collection, top_k, len(out), expr,
        )
        return out

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._ensure_model()
        try:
            vectors = model.encode(
                texts,
                batch_size=8,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:              # noqa: BLE001
            raise VectorSearchError(
                f"向量化失败: {type(exc).__name__}: {exc}") from exc

        return [v.tolist() for v in vectors]

    def get_chunks_by_ids(
        self, chunk_ids: list[str], *, collection: str
    ) -> list[ChunkMeta]:
        if not chunk_ids:
            return []

        # 用 chunk_id in [...] 精确查，不走向量
        quoted = ", ".join(f'"{cid}"' for cid in chunk_ids)
        try:
            rows = self._client.query(
                collection_name=collection,
                filter=f"chunk_id in [{quoted}]",
                output_fields=self._output_fields(collection),
            )
        except Exception as exc:              # noqa: BLE001
            raise VectorSearchError(
                f"按 chunk_id 查询失败: {type(exc).__name__}: {exc}",
                collection=collection,
            ) from exc

        return [_to_chunk_meta(r) for r in rows]


# ===================================================================
# 转换函数：Milvus 行 → 领域对象
# ===================================================================
def _to_chunk_meta(row: dict[str, Any]) -> ChunkMeta:
    """把 Milvus 返回的一行转成 `ChunkMeta`。

    ★ 显式丢弃 `vector`：即使上游误传全字段，也不让 1024 维向量进领域层。
    """
    return ChunkMeta(
        chunk_id=str(row.get("chunk_id", "")),
        doc_id=str(row.get("doc_id", "")),
        text=str(row.get("text", "")),
        title=str(row.get("title", "") or ""),
        section=str(row.get("section", "") or ""),
        category=str(row.get("category", "") or "training"),
        permission_level=str(row.get("permission_level", "") or "internal"),
        review_status=str(row.get("review_status", "") or "approved"),
        data_origin=str(row.get("data_origin", "") or "real"),
        source_file=str(row.get("source_file", "") or ""),
        chunk_index=int(row.get("chunk_index") or 0),
        knowledge_point=str(row.get("knowledge_point", "") or ""),
        question_type=str(row.get("question_type", "") or ""),
        difficulty=int(row.get("difficulty") or 0),
    )
