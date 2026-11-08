"""组装根（Composition Root）。

★ 这是**唯一允许跨层 import** 的地方。
   职责：读配置 → 建连接 → 建适配器 → 注入端口 → 交给用例。
   不含任何业务判断。

P0 阶段的现实：
   7 个适配器尚未实现（P1/P2 才开始），所以这里做**优雅降级**：
   适配器模块不存在时，容器对应字段为 None，并在 `pending` 里记下原因。
   这样 `main.py` 能启动、自检能跑，而不会因为「P1 的东西还没写」直接崩。

P1 之后：把对应 Adapter 写出来，这里自动接上，调用方代码不用改。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import Settings
from domain.ports import (
    CachePort,
    FileStoragePort,
    GraphQueryPort,
    KeywordSearchPort,
    LanguageModelPort,
    RerankPort,
    RepositoryPort,
    VectorSearchPort,
)

logger = logging.getLogger(__name__)


@dataclass
class Container:
    """依赖容器。

    字段类型全部是 **Port**，不是具体适配器。
    这样测试时可以整个换成假实现，用例代码不用改。
    """

    settings: Settings

    graph: GraphQueryPort | None = None
    vector: VectorSearchPort | None = None
    keyword: KeywordSearchPort | None = None
    #: 稀疏检索端口（P0-3）。与 vector 共用同一个适配器实例。
    sparse: Any = None
    llm: LanguageModelPort | None = None
    #: 候选重排。None 表示未启用/不可用——用例层必须容忍这一点，
    #: 直接按 RRF 顺序取前 top_k 条即可（重排是增强，不是必需）。
    rerank: RerankPort | None = None
    repository: RepositoryPort | None = None
    cache: CachePort | None = None
    file_storage: FileStoragePort | None = None

    # 底层连接（适配器需要，但不注入给用例）
    neo4j_driver: Any = None
    milvus_client: Any = None

    # 用例（P1 之后逐步填充）
    use_cases: dict[str, Any] = field(default_factory=dict)

    #: 认证能力（P8）。用例层依赖端口，实现在装配根注入。
    password_hasher: Any = None
    token_signer: Any = None

    #: 任务队列（P3 批量生成）。
    #: ★ 同样放容器：接口层不得 import infrastructure。
    task_queue: Any = None

    #: 图的 checkpoint 存储器（P2 月度报告）。
    #: ★ 放容器里而不是让接口层自己 new：
    #:   接口层不得 import infrastructure（门禁强制），
    #:   而 checkpointer 的实现在 infrastructure。
    checkpoint_saver: Any = None

    #: 案例目录（P5 演练用）。
    #: ★ 放进容器而不是让接口层自己读文件：
    #:   接口层不得 import infrastructure（门禁强制），
    #:   而「读案例文件」是 IO。所以由装配根注入 loader。
    case_catalog: Any = None

    #: 多轮问答会话存储（B2）。纯内存、无外部依赖，不算作端口，
    #: 因此不进 `ready_ports()`；由装配根构造，接口层只读。
    conversation_store: Any = None

    #: 运行指标（D3）。内存态、重启清零，不进 `ready_ports()`。
    metrics: Any = None

    # 尚未实现的适配器：{端口名: 原因}
    pending: dict[str, str] = field(default_factory=dict)

    def ready_ports(self) -> list[str]:
        """已就绪的端口名。"""
        names = ("graph", "vector", "keyword", "sparse", "llm",
                 "repository", "cache", "file_storage", "rerank",
                 "task_queue", "checkpoint_saver")
        return [n for n in names if getattr(self, n) is not None]

    def close(self) -> None:
        """释放底层连接。进程退出时调用。"""
        if self.neo4j_driver is not None:
            try:
                self.neo4j_driver.close()
            except Exception:                     # noqa: BLE001 退出清理不阻塞
                logger.warning("关闭 Neo4j 驱动失败", exc_info=True)
            self.neo4j_driver = None

        if self.milvus_client is not None:
            try:
                self.milvus_client.close()
            except Exception:                     # noqa: BLE001
                logger.warning("关闭 Milvus 客户端失败", exc_info=True)
            self.milvus_client = None


def _try_build(
    container: Container,
    port_name: str,
    factory: Any,
    pending_reason: str,
) -> None:
    """尝试构造适配器，失败则登记为待实现（不抛异常）。

    P0 的意义就是「缺哪个报哪个」，而不是启动即崩。
    """
    try:
        setattr(container, port_name, factory())
    except ImportError as exc:
        container.pending[port_name] = f"{pending_reason}（{exc}）"
        logger.info("端口 %s 暂不可用: %s", port_name, pending_reason)
    except Exception as exc:                      # noqa: BLE001
        container.pending[port_name] = f"构造失败: {type(exc).__name__}: {exc}"
        logger.warning("端口 %s 构造失败", port_name, exc_info=True)


def _warmup_bm25(container: Container, settings: Settings) -> None:
    """用 Milvus 里的知识块构建 BM25 内存索引。

    ★ 为什么不从文件重建：
      数据源是 Milvus，如果从 `06_正式内容/` 重新切块，
      会与入库时的清洗/切片规则产生偏差（一次重建就多一处不一致）。
      直接从 Milvus 读已入库的块，保证两路检索看的是同一份数据。

    ★ 为什么失败只告警不抛：
      关键词路是「锦上添花」，它挂了向量路仍能工作（P3 已做降级）。
      不能让索引构建失败阻塞整个服务启动。
    """
    from domain.value_objects import ChunkMeta

    try:
        # 只看培训库：这是问答的主要语料。
        # 案例/题库是 restricted，问答检索时会被权限过滤挡掉，
        # 放进来只会让索引变大、命中变稀。
        rows = container.milvus_client.query(
            collection_name=settings.coll_training,
            filter="",
            output_fields=["chunk_id", "doc_id", "text", "title",
                           "section", "permission_level"],
            limit=2000,
        )
        chunks = [
            ChunkMeta(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                text=r.get("text", ""),
                title=r.get("title", "") or "",
                section=r.get("section", "") or "",
                permission_level=r.get("permission_level", "internal"),
            )
            for r in rows
        ]
        n = container.keyword.build_index(chunks)
        logger.info("BM25 索引预热完成: %d 块（来自 %s）",
                    n, settings.coll_training)
    except Exception as exc:                      # noqa: BLE001
        logger.warning(
            "BM25 索引预热失败（关键词路将不可用，其余路不受影响）: "
            "%s: %s", type(exc).__name__, exc)


def _warmup_embedding(container: Container) -> None:
    """预加载 bge-m3，把 14s 的模型加载从首请求挪到启动期。

    Args:
        container: 已装配好 vector 端口的容器。

    Note:
        失败只告警：GPU 不可用会回退 CPU，模型缺失会让向量路降级，
        都不该阻塞启动。
    """
    import time

    t0 = time.perf_counter()
    try:
        container.vector.embed(["预热"])
        logger.info(
            "Embedding 预热完成: %.1fs", time.perf_counter() - t0)
    except Exception as exc:                      # noqa: BLE001
        logger.warning(
            "Embedding 预热失败（向量检索首次调用会变慢）: %s: %s",
            type(exc).__name__, exc)


def build_container(settings: Settings) -> Container:
    """构建依赖容器。

    Args:
        settings: 配置。

    Returns:
        容器。未实现的端口为 None，原因见 `container.pending`。
    """
    container = Container(settings=settings)

    # ---------------- 底层连接 ----------------
    # 放在 try 里：Neo4j 未启动时不该让整个容器构建失败
    try:
        from neo4j import GraphDatabase

        container.neo4j_driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            connection_timeout=5,
        )
    except ImportError:
        container.pending["neo4j_driver"] = "neo4j 包未安装"
    except Exception as exc:                      # noqa: BLE001
        container.pending["neo4j_driver"] = f"建连失败: {exc}"

    try:
        from pymilvus import MilvusClient

        container.milvus_client = MilvusClient(uri=settings.milvus_uri)
    except ImportError:
        container.pending["milvus_client"] = "pymilvus 包未安装"
    except Exception as exc:                      # noqa: BLE001
        container.pending["milvus_client"] = f"建连失败: {exc}"

    # ---------------- 适配器 ----------------
    # 导入放在函数内，避免模块级 import 让「P1 还没写」变成启动失败
    def _graph() -> GraphQueryPort:
        from infrastructure.graph.neo4j_graph_adapter import Neo4jGraphAdapter
        if container.neo4j_driver is None:
            raise ImportError("Neo4j 驱动不可用")
        return Neo4jGraphAdapter(container.neo4j_driver, settings.neo4j_database)

    def _vector() -> VectorSearchPort:
        from infrastructure.vector.milvus_search_adapter import MilvusSearchAdapter
        if container.milvus_client is None:
            raise ImportError("Milvus 客户端不可用")
        return MilvusSearchAdapter(container.milvus_client, settings=settings)

    def _keyword() -> KeywordSearchPort:
        from infrastructure.search.bm25_adapter import Bm25SearchAdapter
        return Bm25SearchAdapter()

    def _sparse():
        """稀疏端口：复用向量适配器实例（同一模型、同一 Milvus 连接）。

        ★ 为什么必须复用而不是另建一个 MilvusSearchAdapter：
          它内部持有 bge-m3 权重（2.3 GB）。再建一个实例＝两份模型。
        """
        from infrastructure.vector.milvus_search_adapter import MilvusSearchAdapter

        if not settings.sparse_recall_enabled:
            raise ImportError("SPARSE_RECALL_ENABLED 未开启")
        if container.milvus_client is None:
            raise ImportError("Milvus 客户端不可用")
        if isinstance(container.vector, MilvusSearchAdapter):
            return container.vector
        return MilvusSearchAdapter(container.milvus_client, settings=settings)

    def _llm() -> LanguageModelPort:
        """构建主备双通道 LLM。

        ★ 端点由 settings.llm_endpoints() 决定，按 kind 分派：
          · kind=http   → OpenAICompatibleAdapter（千问 / DeepSeek）
          · kind=ollama → OllamaAdapter（llama.cpp，本地推荐）
          · kind=local  → LocalLLMAdapter（transformers + 4bit，训练后验证用）
          切换逻辑在 `FallbackLanguageModel` 里，用例完全感知不到。
        """
        from infrastructure.llm.fallback_adapter import FallbackLanguageModel
        from infrastructure.llm.local_llm_adapter import LocalLLMAdapter
        from infrastructure.llm.ollama_adapter import OllamaAdapter
        from infrastructure.llm.openai_compatible_adapter import (
            OpenAICompatibleAdapter,
        )

        endpoints = settings.llm_endpoints()
        if not endpoints:
            raise ImportError("未配置任何 LLM 端点（API Key 或本地模型路径）")

        # ★ 推理模型必须显式关思考：不关时 qwen3.7-flash 约 14s/1200 tokens，
        #   且 max_tokens 小的时候正文可能为空（2026-11-07 实测）。
        #   ★ 只有 http 端点支持该参数，本地/Ollama 不传。
        extra = {"enable_thinking": False} if settings.llm_disable_thinking else None

        adapters: list[LanguageModelPort] = []
        for ep in endpoints:
            if ep.get("kind") == "ollama":
                adapters.append(
                    OllamaAdapter(
                        name=ep["name"],
                        base_url=ep["base_url"],
                        model=ep["model"],
                        timeout_sec=settings.ollama_timeout_sec,
                        max_new_tokens_cap=settings.ollama_max_new_tokens,
                    )
                )
            elif ep.get("kind") == "local":
                adapters.append(
                    LocalLLMAdapter(
                        name=ep["name"],
                        model_path=ep["model"],
                        load_in_4bit=settings.local_load_in_4bit,
                        max_new_tokens_cap=settings.local_max_new_tokens,
                        lora_path=settings.local_lora_path,
                    )
                )
            else:
                adapters.append(
                    OpenAICompatibleAdapter(
                        name=ep["name"],
                        api_key=ep["api_key"],
                        base_url=ep["base_url"],
                        model=ep["model"],
                        timeout_sec=settings.llm_timeout_sec,
                        extra_payload=extra,
                    )
                )

        primary, *backups = adapters
        logger.info(
            "LLM 端点: 主=%s(%s)，备=[%s]",
            primary.name, getattr(primary, "model_id", ""),
            ", ".join(f"{b.name}({getattr(b, 'model_id', '')})" for b in backups) or "无",
        )
        return FallbackLanguageModel(
            primary=primary,
            backups=backups,
            primary_retries=settings.llm_primary_retries,
        )

    def _repository() -> RepositoryPort:
        from infrastructure.persistence.mysql_repository_adapter import (
            MySqlRepositoryAdapterFull,
        )
        # ★ 用最外层扩展类：Base → P7（员工/日报）→ P8（账号/审计）。
        #   对外仍是 RepositoryPort，用例层不感知继承层次。
        return MySqlRepositoryAdapterFull(settings.database_url)

    def _cache() -> CachePort:
        """按配置选缓存后端。

        ★ Redis 连不上时**必须降级到内存版**，不能让启动失败：
          缓存是加速手段，不是功能前提。Redis 挂了系统应该变慢，
          而不是变得不可用。
        """
        backend = (settings.cache_backend or "memory").strip().lower()

        if backend == "redis":
            try:
                from infrastructure.cache.redis_cache_adapter import (
                    RedisCacheAdapter,
                )

                adapter = RedisCacheAdapter(
                    url=settings.redis_url,
                    key_prefix=settings.redis_key_prefix,
                    socket_timeout=settings.redis_socket_timeout,
                    socket_connect_timeout=settings.redis_socket_timeout,
                )
                if adapter.ping():
                    logger.info(
                        "缓存后端: Redis (%s, prefix=%s)",
                        settings.redis_url, settings.redis_key_prefix)
                    return adapter
                logger.warning(
                    "Redis 不可达，缓存降级为内存版: %s", settings.redis_url)
            except Exception as exc:                   # noqa: BLE001
                logger.warning(
                    "Redis 适配器构造失败，缓存降级为内存版: %s: %s",
                    type(exc).__name__, exc)

        from infrastructure.cache.inmemory_cache_adapter import InMemoryCacheAdapter

        # ★ 只有「配了未知值」才报未知；配了 redis 但连不上，上面已经报过降级原因
        if backend not in ("memory", "redis"):
            logger.warning("未知缓存后端 %r，使用内存版", backend)
        return InMemoryCacheAdapter()

    def _file_storage() -> FileStoragePort:
        from infrastructure.storage.local_file_storage_adapter import (
            LocalFileStorageAdapter,
        )
        return LocalFileStorageAdapter(
            root=settings.storage_root_path,
            max_bytes=settings.file_storage_max_bytes,
        )

    def _rerank() -> RerankPort:
        """构建 rerank 端口。

        ★ 开关在 settings.rerank_enabled：关闭时这里直接 raise ImportError，
          容器里 rerank 保持 None —— 用例层拿到的就是「无 rerank」的原始链路，
          与历史行为逐字节一致（docs/TDD_检索质量优化 §4.3 降级矩阵）。
        """
        if not settings.rerank_enabled:
            raise ImportError("rerank_enabled=false（默认关闭）")
        from infrastructure.rerank.bge_reranker_adapter import BgeRerankerAdapter

        return BgeRerankerAdapter(
            model_dir=settings.rerank_model_dir,
            max_length=settings.rerank_max_length,
            batch_size=settings.rerank_batch_size,
            slow_warn_sec=settings.rerank_slow_warn_sec,
        )

    _try_build(container, "graph", _graph, "P1 阶段实现 Neo4jGraphAdapter")
    _try_build(container, "vector", _vector, "P2 阶段实现 MilvusSearchAdapter")
    _try_build(container, "keyword", _keyword, "P3 阶段实现 Bm25SearchAdapter")
    _try_build(container, "llm", _llm, "配置 QWEN_API_KEY 或 DEEPSEEK_API_KEY")
    _try_build(container, "repository", _repository,
               "P6/P7 阶段实现 MySqlRepositoryAdapter")
    _try_build(container, "cache", _cache, "实现 InMemoryCacheAdapter")
    _try_build(container, "file_storage", _file_storage,
               "P6 阶段实现 LocalFileStorageAdapter")
    _try_build(container, "rerank", _rerank,
               "RERANK_ENABLED=true 且模型就位后启用")

    # ---------------- 任务队列（P3） ----------------
    def _task_queue():
        backend = (settings.queue_backend or "memory").strip().lower()

        if backend == "rabbitmq":
            if not settings.rabbitmq_url:
                raise RuntimeError("QUEUE_BACKEND=rabbitmq 但未配置 RABBITMQ_URL")
            from infrastructure.queue.rabbitmq_task_queue import (
                RabbitMQTaskQueue,
            )

            q = RabbitMQTaskQueue(url=settings.rabbitmq_url)
            if not q.is_ready():
                raise RuntimeError("RabbitMQ 不可达")
            logger.info("任务队列: RabbitMQ")
            return q

        from infrastructure.queue.inmemory_task_queue import InMemoryTaskQueue

        logger.info("任务队列: 内存版（单进程，重启即丢）")
        return InMemoryTaskQueue(
            max_attempts=settings.queue_max_attempts)

    _try_build(container, "task_queue", _task_queue, "配置 QUEUE_BACKEND")

    # ---------------- 图的 checkpoint（P2） ----------------
    # ★ 只在 cache 就绪时构造。挂不上就没有中断能力，
    #   接口层据此返回 503（而不是悄悄降级成「无法复核」）。
    def _checkpoint_saver():
        from infrastructure.graph.cache_checkpoint_saver import (
            CacheCheckpointSaver,
        )

        saver = CacheCheckpointSaver(
            container.cache, ttl_sec=7 * 86400)
        if not saver.is_ready():
            raise RuntimeError("checkpoint 存储不可用")
        return saver

    if container.cache is not None:
        _try_build(container, "checkpoint_saver", _checkpoint_saver,
                   "配置可用的 CACHE_BACKEND")

    # ---------------- 认证（P8） ----------------
    def _password_hasher():
        from infrastructure.auth.adapters import BcryptPasswordHasher

        return BcryptPasswordHasher()

    def _token_signer():
        from infrastructure.auth.adapters import HmacTokenSigner

        return HmacTokenSigner(
            secret=settings.auth_secret,
            ttl_sec=settings.auth_token_ttl,
        )

    _try_build(container, "password_hasher", _password_hasher,
               "实现 BcryptPasswordHasher")
    _try_build(container, "token_signer", _token_signer,
               "配置 AUTH_SECRET")

    # ---------------- 案例目录 ----------------
    # ★ loader 在这里注入：装配根可以 import infrastructure，
    #   而 application 层的 CaseCatalog 只接受一个函数，不依赖具体实现。
    def _case_catalog():
        from application.services.case_catalog import CaseCatalog
        from infrastructure.content.case_loader import load_cases_from_dir

        return CaseCatalog(Path(settings.case_dir), load_cases_from_dir)

    _try_build(container, "case_catalog", _case_catalog,
               "配置 CASE_DIR 指向 02_情景案例 目录")

    # ---------------- 稀疏端口（P0-3） ----------------
    # ★ 依赖 vector 先构建好，才能复用同一个适配器实例，
    #   所以必须放在 vector 之后。
    _try_build(container, "sparse", _sparse, "开启 SPARSE_RECALL_ENABLED")

    # ---------------- BM25 索引预热 ----------------
    # ★ 为什么必须在这里建索引：
    #   `Bm25SearchAdapter` 是纯内存索引，进程启动时是空的。
    #   若不预热，关键词路每次检索都返回空 —— 不报错，只是静默失效。
    #   实测踩过：端到端跑通但「关键词命中=0」，日志里只有一行 warning。
    if container.keyword is not None and container.vector is not None:
        _warmup_bm25(container, settings)

    # ---------------- Embedding 预热 ----------------
    # ★ 为什么需要：
    #   bge-m3 首次加载约 14s（`import sentence_transformers` 11.5s + 权重 3s）。
    #   不预热的话，**第一个用户请求**要等 14s + LLM 3s ≈ 17s。
    #   实测端到端 20s 里 14s 是这个。
    #   在装配阶段预热，把这段代价从「用户等待」变成「服务启动」。
    #
    # ★ 为什么失败只告警：
    #   GPU 不可用时会自动回退 CPU（更慢但可用）；模型缺失时向量路降级。
    #   都不能阻塞启动——否则整个服务起不来。
    if container.vector is not None and settings.warmup_embedding:
        _warmup_embedding(container)

    # ---------------- 用例 ----------------
    # P1 起逐步注册：container.use_cases["answer_question"] = ...
    # ★ 用例构造函数只接受 Port，不接受具体适配器类型
    container.use_cases = {}

    # ---------------- 多轮会话存储（B2） ----------------
    # ★ 不占端口、不碰外部服务，放在端口装配之后构造，
    #   现有端口的装配顺序与降级行为完全不受影响。
    from application.services.conversation import ConversationStore

    container.conversation_store = ConversationStore(
        max_sessions=settings.multi_turn_max_sessions,
        max_turns=settings.multi_turn_max_turns,
        ttl_sec=settings.multi_turn_ttl_sec,
    )

    # ---------------- 运行指标（D3） ----------------
    from application.services.ops_metrics import OpsMetrics

    container.metrics = OpsMetrics()

    logger.info(
        "容器构建完成: 就绪 %d 个端口 %s，待实现 %d 个",
        len(container.ready_ports()), container.ready_ports(), len(container.pending),
    )
    return container


def build_test_container(settings: Settings) -> Container:
    """构建全假实现的容器。供应用层/接口层测试使用，不碰真服务。

    ★ 这是「绝大多数测试不依赖外部服务」的落地手段。
    """
    from application.services.conversation import ConversationStore
    from application.services.ops_metrics import OpsMetrics
    from tests.fakes import (
        FakeCachePort,
        FakeGraphQueryPort,
        FakeKeywordSearchPort,
        FakeLanguageModelPort,
        FakeRepositoryPort,
        FakeVectorSearchPort,
        InMemoryFileStorage,
    )

    return Container(
        settings=settings,
        graph=FakeGraphQueryPort(),
        vector=FakeVectorSearchPort(),
        keyword=FakeKeywordSearchPort(),
        llm=FakeLanguageModelPort(),
        repository=FakeRepositoryPort(),
        cache=FakeCachePort(),
        file_storage=InMemoryFileStorage(),
        conversation_store=ConversationStore(
            max_sessions=settings.multi_turn_max_sessions,
            max_turns=settings.multi_turn_max_turns,
            ttl_sec=settings.multi_turn_ttl_sec,
        ),
        metrics=OpsMetrics(),
    )
