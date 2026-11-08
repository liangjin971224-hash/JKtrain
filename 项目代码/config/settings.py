"""配置。

★ 这是唯一允许出现默认连接串的地方（CI 门禁要求）。
业务代码一律经 `Settings` 取配置，不硬编码。

用法：
    settings = get_settings()          # 进程内单例，读 .env
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目代码根目录（config/ 的上一级）
BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """全局配置。所有项都能被环境变量或 .env 覆盖。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------- Neo4j ----------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "12345678"
    # 库名走配置，不硬编码（换库不改代码）
    neo4j_database: str = "jiankangshujuku"

    # ---------------- Milvus ----------------
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10

    # ---------------- MySQL ----------------
    database_url: str = (
        "mysql+pymysql://root:12345678@localhost:3306/health_training"
        "?charset=utf8mb4"
    )

    # ---------------- Embedding ----------------
    embed_model_dir: str = (
        r"D:\ai260803\xiangmudaima\pytorch\model\bge-m3"
    )
    expected_embed_dim: int = 1024
    # 4096 序列长度下 8GB 显存必须按长度分桶编码，否则 OOM
    embed_max_seq_length: int = 4096
    # 启动时预加载模型。
    # ★ 开启后启动慢约 14s，但**首个用户请求**从 17s 降到 3s。
    #   演示/PoC 场景必须开启；纯离线批处理可关掉。
    warmup_embedding: bool = True

    # ---------------- Rerank（候选重排） ----------------
    # 作用：对 RRF 融合后的候选重排，补「只看排名不看内容」的盲区。
    #
    # ★ 默认关闭。原因有二：
    #   1) 模型约 2.2GB 显存，与 bge-m3（约 2.5GB）叠加后在 8GB 卡上余量紧张；
    #   2) 需要先实测 R@1 是否真的提升，再决定长期开启。
    # 开启方式：.env 里设 RERANK_ENABLED=true
    rerank_enabled: bool = False
    #: 本地模型目录（不走联网）。缺文件时适配器自动降级为原顺序。
    rerank_model_dir: str = (
        r"D://ai260803//xiangmudaima//pytorch//model//bge-reranker-v2-m3"
    )
    #: 送入重排的候选条数。太少无意义，太多延迟涨。
    rerank_candidates: int = 20
    #: 重排后保留条数（进 prompt 的参考资料数）。
    rerank_top_k: int = 5
    #: 前向批大小，控制峰值显存。
    rerank_batch_size: int = 8
    #: 模型侧最大序列长度。
    rerank_max_length: int = 512

    # ---------------- LLM 多跳规划（可选增强） ----------------
    # 规则先跑，只有结果可疑时才让 LLM 复核「拆几步、谁依赖谁」。
    #
    # ★ 默认关闭，理由：
    #   1) 规则已覆盖带标点+回指词的常见问法，80% 场景不需要模型；
    #   2) 需要先实测「LLM 拆解是否真的比规则好」再决定常开。
    # 开启方式：.env 里设 PLANNER_USE_LLM=true
    planner_use_llm: bool = False
    #: 每请求最多调几次（与语义映射一致的预算机制，守住成本红线）。
    planner_max_calls: int = 1
    #: 最多拆几步（与 domain 层 MAX_STEPS 对齐，不能超 3）。
    planner_max_steps: int = 3
    #: 规划结果缓存秒数。培训场景学员反复问相似问题，缓存收益明显。
    planner_cache_ttl_sec: int = 300

    # ---------------- LLM（主备双通道） ----------------
    # 设计：千问为主，DeepSeek 为备。千问异常时自动切换，对上层透明。
    #
    # 为什么不用单 Key + 单 BaseURL：
    #   单点故障。DeepSeek 一次限流就让整个问答不可用，
    #   而本项目的问答是核心功能（P4 演示点）。
    #
    # 为什么千问为主：
    #   阿里云百炼国内链路更稳，且 qwen-plus 中文表达更贴合本项目的
    #   「合规话术」场景（要说得像门店顾问，不能像说明书）。
    llm_primary: str = "qwen"                 # qwen | deepseek | local
    llm_fallback_enabled: bool = True
    llm_primary_retries: int = 1              # 切换前重试次数（不含首次）
    llm_timeout_sec: int = 30

    # 主：千问（DashScope OpenAI 兼容模式）
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"

    # 备：DeepSeek
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # ---------------- 本地大模型（可选） ----------------
    # 用途：内网 / 无外部 API / 离线批处理。在线问答速度不足（实测 2.4 tok/s）。
    # 启用方式：LLM_PRIMARY=local
    # ★ 8GB 卡上 7B 必须 4bit（fp16 需 15.2GB）；且与 bge-m3 不能同时驻留
    #   （4bit 5.65GB + bge-m3 2.5GB = 8.15GB > 8GB）
    local_model_path: str = ""
    local_load_in_4bit: bool = True
    local_max_new_tokens: int = 1024          # 输出上限，防止误传打爆显存
    #: LoRA 适配器目录（可选）。为空则用基座模型。
    #: 实测：基座模型会把「降血压」说成「不属于功能声称」，
    #: 微调后能指出不合规并给出正确表述。
    local_lora_path: str = ""

    # ---------------- Ollama（llama.cpp 推理引擎） ----------------
    # 用途：**推荐的本地推理方式**。transformers 实测仅 2.2 tok/s，
    # 同一 3B 模型换 Ollama（GGUF Q4_K_M）达 101–108 tok/s，快 45–49 倍。
    # 启用方式：LLM_PRIMARY=ollama
    #
    # ★ 必须走 /api/chat（OllamaAdapter 已固定），不能用 /api/generate：
    #   generate 不套对话模板，微调模型会退化成「知识库没有资料」。
    # ★ 模型来自 LoRA 全链路转换：merge → convert_hf_to_gguf → ollama create
    #   产出物 health-advisor:3b（1.9GB），耗时 43s + 42s + 2min。
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "health-advisor:3b"
    #: 本地推理首 token 较慢，超时放宽；正常回答 < 2s
    ollama_timeout_sec: int = 120
    #: 输出上限（防止误传超大值把耗时拉长）
    ollama_max_new_tokens: int = 1024

    # ---------------- Redis ----------------
    redis_url: str = "redis://127.0.0.1:6379/0"

    #: 缓存后端：memory = 进程内字典；redis = Redis（多实例共享、重启可恢复）
    #:
    #: ★ 什么时候必须换 redis：
    #:   水平扩容（多个进程/实例）时**会话状态必须共享**，否则演练游标
    #:   （当前问到第几句）会错乱——A 实例存、B 实例读不到。
    #:
    #: ★ 规划结果缓存不需要 redis：
    #:   它幂等（同一问题任何进程算出来一样），内存版最坏只是少省一次调用。
    cache_backend: str = "memory"
    #: 全局键前缀。和其他应用共用一个 Redis 实例时用它隔离命名空间，
    #: 保证 clear_prefix 不会误删别人的键。
    redis_key_prefix: str = "health_training:"
    #: 连接/读写超时（秒）。必须设——Redis 卡住不该拖死请求线程。
    redis_socket_timeout: float = 2.0

    # ---------------- 任务队列（P3 批量生成） ----------------
    #: memory = 进程内队列（本地开发/测试，重启即丢）
    #: rabbitmq = RabbitMQ（跨进程、重启不丢、可多 worker）
    #:
    #: ★ 什么时候必须换 rabbitmq：
    #:   批量生成成为常规操作（月度例行），或需要多 worker 分摊。
    #:   当前 2 个员工时 memory 够用 —— 端口已抽好，切换只改配置。
    queue_backend: str = "memory"
    rabbitmq_url: str = ""
    #: 单任务最大尝试次数。
    queue_max_attempts: int = 3
    #: 消费者预取数 = 同时处理上限。
    #: ★ 每个任务是一次 LLM 调用，并发太高会打满配额、吃紧 8GB 显存。
    queue_prefetch: int = 2
    #: 是否在 Web 进程内启动消费者。
    #:
    #: ★★ 默认 `true`（单机够用），但**多进程部署必须关掉**：
    #:   uvicorn --workers 4 会起 4 个消费者，加上独立的 worker 进程，
    #:   同一批任务会被抢多次。虽然幂等能兜住不重复生成，
    #:   但会白跑 LLM、浪费配额。
    #:
    #: 正确做法：Web 进程设 `QUEUE_CONSUMER_IN_WEB=false`，
    #:          另起独立 worker 进程（`scripts/run_worker.py`）。
    queue_consumer_in_web: bool = True

    # ---------------- 文件存储 ----------------
    file_storage_root: str = "./data"
    file_storage_max_mb: int = 20

    # ---------------- 向量 collection 名 ----------------
    coll_training: str = "health_training_docs"
    coll_case: str = "health_case_library"
    coll_question: str = "health_question_bank"
    coll_self_test: str = "health_self_test"

    # ---------------- 内容资产 ----------------
    #: 情景案例目录（P5 演练用）。
    #: ★ 改了案例 markdown 需重启服务才生效（接口层有进程内缓存）。
    case_dir: str = r"D:\项目\某健康公司培训系统\06_正式内容\02_情景案例"

    # ---------------- 认证（P8） ----------------
    #: 令牌签名密钥。
    #:
    #: ★ 生产环境必须从环境变量注入（写进 .env，且不进版本库）。
    #:   这里的默认值只用于本地开发：长度 >=32，但不安全。
    #:   启动自检会警告「用的是默认密钥」。
    auth_secret: str = "dev-only-insecure-secret-change-me-32bytes"
    #: 令牌有效期（秒），默认 8 小时（一个工作日）
    auth_token_ttl: int = 8 * 3600
    #: 是否强制要求登录。
    #:
    #: ★ 默认 **True**（安全优先）。
    #:   本地调试/自动化测试可通过环境变量 AUTH_REQUIRED=false 关掉，
    #:   关闭时未带令牌的请求按 X-Debug-Role 头处理。
    #:   生产环境绝不可关闭。
    auth_required: bool = True

    #: 接口层配置
    http_host: str = "127.0.0.1"
    http_port: int = 8000
    #: 允许跨域的来源。开发期允许全部；生产要收窄。
    cors_origins: tuple[str, ...] = ("*",)

    # ---------------- 运行 ----------------
    log_level: str = "INFO"
    app_env: str = "local"

    # ---------------- 派生属性 ----------------
    @property
    def milvus_uri(self) -> str:
        """Milvus 连接串。"""
        return f"http://{self.milvus_host}:{self.milvus_port}"

    @property
    def storage_root_path(self) -> Path:
        """文件存储根目录的绝对路径。相对路径按项目代码根目录解析。"""
        p = Path(self.file_storage_root)
        return p if p.is_absolute() else (BASE_DIR / p).resolve()

    @property
    def file_storage_max_bytes(self) -> int:
        """文件大小上限（字节）。"""
        return self.file_storage_max_mb * 1024 * 1024

    @property
    def ollama_configured(self) -> bool:
        """Ollama 是否配置完成（只查配置项，不探测服务）。

        ★ 不在这里 ping 服务：配置检查会被装配层和自检多次调用，
        每次都发 HTTP 会让启动变慢。连通性由 `OllamaAdapter.is_available()` 负责。
        """
        return bool(self.ollama_base_url.strip() and self.ollama_model.strip())

    @property
    def local_llm_configured(self) -> bool:
        """本地模型是否配置了可用路径（只查目录与 config.json，不做加载）。"""
        path = (self.local_model_path or "").strip()
        if not path:
            return False
        p = Path(path)
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        return p.is_dir() and (p / "config.json").is_file()

    @property
    def llm_enabled(self) -> bool:
        """是否有可用的 LLM（云端配了 Key，或本地模型路径有效）。"""
        if self.llm_primary == "local":
            return self.local_llm_configured
        return bool(self.qwen_api_key.strip() or self.deepseek_api_key.strip())

    @property
    def llm_primary_available(self) -> bool:
        """主模型是否可用。"""
        if self.llm_primary == "local":
            return self.local_llm_configured
        if self.llm_primary == "qwen":
            return bool(self.qwen_api_key.strip())
        return bool(self.deepseek_api_key.strip())

    @property
    def llm_fallback_available(self) -> bool:
        """备用模型是否可用。

        注意：主备是同一个模型时没有兜底意义，视为不可用。
        """
        if not self.llm_fallback_enabled:
            return False
        backup = "deepseek" if self.llm_primary == "qwen" else "qwen"
        if backup == "deepseek":
            return bool(self.deepseek_api_key.strip()) and self.llm_primary != "deepseek"
        return bool(self.qwen_api_key.strip()) and self.llm_primary != "qwen"

    def llm_endpoints(self) -> list[dict[str, str]]:
        """按优先级返回 LLM 端点列表。

        第一项是主模型，后续是备用。
        这样 `FallbackLanguageModelAdapter` 只需顺序遍历，不用关心谁是主备。
        """
        all_eps = {
            "qwen": {
                "name": "qwen",
                "api_key": self.qwen_api_key,
                "base_url": self.qwen_base_url,
                "model": self.qwen_model,
            },
            "deepseek": {
                "name": "deepseek",
                "api_key": self.deepseek_api_key,
                "base_url": self.deepseek_base_url,
                "model": self.deepseek_model,
            },
        }

        # Ollama 不靠 api_key 判定可用性（本地服务，模型名即配置）
        # ★ 放在最前：LLM_PRIMARY=ollama 时它是主模型
        if self.llm_primary == "ollama":
            out_ollama: list[dict[str, str]] = [
                {
                    "name": "ollama",
                    "kind": "ollama",
                    "api_key": "",
                    "base_url": self.ollama_base_url,
                    "model": self.ollama_model,
                }
            ]
            # Ollama 服务挂了时回落云端（若配了 Key）
            if self.llm_fallback_enabled:
                for name, ep in all_eps.items():
                    if ep["api_key"].strip():
                        out_ollama.append({**ep, "kind": "http"})
            return out_ollama

        # 本地模型不靠 api_key 判定可用性，单独走一条
        if self.llm_primary == "local":
            if not self.local_llm_configured:
                return []
            out: list[dict[str, str]] = [
                {
                    "name": "local",
                    "kind": "local",
                    "api_key": "",
                    "base_url": "",
                    "model": self.local_model_path,
                }
            ]
            # 本地失败时回落云端（若配了 Key）——本地 OOM 而非网络问题，兜底更必要
            if self.llm_fallback_enabled:
                for name, ep in all_eps.items():
                    if ep["api_key"].strip():
                        out.append({**ep, "kind": "http"})
            return out

        out: list[dict[str, str]] = []
        primary = all_eps.get(self.llm_primary)
        if primary and primary["api_key"].strip():
            out.append({**primary, "kind": "http"})

        if self.llm_fallback_enabled:
            for name, ep in all_eps.items():
                if name == self.llm_primary:
                    continue
                if ep["api_key"].strip():
                    out.append({**ep, "kind": "http"})
            # 本地模型作为最终兜底（云端全挂时仍可答）
            if self.local_llm_configured:
                out.append(
                    {
                        "name": "local",
                        "kind": "local",
                        "api_key": "",
                        "base_url": "",
                        "model": self.local_model_path,
                    }
                )

        return out

    def required_fields_missing(self) -> list[str]:
        """返回缺失的必填项（运维排查用）。

        P0 阶段所有项都有默认值，所以正常情况下返回空。
        连接类配置一旦改成无默认值，这里就是第一道防线。
        """
        missing: list[str] = []
        if not self.neo4j_uri:
            missing.append("NEO4J_URI")
        if not self.neo4j_database:
            missing.append("NEO4J_DATABASE")
        if not self.embed_model_dir:
            missing.append("EMBED_MODEL_DIR")
        return missing


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。测试里可用 `get_settings.cache_clear()` 重置。"""
    return Settings()
