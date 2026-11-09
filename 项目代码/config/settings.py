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
    #: bge-m3 模型目录。
    #:
    #: ★ 默认值按「项目根目录../models/bge-m3」推导（迁移时改，2026-10-23）：
    #:   原先硬编码 `D:\ai260803\...`，换机器必报错。
    #:   你机器上模型放哪都行，用 .env 的 EMBED_MODEL_DIR 指定即可。
    embed_model_dir: str = str(BASE_DIR.parent / "models" / "bge-m3")
    expected_embed_dim: int = 1024
    # 4096 序列长度下 8GB 显存必须按长度分桶编码，否则 OOM
    embed_max_seq_length: int = 4096
    # 启动时预加载模型。
    # ★ 开启后启动慢约 14s，但**首个用户请求**从 17s 降到 3s。
    #   演示/PoC 场景必须开启；纯离线批处理可关掉。
    warmup_embedding: bool = True

    # ---------------- 稀疏召回（P0-3 结论） ----------------
    #: 是否启用 bge-m3 稀疏路（lexical_weights → Milvus SPARSE_FLOAT_VECTOR）。
    #:
    #: ★ 为什么需要（205 题全量实测，2026-11-07）：
    #:   案例库 34 题，纯向量路 R@5=21/34，而稀疏单独 25/34、两路并集 28/34。
    #:   失败模式是**表格型内容**（必答要素/权重、违禁正则、评分示例），
    #:   合并成长块后向量表征被稀释，字面词权重反而抓得住。
    #:   文档库相反（hybrid 105/131 > dense 98），所以并集只对案例库开。
    sparse_recall_enabled: bool = True
    #: 启用「RRF 头部 + 稀疏候选」并集窗口的 collection（逗号分隔）。
    #: 空串表示所有 collection 都做并集（不推荐：文档库上无收益）。
    sparse_recall_collections: str = "health_case_library"
    #: 稀疏召回的绝对下限（IP 尺度）。
    #:
    #: ★ 这不是相关性阈值，只是**sanity floor**：
    #:   实测可答/不可答查询的稀疏 top1 分数几乎完全重叠
    #:   （可答中位 0.209 / 不可答最大 0.222），任何能拦住噪声的值
    #:   都会误拒 60%+ 的正常查询。取 0.02 只拦「向量算坏了」的情况；
    #:   拒答判断仍走规则 + LLM「证据不足」路径。
    min_sparse_score: float = 0.02

    # ---------------- 窗口构图（P1-a） ----------------
    #: 启用「每路头部进窗口」的 collection（逗号分隔）。
    #:
    #: ★ 为什么需要（205 题离线模拟，2026-11-07）：
    #:   文档库的 BM25 强项是精确串（编号、流程术语），但意图权重
    #:   (0.1, 0.8, 0.1) 把 BM25 第 1 名压到 0.0016——约等于向量第 20 名的
    #:   六分之一，gold 永远进不了前 5。把每路第 1 名直接拎进窗口，
    #:   文档库 105 → 111。
    #:   案例库相反（22 < 稀疏并集 26），走 `sparse_recall_collections`；
    #:   题库 40/40 无变化，不配置即保持纯 RRF。
    #:
    #: 空串 = 全部回退纯 RRF（回滚开关）。与 `sparse_recall_collections`
    #: 重叠时稀疏并集优先（构造时会告警）。
    path_head_collections: str = "health_training_docs"

    # ---------------- 参考资料条数（P1-b） ----------------
    #: 提示词里放多少条参考资料（`retrieval.fused` 的前 N 条）。
    #:
    #: ★ 默认 5 是历史值；扩到 8/10 的覆盖收益（用 app_path_head_v1 的
    #:   gold_rank 直接算）：5 → 177、8 → 181、10 → 185，其中案例库
    #:   26 → 28 → 30。但多出来的段落是噪声，可能把答案带偏或触发护栏，
    #:   必须经 `train/scripts/95_answer_ab.py` 的答案级 A/B 才能改默认。
    #:   回滚：CONTEXT_DOCS=5。
    context_docs: int = 5

    # ---------------- 系统提示词版本（P1-c） ----------------
    #: `v3` = 八条硬约束 + 「回答前自检」四条（默认，已验证）。
    #: `v4` = v3 + 先查资料/必答要素/不复述禁用词/不串产品四条的**实验版**；
    #:   100 题 A/B 未达门禁（65 vs 64，拒答 +1），**不作默认**，仅留作实验开关。
    #: `v2` = 只有八条硬约束（回滚用）。
    #:
    #: ★ 验证（2026-11-07）：45 题 rubric 39→41（3 上 1 下，1 下为词形误报）；
    #:   独立 held-out 24 题 17→19，禁用 3→1、拒答 1→0。
    #:   代价：答案更长（+14%~37%）、延迟约 +20~30%。
    #:   回滚：PROMPT_VARIANT=v2。
    prompt_variant: str = "v3"

    # ---------------- Rerank（可选重排序） ----------------
    # ★ 默认 False：合入时零行为变更。收益经 train/scripts/81_retrieval_eval.py
    #   评估达标（R@5 ≥ 基线+3pp，见 docs/PRD_检索质量优化）后再开启。
    rerank_enabled: bool = False
    #: CrossEncoder 模型目录（bge-reranker-v2-m3，~2.2GB）。
    #: 默认按「项目根目录/models/」推导，与 embed_model_dir 同一约定。
    rerank_model_dir: str = str(BASE_DIR.parent / "models" / "bge-reranker-v2-m3")
    #: 送入 rerank 的融合候选窗口大小（RRF 融合后取前 N 条重排）。
    rerank_candidate_k: int = 20
    #: 前向批大小，控制峰值显存。
    rerank_batch_size: int = 8
    #: 模型侧最大序列长度。
    rerank_max_length: int = 512
    #: 单次调用超过该秒数记 warning（软性观测；不做硬超时，见适配器注释）。
    rerank_slow_warn_sec: float = 3.0

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
    #
    # ★ 四类后端按 `kind` 分派（见 `llm_endpoints`）：
    #   http（千问/DeepSeek）/ ollama（推荐本地）/ local（transformers+4bit）
    llm_primary: str = "qwen"                 # qwen | deepseek | local | ollama
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

    #: 是否在请求体里显式关闭推理模型的「思考」。
    #:
    #: ★ 必须默认开启（2026-11-07 实测）：qwen3.7-flash 不关思考时
    #:   单次生成约 14s、completion 1200+ tokens；max_tokens 较小时
    #:   思考会把额度耗尽导致正文为空。关闭后约 2s / 200 字。
    #:   DeepSeek 的 flash 系模型同样受益（仍会返回 reasoning_content，
    #:   但正文正常）。
    #:   若换成不支持该参数的非推理模型且被 400 拒绝，设 false 关掉。
    #:   ★ 只对 kind=http 的端点生效。
    llm_disable_thinking: bool = True

    #: 单次生成的 token 上限。
    #:
    #: ★ 默认 2048（原 1024）：推理模型在长 prompt 下可能把 1024 全花在
    #:   reasoning 上，正文为空（finish_reason=length）。2048 留出余量；
    #:   若仍偶发空正文，Fallback 适配器会把它当失败并切备用端点。
    llm_max_tokens: int = 2048

    #: 条件式答案补全（A4）。
    #: 默认 false；只有 A/B 证明「通过率提升且拒答/禁用不增」后才改默认。
    #: 只影响非流式问答；流式仍走原链路（补全需要拿到完整初稿）。
    answer_repair_enabled: bool = False

    #: 培训助教 Agent（只读版）开关。
    #: 默认 false；本地 .env 打开，生产按需开启。
    agent_enabled: bool = False

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
    queue_backend: str = "memory"
    rabbitmq_url: str = ""
    #: 单任务最大尝试次数。
    queue_max_attempts: int = 3
    #: 消费者预取数 = 同时处理上限。
    #: ★ 每个任务是一次 LLM 调用，并发太高会打满配额、吃紧显存。
    queue_prefetch: int = 2
    #: 是否在 Web 进程内启动消费者。
    #:
    #: ★★ 默认 `true`（单机够用），但**多进程部署必须关掉**：
    #:   uvicorn --workers 4 会起 4 个消费者，加上独立的 worker 进程，
    #:   同一批任务会被抢多次。虽然幂等能兜住不重复生成，
    #:   但会白跑 LLM、浪费配额。
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
    #: ★ 默认按「项目根目录../06_正式内容/02_情景案例」推导（迁移时改）。
    #:   改了案例 markdown 需重启服务才生效（接口层有进程内缓存）。
    case_dir: str = str(BASE_DIR.parent / "06_正式内容" / "02_情景案例")

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

    # ---------------- 多轮问答（B2~B4，默认关闭） ----------------
    #: 总开关。★ 默认 False：不带 `session_id`（或开关关闭）时
    #: 走原来的单轮链路，答案与响应逐字节不变；关掉即回滚。
    #: ★ 合并后生产默认 true（见 .env）；这里保持 false 以保护测试基线。
    multi_turn_enabled: bool = False
    #: 每个会话保留最近几轮（改写只读最近 3 轮，更多轮是噪声）。
    multi_turn_max_turns: int = 3
    #: 会话空闲过期时间（秒，30 分钟）。
    multi_turn_ttl_sec: int = 1800
    #: 进程内最多保留多少个会话（LRU 淘汰）。
    multi_turn_max_sessions: int = 200

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
        if self.llm_primary == "ollama":
            return self.ollama_configured
        return bool(self.qwen_api_key.strip() or self.deepseek_api_key.strip())

    @property
    def llm_primary_available(self) -> bool:
        """主模型是否可用。"""
        if self.llm_primary == "local":
            return self.local_llm_configured
        if self.llm_primary == "ollama":
            return self.ollama_configured
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

        ★ 每个端点带 `kind`：http / ollama / local，装配根据此选适配器。
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

        out = []
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
