# 健康食品门店培训系统 · 技术设计文档（TDD，反推版）

> **文档性质**：依据现有代码反向推导的系统技术设计说明，用于交接、评审与后续演进。
> **代码基线**：`项目代码/` v0.5.0（FastAPI + Python 3.11）
> **生成日期**：2026-10-31
> **范围**：应用系统本身（问答 / 演练 / 试卷 / 员工日报 / 认证）。**不含模型微调**。
> **原则**：以代码与测试为事实来源；阶段文档仅作背景，冲突时以代码为准。

---

## 1. 设计目标与约束

| 目标 | 技术落点 |
|---|---|
| 合规判定必须可复现、可解释 | 护栏做成纯领域规则，**不用 LLM 判合规** |
| 无依据不作答 | 召回阈值 + `failed` 标记，在用例层短路，不进入生成 |
| 下游故障不影响主流程 | 三路召回单路降级、LLM 主备切换、端口缺失优雅降级 |
| 业务逻辑可单测 | DDD 四层 + 端口/适配器，领域层零框架依赖 |
| 权限不可绕过 | 三类权限分别在检索层、用例层、接口层生效 |
| 单人本机可演示 | 进程内缓存/会话 + 一键启动脚本，不强依赖 Redis |

**核心约束（贯穿全文的三条红线）**：

1. 召回为空必须拒答，且不调用 LLM。
2. 输出护栏是必经节点（含流式）。
3. 普通食品不得有任何功能声称。

---

## 2. 架构总览

### 2.1 分层

```
┌─────────────────────────────────────────────────────────────┐
│ interfaces/         FastAPI 路由、Schema、鉴权依赖、静态前端    │
│                     只允许依赖 application / domain / config   │
├─────────────────────────────────────────────────────────────┤
│ application/        用例编排（Use Case）、应用服务、DTO         │
│                     只依赖 domain 端口，不碰具体实现            │
├─────────────────────────────────────────────────────────────┤
│ domain/             值对象、端口(ABC)、领域服务、规则、异常      │
│                     零框架、零 IO、零外部依赖                  │
├─────────────────────────────────────────────────────────────┤
│ infrastructure/     Neo4j / Milvus / MySQL / BM25 / LLM / 存储  │
│                     实现 domain 端口                          │
├─────────────────────────────────────────────────────────────┤
│ config/             配置、组装根(Composition Root)、启动自检    │
│                     唯一允许跨层 import 的地方                 │
└─────────────────────────────────────────────────────────────┘
```

依赖方向严格单向：`interfaces → application → domain ← infrastructure`，`config` 负责把实现注入端口。三条架构门禁在 CI 中强制（AST 级，不是 grep）。

### 2.2 门禁（`scripts/ci_gate.py` + `tests/test_architecture.py`）

| 门禁 | 规则 | 目的 |
|---|---|---|
| G1 | `domain/` 不得 import 任何框架（fastapi/neo4j/pymilvus/torch/pydantic…） | 领域层可脱离环境单测 |
| G2 | `interfaces/` 不得 import `infrastructure/` | 换实现不改路由；接口可用假实现测 |
| G3 | 安全判定逻辑不得散落在 `domain/` 以外（词根 `contraindicat`/`forbidden_claim`/`allergen`） | 防止「两处规则漂移」 |

G3 配了**文件+方法两级白名单**：`infrastructure/graph/neo4j_graph_adapter.py::check_contraindication` 放行（它只是查数据，不做判定）。白名单只按方法名放行会被任意同名函数绕过，因此必须限定文件路径。

### 2.3 运行时拓扑

```
浏览器（原生 HTML/CSS/JS，无构建）
   │  HTTP / SSE
   ▼
FastAPI（uvicorn，127.0.0.1:8000）
   ├── Neo4j  jiankangshujuku   156 节点 / 184 关系（产品事实、禁忌、用法）
   ├── Milvus 4 collections     580 块（培训 293 / 案例 117 / 题库 140 / 自测 30）
   ├── MySQL  jiankangshujuku   9 张业务表
   ├── bge-m3 本地模型          1024 维，进程内加载，启动预热
   └── LLM   千问(qwen-plus) 主 → DeepSeek(deepseek-chat) 备
```

---

## 3. 技术选型

| 层 | 选型 | 理由（代码中已写明） |
|---|---|---|
| Web 框架 | FastAPI + uvicorn | 自动 OpenAPI 文档；SSE 用 `StreamingResponse` 足够 |
| 配置 | pydantic-settings | `.env` + 环境变量覆盖；唯一允许出现默认连接串的地方 |
| 图库 | Neo4j 5.x + APOC | 禁忌/替代是典型多跳关系，一条边能查到向量检索做不到的结论 |
| 向量库 | Milvus 2.5（standalone） | 需要标量字段过滤（权限）+ HNSW 向量索引 |
| 向量模型 | bge-m3（1024 维，4096 长度） | 中文语义；长文分桶编码避免 OOM |
| 关键词 | 自研中文 2-gram BM25 | 精确匹配批准文号、产品名；无需 ES |
| 业务库 | MySQL 8 + pymysql + SQLAlchemy | 事务、唯一约束（日报每人每天一条、卷号唯一） |
| LLM | OpenAI 兼容协议 | 千问与 DeepSeek 都是兼容端点，一套适配器两种配置 |
| 缓存/会话 | 进程内实现 | 培训场景并发低；`CachePort` 已就绪，装 Redis 即可替换 |
| 前端 | 原生 JS，无框架 | 无构建、改完刷新；4 个业务页 + 登录足够 |
| 测试 | pytest + 自建替身（7 个 fake） | 领域层秒级反馈；集成测试打 `integration` 标记默认跳过 |

---

## 4. 模块设计

### 4.1 端口契约（domain/ports）

7 个基础端口 + 2 个认证端口，全部 ABC、只有签名。

| 端口 | 方法数 | 实现 | 关键契约 |
|---|---|---|---|
| `GraphQueryPort` | 9 | Neo4j | 查事实/功效/别名/禁忌/相似；`check_contraindication` 只查不判 |
| `VectorSearchPort` | 3 | Milvus | `search` 必须支持 `allowed_permissions` 拼进 `filter_expr` |
| `KeywordSearchPort` | 2 | BM25 | `build_index` 返回索引块数 |
| `LanguageModelPort` | 4 | 千问+DeepSeek | `generate` / `classify` / `generate_stream` / `is_available` |
| `RepositoryPort` | 14 | MySQL | 演练、日报、试卷、题库查询、员工、未命中、审计 |
| `CachePort` | 5 | 内存 | get/set/delete/exists/clear_prefix |
| `FileStoragePort` | 4 | 本地磁盘 | 路径逐段校验，防穿越 |
| `PasswordHasherPort` | 2 | bcrypt | hash / verify |
| `TokenSignerPort` | 2 | HMAC 签名 | issue / verify，8h TTL |

**为什么端口方法这么设计**：用例只依赖签名，因此 `tests/fakes/` 的假实现能覆盖全部分支（降级、空召回、护栏拦截、LLM 全挂）。

### 4.2 用例（application/use_cases）

| 用例 | 文件 | 职责 | 是否调 LLM |
|---|---|---|---|
| 问答 | `answer_question.py` | 意图 → 召回 → 分区组装 → 生成 → 护栏（+重试）→ 拒答/返回 | 是 |
| 检索 | `retrieve.py` | 三路并行召回、阈值过滤、RRF 融合、降级判定 | 否 |
| 多跳问答 | `query_plan.py` | 拆子查询 → 逐跳执行 → 证据收集 → 合并 | 仅在普通跳 |
| 演练会话 | `drill_session.py` | 脚本化状态机，推进客户台词 | 否 |
| 演练评阅 | `review_drill.py` | 风险定位 + 要素覆盖 → 摘要卡 | 否 |
| 组卷 | `build_paper.py` | 取题 → 排除窗口内已用 → 纯函数组卷 → 落库 → 记录使用 | 否 |
| 员工/日报 | `employee.py` | 导入、列表、日报保存/提交、评语、未命中清单 | 否 |
| 认证 | `auth.py` | 登录、签发令牌、解析身份（每请求回库取角色） | 否 |

### 4.3 领域服务与规则（domain）

| 模块 | 作用 | 关键数字 |
|---|---|---|
| `rrf_fusion.py` | RRF 融合 + 意图权重表 | k=60；13 套权重，每组和为 1.0 |
| `intent_classifier.py` | 规则意图分类 | 13 标签；置信度 0.35/0.60 起步，上限 0.95 |
| `query_planner.py` | 多意图拆分 + 依赖标注 | 最多 3 步；识别「其中/那个/最便宜」等指代 |
| `output_guardrail.py` | 三项检测统一入口 + 标注 | 42 条禁用规则 / 6 类 |
| `stream_guard.py` | 流式分句护栏 | 缓冲上限 200 字强制切分 |
| `paper_builder.py` | 纯函数组卷 | 覆盖优先，难度降级写 warnings |
| `case_parser.py` | 解析案例 Markdown | 标题层级 `#{2,4}`（两个源文件层级不同） |
| `risk_locator.py` | 风险定位到「轮/句」 | 返回 `turn_index` + `sentence_index` |
| `element_check.py` | 必答要素覆盖判定 | 严格匹配 + 兜底宽松匹配（兜底项进待确认） |
| `demand_matcher.py` | 口语需求 → 目录功能项 | 用于「改善体脂」这类说法 |
| `forbidden_claims.py` | 六类禁用表述 | 疾病治疗 / 药物替代 / 效果承诺 / 绝对化 / 经验背书 / 无依据权威暗示 |
| `product_category.py` | 普通食品功能声称准入 | 61 个禁词 + 功效语境判定 |
| `warning_text.py` | 法定警示语 | 「保健食品不是药物，不能代替药物治疗疾病」+ 7 种等价片段 |
| `data_permission.py` | 数据权限判定 | 改正文 vs 写评语严格区分 |

### 4.4 适配器（infrastructure）

| 适配器 | 说明 |
|---|---|
| `neo4j_graph_adapter.py` | 9 个查询方法；`severity` 未知值降级为 `critical`（fail-safe） |
| `milvus_search_adapter.py` | 权限过滤拼 `permission_level in [...]`；模型惰性加载（首次 `embed` 才 import ST） |
| `bm25_adapter.py` | 中文 2-gram；索引来源是 Milvus 培训库（与向量路同源） |
| `openai_compatible_adapter.py` | 一套代码适配两个厂商端点 |
| `fallback_adapter.py` | 端点级故障切换；全挂抛异常，**绝不返回空串** |
| `mysql_repository_adapter.py` | 仓储实现，含试卷快照批量写入 |
| `local_file_storage_adapter.py` | 输入逐段校验后拼接（防 `submissions/a/../../evil.pdf`） |
| `case_loader.py` | 扫描案例目录，调 `case_parser` |
| `inmemory_cache_adapter.py` | TTL 缓存 |

---

## 5. 关键链路设计

### 5.1 知识问答（非流式）

```
POST /api/ask
  │
  ├─ 1. 鉴权：current_identity → Role（回库取最新角色）
  ├─ 2. 意图：_resolve_intent()
  │       显式传入 > 规则分类器 > 未知标签退回「开放问答」
  ├─ 3. 召回：RetrieveUseCase
  │       ├ 图谱：抽产品名 → resolve_alias → get_product_facts（不参与排名融合）
  │       ├ 向量：search(allowed_permissions=[public,internal])，Top1 < 0.50 视为无效
  │       └ 关键词：BM25，绝对分 ≥ 14.0 且 ≥ 最高分×0.35
  │       → RRF 融合（仅向量+关键词）→ fused_limit=10
  │       → 三路全无效 ⇒ failed=True
  ├─ 4. 短路：failed → 拒答模板 + 写 miss_log（FR-047），**不调 LLM**
  ├─ 5. 组装：结构化事实（权威分区）→ 文档块（辅助分区），剔除「错误示范」行
  ├─ 6. 生成：LLM generate（temperature 低）
  ├─ 7. 护栏：OutputGuardrail.check
  │       禁用词 → 普通食品声称 → 警示语
  │       拦截 ⇒ 重试 1 次；仍拦截 ⇒ BLOCKED_TEMPLATE（说明原因，不复述原文）
  └─ 8. 返回：answer（合规正文）+ annotation（违规句标注，仅前端）+ meta
```

**为什么图谱不参与排名融合**：图谱返回 `ProductRef`（结构化事实），向量/关键词返回 `chunk_id`（文本块）。混在一起会产生 `chunk=None` 的条目，且「这个产品对不对」与「这段文档相关不相关」是两种语义。

**为什么必须有阈值**：向量永远返回 Top-N。没有 `MIN_VECTOR_SCORE=0.50`，拒答逻辑永远触发不了，LLM 会拿着无关段落编答案。

### 5.2 流式问答

```
LLM token 流
  → SentenceBuffer（按 。！？；\n 切句，>200 字强制切）
  → 逐句 OutputGuardrail（禁用词 + 类别声称）
      ├ 通过 → SSE event: {"kind":"text"}
      └ 不通过 → 立即截断，后续 token 全部丢弃
  → 流结束 → final_verdict（补查警示语）
  → SSE event: {"kind":"done", refused, meta}
```

**两个必须处理的细节**：

1. 警示语不能逐句判（通常只在最后一句出现），否则首句就被拦。
2. 已 emit 的每一句都过了护栏，因此「护栏必经」在流式下仍成立。

**已知限制**：流式不支持违规句标注（需重构 `stream_guard`），前端提示可切非流式。

### 5.3 多跳问答

```
QueryPlanUseCase.execute(question)
  ├ 规划：QueryPlanner → [SubQuery(step0 独立), SubQuery(step1 依赖)]
  │        规划失败 ⇒ 按单步处理（不能因此答不了）
  ├ 逐跳：
  │   step0 正常召回 → product_ids = [P001, P002]
  │   step1 若 depends_on_previous：
  │         candidate_ids = step0.product_ids
  │         为空 ⇒ 「上一问没有找到符合条件的产品，所以无法比价」
  │         禁止自己全库检索（否则会答成「最便宜的维生素C」，错得有理有据）
  ├ 证据：_gather_evidence 把「目录里有没有这个功能」「在售产品有哪些」变成
  │        权威事实句，注入 prompt 最高优先分区
  ├ 比价：纯模板 + 图谱价格，不经 LLM；最后过护栏
  └ 合并：单跳原样返回；多跳分段；普通食品必须加注（notes + 正文）
```

**性能约束**：单步问题不得退化成多跳（多一步就多一次召回 + 一次生成）。

### 5.4 演练会话与评阅

```
POST /api/drill/start  → CaseCatalog.get(case_id) → DrillSession(spec)
                            ↓ 存 DrillSessionStore（LRU 200，TTL 30min）
POST /api/drill/answer → session.answer(text) 推进状态机
POST /api/drill/review → ReviewDrillUseCase.execute(spec, turns)
                            ├ risk_locator：正则匹配 → (轮, 句, 级别, 类型)
                            ├ element_check：必答要素覆盖 + 证据片段
                            ├ pending_human：兜底判定 / 建议要素部分满足 / high 风险
                            └ SummaryCard（无 score、无 verdict）
```

**为什么客户台词不用 LLM**：同一份答卷两次演练难度不同 → 学员间不可比；追问跑偏会漏考察点；申诉时说不清客户到底问了什么。

**为什么评阅不用 LLM**：判定标准写在案例文档里（要素 + 正则），LLM 判会「今天说到了、明天说没到」，学员无法信服也无法复核。

### 5.5 组卷

```
POST /api/exam/build
  ├ 题库：Milvus health_question_bank → build_bank（解析题干/选项/答案，坏题跳过并记日志）
  ├ 排除：question_usage 查该学员 30 天内用过的题
  ├ 组卷（纯函数 build_paper）
  │   ① 每知识点保 1 题（硬约束）
  │   ② 按难度配额补齐（软约束，不足则降级 + warnings）
  │   ③ 轮转知识点补足题量
  ├ 落库：exam_paper + exam_paper_question（题目快照）
  ├ 记录使用：question_usage（**组卷成功后**才写）
  └ 返回：PaperView（含 warnings、覆盖知识点、难度分布）
```

**卷号规则**：`{stage}-{YYYYMMDD}-{三位序号}`，同日顺延，唯一索引兜底。

---

## 6. 数据设计

### 6.1 Neo4j 图模型

| 节点 | 数量 | 关键属性 |
|---|---|---|
| Product | 20 | id、name、category（保健食品/普通食品）、price |
| Alias | 35 | 口语名 → 产品 |
| Usage | 20 | dosage / daily_limit / note |
| Contraindication | 18 | 不适宜人群 + **reason 必填** |
| Efficacy | 15 | 12 项批准功能 + 3 项营养素补充 |
| Ingredient | 14 | 12 种原料 + 2 种禁忌中药 |
| Rule | 10 | 合规规则 |
| TargetGroup / Scenario / Need | 8 / 8 / 7 | 人群、场景、需求 |
| WarningMark | 1 | 普通食品「禁止功能声称」 |

| 关系 | 数量 | 约束 |
|---|---|---|
| CONTRAINDICATED_FOR | 47 | 必带 reason，severity ∈ low/medium/high/critical |
| ALIAS_OF | 35 | 别名唯一指向 |
| DOSED_AS | 20 | 每产品必有 |
| CLAIMS | 18 | **普通食品必须 0 条** |
| SUITABLE_FOR / USED_IN / HAS_INGREDIENT | 17 / 15 / 12 | — |
| SIMILAR_TO | 4 | 只存客观差异，不存「哪个更好」 |
| INCOMPATIBLE_WITH / HAS_WARNING | 3 / 5 | 配伍禁忌、普通食品标记 |

**数据纪律**：Efficacy 只能存目录原文；禁忌必须有 reason；相似关系只存客观差异。七项完整性校验全部通过（`verify_final.py`）。

### 6.2 Milvus

4 个 collection，统一 schema（`chunk_id / text / source_* / doc_id / title / category / permission_level / review_status / data_origin / section / chunk_index / extra`），题库额外带 `knowledge_point / kp_name / question_type / difficulty`。

- 索引：`vector` HNSW（M=16, efConstruction=200，COSINE）；标量字段 `permission_level / category / doc_id / knowledge_point` 建 INVERTED。
- 分块：培训 C3 清洗 + 标题切；案例按「一、二、三…」分段；题库每题一块；自测题与参考答案分流到 `health_self_test`（restricted）。
- 关键修正：案例分段正则必须是 `#{2,4}`（两个源文件标题层级不同，写死 `###` 会让 14 个案例退化成整块）；答案块不能留在 public（否则学员能直接搜到答案，违反 FR-050）。

### 6.3 MySQL（9 张表）

| 表 | 主键/唯一 | 用途 | 关键设计 |
|---|---|---|---|
| `employee_account` | employee_no | 账号与凭据 | bcrypt 哈希；`active` 停用 |
| `audit_log` | id | 审计 | 无任何凭据字段；`idx_actor_time` |
| `employee` | employee_no | 档案 | 姓名、入职、岗位、团队、带教 |
| `exam_paper` | paper_no 唯一 | 试卷主表 | `spec_json` 存组卷参数快照 |
| `exam_paper_question` | (paper_no, seq) 唯一 | 题目快照 | 存完整副本，FR-039 |
| `question_usage` | id | 防重复依据 | `idx_trainee_time` |
| `drill_record` | drill_id | 演练记录 | `card_json` 存摘要卡快照 |
| `daily_summary` | (trainee_no, summary_date) 唯一 | 日报 | 提交后冻结，记 `submitted_at` |
| `miss_log` | id | 未命中 | 供补内容；不做绩效 |

字符集统一 `utf8mb4_unicode_ci`（题目含 ★、「」等符号）。

---

## 7. 接口设计

### 7.1 REST 清单（25 个业务接口 + 首页）

| 方法 | 路径 | 权限 | 说明 |
|---|---|---|---|
| GET | `/api/health` | 公开 | 端口就绪、LLM 端点、护栏规则数 |
| POST | `/api/ask` | `ask` | 问答（非流式） |
| POST | `/api/ask/stream` | `ask` | SSE 流式 |
| GET | `/api/cases` | `drill` | 案例列表 |
| GET | `/api/cases/{case_id}` | `drill` | 案例详情 |
| POST | `/api/drill/start` | `drill` | 开始演练 |
| POST | `/api/drill/answer` | `drill` | 提交一轮 |
| POST | `/api/drill/review` | `drill` | 评阅 |
| POST | `/api/exam/build` | `exam_build` | 组卷 |
| GET | `/api/exam/papers` | `exam_manage` | 试卷列表 |
| GET | `/api/exam/papers/{no}` | `exam_manage` | 试卷详情 |
| POST | `/api/exam/papers/{no}/status` | `exam_manage` | 状态流转 |
| POST | `/api/exam/papers/{no}/export` | `exam_manage` | 导出 Markdown |
| GET | `/api/employees` | `employee_self` / `employee_view` | 员工列表（学员强制本人） |
| POST | `/api/employees/import` | `employee_import` | 批量导入 |
| POST | `/api/summaries` | `daily_write` | 保存/提交日报 |
| GET | `/api/summaries` | `daily_write` | 日报列表 |
| POST | `/api/summaries/review` | `daily_review` | 写评语 |
| GET | `/api/miss-logs` | `miss_log_view` | 未命中清单 |
| POST | `/api/auth/login` | 公开 | 登录 |
| GET | `/api/auth/me` | 登录态 | 当前身份 |
| GET | `/api/accounts` | `employee_view` | 账号列表 |
| POST | `/api/accounts/role` | 主管 | 改角色 |
| POST | `/api/accounts/active` | 主管 | 启停用 |
| GET | `/api/audit-logs` | `audit_log_view` | 审计日志 |

（计数：问答 2 + 演练 5 + 试卷 5 + 员工日报 6 + 认证账号审计 6 + 健康检查 1 = 25；另有 `/` 静态首页。）

### 7.2 SSE 协议

```
data: {"kind":"text","text":"康源牌鱼油软胶囊（P001）是保健食品，","ok":true}
data: {"kind":"text","text":"其批准功能为：…","ok":true}
data: {"kind":"done","refused":false,"meta":{...}}
```

### 7.3 错误模型

统一响应体：`{code, message, detail}`。

| 异常 | HTTP | code |
|---|---|---|
| `AuthenticationError` | 401 | `AUTHENTICATION_ERROR` |
| `PermissionDeniedError` | 403 | `PERMISSION_DENIED` |
| `ProductNotFoundError` / `CaseNotFoundError` / `QuestionNotFoundError` | 404 | `*_NOT_FOUND` |
| `ExamPaperLockedError` / `DuplicateDailySummaryError` | 409 | `EXAM_PAPER_LOCKED` / `DUPLICATE_DAILY_SUMMARY` |
| `DraftAlreadySubmittedError` | 403 | `DRAFT_ALREADY_SUBMITTED` |
| `UnsupportedFileTypeError` / `FileTooLargeError` | 415 / 413 | `UNSUPPORTED_FILE_TYPE` / `FILE_TOO_LARGE` |
| `PathTraversalError` | 400 | `INVALID_PATH` |
| 其他 `DomainError` | 400 | `DOMAIN_ERROR` |
| 基础设施异常（按类名+MRO 判定） | 503 | `INFRASTRUCTURE_ERROR` |
| 未捕获异常 | 500 | `INTERNAL_ERROR` |

**两个易错点**：401 与 403 必须分开（前端处理不同）；注册 `Exception` 兜底后必须放行 `HTTPException`，否则 FastAPI 的 404/422 全变 500。

---

## 8. 横切设计

### 8.1 配置

- 唯一配置源 `config/settings.py`，`.env` 覆盖，进程内单例 `get_settings()`。
- 路径全部按项目根推导（`BASE_DIR.parent / "models/bge-m3"`），换机器只改 `.env`。
- 凭据不出现在代码与日志；`required_fields_missing()` 供运维排查。

### 8.2 组装与生命周期

- `config/assembly.py` 是唯一跨层 import 处：读配置 → 建连接 → 建适配器 → 注入 `Container`。
- `_try_build` 捕获 ImportError/异常并记入 `Container.pending`，实现「缺哪个报哪个」而非启动即崩。
- `Container` 在 FastAPI `lifespan` 中装配一次（bge-m3 14s + BM25 1s），测试用 `set_container()` 注入假容器（接口测试从 68s 降至 1.27s）。

### 8.3 降级策略

| 故障 | 行为 |
|---|---|
| 图谱不可用 | 该路返回空并记 `degraded`，向量/关键词继续 |
| 关键词不可用 | BM25 预热失败只告警，服务照常启动 |
| 向量不可用 | 仅剩关键词与图谱；仍无有效结果则拒答 |
| LLM 主端点故障 | 超时/429/5xx/401/403/解析失败 → 切备用；全挂抛异常（不返回空串） |
| MySQL 不可用 | 试卷/日报/审计相关接口 503；问答不受影响（miss_log 写失败仅告警） |
| 护栏触发 | 重试 1 次 → 说明式拒答；正文与标注分离 |

### 8.4 可观测

| 手段 | 内容 |
|---|---|
| 结构化日志 | 每请求记录意图、各路命中、耗时、是否降级 |
| `meta` 字段 | 回答可解释（为什么这条排第一） |
| `miss_log` | 未命中问题，补知识库依据 |
| `audit_log` | 登录成功/失败、组卷、导出、权限变更 |
| `FallbackLanguageModel.stats()` | 主备成功/失败次数 |
| `/api/health` | 端口就绪、LLM 端点、护栏规则数 |

### 8.5 安全

| 面 | 措施 |
|---|---|
| 密码 | bcrypt；不存在账号也跑假哈希，消除时序侧信道 |
| 账号枚举 | 登录失败统一「工号或密码不正确」 |
| 会话 | Bearer 令牌 8h；每请求回库取角色（降职权即时生效） |
| 越权 | 功能权限白名单 + 数据权限用例层强制 + 检索层 `filter_expr` |
| IDOR | `test_idor_scan.py` 覆盖跨用户访问 |
| 路径穿越 | 存储适配器逐段校验分隔符/`..`/盘符 |
| 传输 | **当前 HTTP 明文，上线必须 HTTPS** |
| CORS | 当前 `*`，生产必须收窄 |

---

## 9. 测试策略

| 层次 | 数量级 | 做法 |
|---|---|---|
| 领域单元 | 主体 | 纯函数 + 规则表；`test_domain/` 覆盖护栏误报、否定句、引用教学、数字接地等边界 |
| 应用用例 | 中 | 7 个 fake（graph/vector/keyword/llm/repo/cache/filestorage）+ fake_clock |
| 接口 | 中 | TestClient + 假容器；覆盖 401/403、IDOR、前端鉴权头 |
| 集成 | 84 条（默认 deselect） | 真 Neo4j/Milvus/MySQL，打 `integration` 标记 |
| 架构 | 3 条门禁 + 门禁自证 | 注入违规必须 FAIL |
| 验收脚本 | 多个 | `verify_intent.py`、`verify_v2.py`、`verify_final.py`、HTTP 冒烟 |

**总量**：1257 passed / 1 skipped / 84 deselected，58 秒。  
**评测集**：36 条，R@1 0.4722 / R@3 0.6944 / R@5 0.7778 / MRR 0.6028。

**测试里的两个反模式（已被踩过）**：

1. 门禁只按方法名放行 → 任意同名函数可绕过，必须限定文件。
2. `TestClient` 进入 `with` 会触发 lifespan → 每个测试文件多等 15 秒，必须替换 `lifespan_context`。

---

## 10. 部署与运维

### 10.1 依赖服务

| 服务 | 端口 | 必需 | 备注 |
|---|---|---|---|
| Neo4j | 7687 / 7474 | 是 | 库名 `jiankangshujuku` |
| Milvus | 19530 | 是 | 4 collections |
| MySQL | 3306 | 是 | 9 张表 |
| LLM API | 公网 | 是（无 Key 则问答不可用） | 千问主 / DeepSeek 备 |
| Redis | 6379 | 否 | 不装走内存实现 |

### 10.2 启动路径

```
python 一键启动.py            # 检查并拉起 MySQL/Neo4j/Milvus → 等待 7/7 → 开浏览器
python 一键启动.py --check    # 只检查
python 一键启动.py --stop     # 只停应用
python main.py                # 自检并退出（0/1 退出码）
python serve.py               # 只起 Web（需自行保证三库可用）
```

### 10.3 数据重建

```
08_数据入库/neo4j/01~03_*.cypher → verify_final.py
08_数据入库/milvus/build_collections.py → verify_v2.py
项目代码/scripts/init_db.py → check_db.py → seed_accounts.py
```

**顺序要求**：先建库（init_db / cypher），再灌向量（依赖 `06_正式内容`），最后起应用（BM25 从 Milvus 读）。

---

## 11. 技术债与演进建议

| 优先级 | 项 | 影响 | 建议 |
|---|---|---|---|
| P0 | 无 Rerank | R@1 0.4722 | 加交叉编码器重排 Top20 → Top10；先修 R@1 退化 |
| P0 | 密码重置缺失 | 导入员工后登不进去 | 两个接口：主管重置、自助改密 |
| P1 | 会话/缓存进程内 | 多实例不共享、重启丢失 | 换 Redis（端口已就绪，装包即可） |
| P1 | 案例/题库改动需重启 | 运维不便 | 加 `POST /api/admin/reload` |
| P1 | 并发组卷卷号竞争 | 极端情况冲突 | 重试 + 重新取号，唯一索引兜底 |
| P2 | 题库向量未重算 | 组卷不走向量，暂无影响 | 题库文本变更后重建 collection |
| P2 | 流式不支持标注 | 体验不一致 | 重构 `stream_guard` 支持句级标注 |
| P2 | 多跳未做流式 | 复合问题体验差 | 同上 |
| P3 | 前端无移动端适配 | 门店多用手机 | 响应式改造 |
| P3 | 无压测 | 容量未知 | 目标：50 并发下 P95 ≤ 8s |

---

## 12. 交接速查

| 想知道 | 看这里 |
|---|---|
| 系统能做什么 | `docs/PRD_健康培训移动_反推.md` |
| 分层与依赖规则 | `scripts/ci_gate.py`、`tests/test_architecture.py` |
| 为什么这么设计 | `CHANGELOG.md`、各模块头部注释（本项目的注释密度很高，是主要设计文档） |
| 还差什么 | `项目代码/文档/未完成事项清单.md`（注意部分条目已过时） |
| 数据怎么来的 | `08_数据入库/*/入库记录.md` |
| 怎么跑起来 | `本机部署记录.md`、`部署与使用说明.md` |
| 接口契约 | `http://127.0.0.1:8000/docs` 或 `interfaces/http/schemas/models.py` |
