# CHANGELOG

## P0 骨架与契约（2026-10-22）

### 新增

**四层骨架**
- `domain/` 值对象 5 个、端口 7 个、异常体系
- `application/` 基础类型（`UseCaseResult` / `QuestionIntent`）
- `infrastructure/` 2 个可用适配器 + 异常体系
- `interfaces/` 领域异常 → HTTP 映射
- `config/` settings / assembly / startup_check
- `tests/` 155 个用例 + 7 个测试替身

**7 个端口（纯 ABC，只有签名）**

| 端口 | 方法数 | 实现阶段 |
|---|---|---|
| `GraphQueryPort` | 7 | P1 |
| `VectorSearchPort` | 3 | P2 |
| `KeywordSearchPort` | 2 | P3 |
| `LanguageModelPort` | 3 | P4 |
| `RepositoryPort` | 10 | P6/P7 |
| `CachePort` | 5 | ✅ 已实现（内存） |
| `FileStoragePort` | 4 | ✅ 已实现（本地磁盘） |

**工程设施**
- `scripts/ci_gate.py`：3 条架构门禁（AST 级，非 grep）
- `tests/test_ci_gate.py`：门禁自证（注入违规 → 必须 FAIL）
- `main.py`：启动自检入口
- `.env.example` / `pyproject.toml` / `requirements.txt`

### 修复（红灯阶段抓出）

- **门禁 3 漏检**：正则词根 `contraindicated` → `contraindicat`。
  原写法漏掉最常见的 `check_contraindication()`，实测三种命名漏两种。
- **文件名路径穿越**：`LocalFileStorageAdapter` 原只在拼接后校验「是否在 root 内」，
  而 `submissions/a/../../evil.pdf` 解析后仍在 root 内、但用户已能控制落盘目录。
  改为输入逐个校验（分隔符 / `..` / 盘符）后再拼接。替身同步修复。

### 已知限制

- 5 个适配器待实现（graph / vector / keyword / llm / repository）
- `redis` 包未安装，CachePort 走内存实现（多进程不共享）
- `LLM_API_KEY` 未配置（P4 前不影响）
- `pymysql` 未安装（P6/P7 才需要）

---

## 数据层（2026-10-22）

见 `../08_数据入库/milvus/重建记录_V2.md`。

---

## P1 图谱查询（2026-10-22）

### 新增
- `infrastructure/graph/neo4j_graph_adapter.py`：`GraphQueryPort` 7 个方法
- `tests/test_infrastructure/test_graph.py`：35 条集成测试（真 Neo4j）
- `tests/test_infrastructure/test_graph_mapping.py`：29 条转换函数测试

### 修复
- **severity 转换 fail-open → fail-safe**：未知值原降级为 `medium`，
  会让 critical 禁忌逃过 P4 拦截；改为降级为 `critical`（从严）
- **门禁 3 误报**：`check_contraindication` 是端口契约的查询方法，
  不是判定逻辑；加白名单放行
- **白名单可绕过**：第一版只按方法名放行，任何位置写同名函数都能过；
  改为 `{文件路径: {方法名}}` 两级限定 + 2 条守门测试

### 契约变更
- `Usage` 值对象字段按真实库对齐：`method/frequency/timing` → `dosage/daily_limit/note`

---

## P2 向量检索（2026-10-22）

### 新增
- `infrastructure/vector/milvus_search_adapter.py`：`VectorSearchPort` 3 个方法
- `tests/test_infrastructure/test_vector.py`：28 条集成测试
- `tests/test_infrastructure/test_vector_unit.py`：23 条单元测试（不触网）

### 关键设计
- **权限过滤在检索层**（拼进 `filter_expr`），不是检索后过滤
- **模型惰性加载**：`import ST` 占启动 87%，首次 `embed()` 才加载
- 输出字段按 collection 真实 schema 动态取（`knowledge_point` 只在题库有）

### 验收
- 学员查受限库（案例/题库/自测）**均 0 条**
- R@5 = **0.778**（目标 ≥0.70）

---

## P3 三路融合（2026-10-22）

### 新增
- `domain/services/rrf_fusion.py`：RRF 融合 + 意图动态权重表
- `infrastructure/search/bm25_adapter.py`：中文 2-gram BM25
- `application/use_cases/retrieve.py`：三路并行召回 + 降级 + 阈值

### 关键设计
- **单路失败降级**，三路全挂才 `failed=True`
- **分数阈值 `MIN_VECTOR_SCORE=0.50`**：向量永远有 Top-N，
  不设阈值则 P4 拒答永远触发不了
- **图谱结果不参与文档排名**：产品事实走事实卡，与文档块是两类东西

### 验收
- R@5：0.722 → **0.833（+15.4%）**，勉强达标
- R@1：0.528 → 0.500（RRF 固有取舍，已成文记录）

---

## P4 输出护栏（2026-10-22）

### 新增
- `domain/rules/forbidden_claims.py`：六类禁用表述，35 条规则
- `domain/rules/product_category.py`：普通食品 61 个禁词
- `domain/rules/warning_text.py`：法定警示语 + 7 种等价表述
- `domain/services/output_guardrail.py`：三项检测统一入口
- 测试 198 条（68 基础 + 68 误报回归 + 63 安全验收）

### 修复（真实语料扫描发现）
- 「第一条铁律」「## 第一步」被误判绝对化
- 「您最好先咨询医生」（合规建议）被误判绝对化
- `第一选择`、`销量第一`、`效果最强`、`见效最快` 漏检
- 误报率 10.9% → 9.0%（剩余为「引述违规示例」类，内容特性）

### 验收
- 20 个越界问题拦截率 **100%**
- 普通食品功能声称拦截 **100%**
- 保健食品缺警示语拦截 **100%**
- 合规回答不误拦 10/10

### 未完成（阻塞）
- LLM 生成部分：`LLM_API_KEY` 未配置

---

## P4 LLM 主备双通道（2026-10-22）

### 新增
- `infrastructure/llm/openai_compatible_adapter.py`：千问 / DeepSeek 通用（urllib，无 SDK 依赖）
- `infrastructure/llm/fallback_adapter.py`：主备切换 + 重试 + 统计 + 留痕
- `config/settings.py`：`llm_endpoints()` 按 `LLM_PRIMARY` 生成有序端点
- 测试 35 条（19 假替身 + 16 真实 API）

### 配置
- 主 qwen-plus / 备 deepseek-chat，双 Key 实测可用
- 换主备只改 `LLM_PRIMARY`，代码零改动

### 切换策略
- 切：超时 / 429 / 5xx / 401 / 格式错
- 不切：重试成功（避免同会话风格突变）
- 全挂：抛异常，**不返回空串**（空串会掩盖故障）

### 修复
- `stats()` 缺未调用端点 → 监控会 KeyError
- 切换成功时不记 `last_error` → 运维看不到降级发生过

---

## P4 问答链路（2026-10-22）

### 新增
- `application/use_cases/answer_question.py`：召回 → 分区组装 → 生成 → 护栏
- 装配根预热：`_warmup_bm25()` + `_warmup_embedding()`
- 测试 63 条（16 用例 + 47 引述回归）

### 红线验证
- **召回为空不调 LLM**：`FailOnCallLLM` + 真机验证（54ms，无生成）

### 修复（端到端真机暴露的 5 个问题）
1. **引述语境被误拦**：合规回答里的 `❌ 不能说"降血脂"` 被当违规
   → 加 `_QUOTED_NEGATION`（只放行「否定词 + 引号片段」）
2. **劝阻性表述被误拦**：`不可擅自停药或替代治疗` → `_DISSUASION`
3. **否定性陈述被误拦**：`不用于治疗高血脂症` → `_NEGATED_CLAIM`（要求紧邻）
4. **BM25 索引从未构建**：静默失效，三路融合实际只有两路 → 装配根预热
5. **BM25 噪声导致编答案**：问「量子能量水」也答
   → 关键词双阈值（绝对 14.0 + 相对 35%，实测标定）

### 性能
- 预热后首请求 17s → **2.73s**
- ⚠️ **P95 < 3s 未达标**（LLM 生成 3–5s），建议上流式输出

---

## P4 流式输出优化（2026-10-22）

### 新增
- `domain/services/stream_guard.py`：SentenceBuffer + StreamGuard（分句过护栏）
- `LanguageModelPort.generate_stream`：SSE 流式能力（abstractmethod）
- `AnswerQuestionUseCase.execute_stream`：流式问答
- `application/types.py::StreamEvent`：text / done 两类事件
- 测试 39 条（26 护栏 + 13 用例）+ 5 条真实 API

### 关键指标
- 首块延迟：3.70s → **1.18s（中位）**，原生首 token 0.46s
- 护栏零泄漏（1/3/5 字多种分块验证）

### 修复
- **按句号切分几乎无收益**：首句常 60 字，等句号等于等大半篇
  → 允许在逗号处提前切（累积 24 字）
- **逗号切分破坏跨段规则**：`它不用于降血脂治疗，` 被切成两段后误拦
  → 检测带前文 20 字上下文（`CONTEXT_TAIL_CHARS`）
- **端口塞默认实现**：被门禁挡下 → 改 `@abstractmethod`

---

## P5 演练评阅内核（2026-10-22）

### 新增
- `domain/value_objects/drill_review.py`：SummaryCard / RiskItem / ElementCheck / CaseSpec
- `domain/rules/element_check.py`：必答要素覆盖判定（纯规则）
- `domain/services/risk_locator.py`：风险定位到「第几轮第几句」
- `domain/services/case_parser.py`：案例 markdown 纯文本解析
- `infrastructure/content/case_loader.py`：案例文件加载（IO）
- `application/use_cases/review_drill.py`：评阅用例
- 测试 96 条

### 关键指标
- 16 个案例 / **55 条违禁规则**，全部解析成功
- **critical 检出率 100%**（逐条验证）
- 摘要卡**无 score / verdict**（4 条测试守护）

### 修复
- **合规警示语被判违规**：`不能代替药物治疗疾病` 含「治疗」被拦
  → 复用 P4 白名单 `_strip_compliant`
- **critical 覆盖缺口**：「很多人吃了血脂都降下来了」两条规则都漏
  → 补 TS-05 / TS-06
- **案例规则表不完整** → 叠加全局六类规则
- **案例解析 4 种格式变体**（标题层级/章节名/列数/正则含 `|`）
- **domain 层做 IO** → 拆成 domain 纯解析 + infra 读文件
- **「我看看」误配「我们」** → 加 `(?![们])` + 调整判断顺序

---

## P5 演练对话（2026-10-22）

### 新增
- `application/use_cases/drill_session.py`：`DrillSession` 状态机
- 测试 35 条（22 单元 + 13 集成，覆盖 16 个真实案例）

### 核心设计
- **客户台词 100% 来自案例脚本**，只有「何时进下一句」由状态机判断
- 不让 LLM 扮演客户：难度不可控（学员无法横向比较）、
  追问会跑偏（漏 F3 就测不到禁忌识别）、无法复现（申诉说不清）
- 纯状态机，零 LLM、零 IO

### ⚠️ 发现并修掉项目级漏洞：新增第七类「替客户判断」
- 演练时「可以的，一起吃没问题」完全不报
- **P4 输出护栏也没有这个类别**（`check_forbidden_claims` 返回空）
- 风险：客户在服降压药，学员替其判断同服安全 → 可能致害
- 补 5 条规则 JD-01～JD-05，覆盖三种语序
- 护栏：35 条 6 类 → **42 条 7 类**

### 修复
- **同一句被报两次**：去重维度踩了三次
  v1 键含规则类型 → 同片段重复报
  v2 键为位置+类别 → 丢同句不同类别
  v3 键为位置，片段互为子串才合并 → 正确
  关键：同位置**互不相关**的违规必须都报
- 清理一处死代码（`_finished = ... and False`）

---

## P9 意图识别接入（2026-10-23）

### 背景
P3 就建好了 13 套意图权重表（`INTENT_WEIGHTS`），但接口层从不传 `intent`，
`answer_question` 里 `intent or "开放问答"` 让**全部问题走同一套权重**
(图谱 0.1 / 向量 0.8 / 关键词 0.1)。禁忌类问题本该让图谱主导（0.6），
实际是向量主导 —— **权重结构与业务判断相反**。

### 新增
- `domain/services/intent_classifier.py`：纯规则分类器（零依赖，~0.05ms）
- `tests/test_domain/test_intent_classifier.py`（31 条）
- `tests/test_application/test_intent_integration.py`（13 条，证明「接上了」）
- `scripts/verify_intent.py`：端到端验证（真实 Neo4j / Milvus / BM25 / LLM）
- `文档/阶段P9-意图识别.md`

### 改动
- `application/use_cases/answer_question.py`：新增 `_resolve_intent()`，
  **流式与非流式共用同一条意图路径**；meta 增 `intent_source` /
  `intent_confidence` / `intent_matched`
- `domain/services/rrf_fusion.py`：`DEFAULT_WEIGHTS` 由「产品功效」改为
  「开放问答」——原来两处兜底不一致（代码里看着是图谱主导，实际是向量主导）
- 前端问答页：显示意图 + 来源 + 三路权重 + 命中词

### 核心设计判断
- **分类器放用例层，不放接口层**：接在接口层的话，`/api/ask`、
  `/api/ask/stream`、CLI、批处理各自都要记得传 ——
  而「某个入口忘了传」正是本缺陷的成因
- **默认不调 LLM**：用例红线是「召回为空不调 LLM」。分类器在链路最前面，
  一旦调 LLM，乱码输入也会打一次模型，红线在链路上被绕过。
  LLM 兜底做成开关 `use_llm_fallback`，默认关
- **非法标签不静默兜底**：`weights_for_intent` 对未知标签返回默认权重，
  日志无异常但权重完全不同 → 显式退回开放问答并打 warning

### 红灯抓出的两个真实误判
| 现象 | 根因 | 修法 |
|---|---|---|
| 「鱼油有什么功能」判成产品名查询 | 规则 `有.{1,6}么` 的 `.` 匹配到「什」 | 负向前瞻排除「什么/啥/哪些」 |
| 「吃了能停降压药吗」判成禁忌咨询 | 规则只写了「停药」，客户说的是「能停」 | 补「能停/能不能停/可以停」 |

★ 教训：**规则要按客户的实际说法写，不按标准术语写**——
标准词在培训文档里，不在客户嘴里。

### 验收
```
1157 passed / 1 skipped（新增 44 条）
架构门禁 3 条全过 | main.py 自检 6 通过 / 1 警告
真实接口（学员账号）6 类意图权重全部符合预期
流式 / 非流式意图一致
```

融合排序确有变化（「吃了能停降压药吗」：`D009_c016` → `D004_c022` 顶到第一）。
但 `在吃阿司匹林能吃鱼油吗` Top1 未变——该问题禁忌段落向量分本来就高，
**是「碰巧对」变成「因为对所以对」，不能当成「改造没效果」**。

---

## P10 多意图拆解 + 多跳检索（2026-10-23）

### 背景
未完成事项清单里那条「多意图还没拆子查询」。
用户进一步要求：「前一句话查询的结果是后一个查询的前提，
比如改善体脂都可以吃什么保健品，其中那种最便宜。」

### 动手前查到的关键事实
```
EFF_09「有助于控制体内脂肪」     目录第 9 项，存在
声称该功能的产品数               0
P020 代餐奶昔  普通食品  HAS_WARNING「禁止功能声称」，无 CLAIMS
```
**这道题的正确答案是「本店没有」**，不是「法规不允许」——
两句话完全不同，教错就是教错法规。

### 改造前实测
- 「补钙的有什么推荐，有什么禁忌」→ 整句判成禁忌咨询，产品 `()`，推荐被答成反问客户
- 「改善体脂…最便宜」→ 回答「目录中**没有**改善体脂等表述」（**说错法规**），且完全没比价

### 新增
- `domain/services/query_planner.py`：拆子查询 + 依赖标注
- `domain/services/demand_matcher.py`：口语需求词 → 目录功能项（同义词表 + 核心词对撞）
- `application/use_cases/query_plan.py`：逐跳执行、证据收集、比价、合并
- 测试 78 条（规划 25 / 多跳 23 / 接口 9 / JD-02 双向 21）
- `文档/阶段P10-多意图与多跳.md`

### 改动
- `GraphQueryPort`：+`find_efficacy_by_keyword` / `list_efficacy_catalog`，+`EfficacyRef`
  （**必须与按产品反查分开**：目录有没有是法规问题，与库存无关）
- `answer_question.execute/execute_stream`：+`fact_hints`，进 prompt 的**分区零**（最高权威）
- `ask` 路由：+`multi_hop`（默认 true）、`hops` 展示；**修复 fact_hints/product_ids 漏透传**
- `product_category`：体脂类改组合式规则（裸词「体脂」会误拦「体脂仪」4/4）
- `forbidden_claims`：JD-02 收窄（误报「可以喝牛奶」「孕妇能喝吗」「能喝的产品」）

### 核心设计判断
- **并列 ≠ 依赖**：并列拆开可同时跑；依赖必须等前一步。
  把并列当依赖会无端缩小范围；把依赖当并列会比出第 1 步里没有的产品
- **候选集显式传递**：`depends_on_previous=True` 时候选集只来自前一步，
  为空就说「无法比价」，**不许自己全库找**（否则比出 98 元维生素C，答非所问）
- **只让比价用候选集 = 用了一半前提**：所以每跳先查三种证据并喂进 prompt
- 规则拆解，不用 LLM：破坏「召回为空不调 LLM」红线，且不可复现

### 红灯抓出 4 个真实缺陷
| # | 现象 | 根因 |
|---|---|---|
| 1 | 多跳「假通过」 | 假替身 `find_products_by_need` 返回全部产品，第 1 跳集合本身就错 |
| 2 | 「鱼油有什么功能」判成产品名查询 | `有.{1,6}么` 的 `.` 匹配到「什」 |
| 3 | 「鱼油…？蛋白粉…？」被拼成「鱼油蛋白粉」 | 主题补全没判断「这句是否已有主语」 |
| 4 | **单测绿但接口编错法规** | 路由 `answer_one` 漏透传 `fact_hints`/`product_ids` |

第 4 条已做自证：删掉透传 → 接口测试准确 FAIL（2 failed）。

### 验收
```
1248 passed / 1 skipped（新增 78 条）
架构门禁 3 条全过
HTTP 真链路：两个问题都跑通，法规说对了，比价如实说无法比价
单步问题跳数=1、LLM 调用次数不变
```

★ 教训：**单测绿 ≠ 接口可用**（P4 已记一次，这是第二次）；
**误报的代价与漏报同级**（JD-02 把整段合规回答换成了拒答模板）。
