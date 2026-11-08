# P4 阶段验收记录（第二部分：LLM 主备双通道）

> 阶段：**P4 生成与护栏**
> 日期：2026-10-22
> 代码根目录：`D:\项目\某健康公司培训系统\项目代码\`

---

## 1. 交付清单

| # | 交付物 | 验收标准 | 状态 |
|---|---|---|---|
| 1 | `OpenAICompatibleAdapter` | 千问 / DeepSeek 通用 | ✅ 3 个方法 |
| 2 | `FallbackLanguageModel` | 主备切换 | ✅ 主千问 → 备 DeepSeek |
| 3 | 错误分类 | 区分该切/不该切 | ✅ 5 类 |
| 4 | 可观测性 | 切换留痕 | ✅ `last_used` / `last_error` / `stats()` |
| 5 | 真实调用验证 | 两家都能返回 | ✅ 16 条集成测试 |
| 6 | **真实故障切换** | 401 能触发切换 | ✅ 实测通过 |

**代码量**：两个适配器约 400 行，测试 35 条（19 单元 + 16 集成）。

---

## 2. 配置（已写入 `.env`）

```bash
# 主：千问（阿里云百炼 DashScope，OpenAI 兼容模式）
LLM_PRIMARY=qwen
LLM_FALLBACK_ENABLED=true
LLM_PRIMARY_RETRIES=1
LLM_TIMEOUT_SEC=30

QWEN_API_KEY=sk-ws-...（已配置）
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus

# 备：DeepSeek
DEEPSEEK_API_KEY=sk-4c2c...（已配置）
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

**两个 Key 都已实测可用**（真实调用返回正常）。

### 2.1 切换主备只需改一个值

```bash
LLM_PRIMARY=deepseek    # 改这一行即可，代码零改动
```

`Settings.llm_endpoints()` 按 `llm_primary` 生成有序端点列表，
`FallbackLanguageModel` 只按顺序遍历，不关心谁主谁备。

---

## 3. ★ 什么情况切、什么情况不切（最关键的设计）

双重保险最容易做错的地方**不是「能不能切」，是「什么时候该切」**：

| 情况 | 是否切换 | 理由 |
|---|---|---|
| 超时 | ✅ 切 | 端点级故障，换端点大概率能好 |
| 限流（429） | ✅ 切 | 该端点配额用尽 |
| 服务端错误（5xx） | ✅ 切 | 厂商侧故障 |
| **鉴权失败（401/403）** | ✅ 切 | **Key 失效/欠费，必须换** |
| 返回体格式无法解析 | ✅ 切 | 该端点协议异常 |
| **重试成功后** | ❌ **不切** | 避免不必要的模型切换导致同会话回答风格突变 |
| 全部端点失败 | ❌ 不切，**抛异常** | 见 §3.2 |

### 3.1 「不该切」的判定同样重要

重试成功就不切。理由：
一次问答里前后两个环节用了不同模型，回答风格会不一致
（千问偏「顾问口吻」，DeepSeek 偏「说明书口吻」），
而本项目要求话术统一。所以优先重试同一端点，实在不行才换。

### 3.2 全部失败必须抛异常，不返回空

```python
# ❌ 错误：返回空串
return GenerationResult(text="")

# ✅ 正确：抛异常并列出所有端点的错误
raise LLMCallError(
    f"所有 LLM 端点均失败（2 个）。qwen: HTTP 401...；deepseek: 超时..."
)
```

**为什么**：空串会被上层当成「模型说没内容」，掩盖真实故障。
而本项目最怕的就是「看起来正常但实际没依据」的回答——
护栏能拦住违规措辞，但拦不住「因为模型挂了所以什么都没说」。

---

## 4. 真实故障切换实测

**这是本阶段最有力的一条证据：不是用假替身模拟，而是真的把千问 Key 改坏。**

### 4.1 实测过程

```python
broken_qwen = OpenAICompatibleAdapter(
    name="qwen(坏Key)", api_key="sk-invalid-key-test", ...)
real_ds = OpenAICompatibleAdapter(name="deepseek", ...)
llm = FallbackLanguageModel(primary=broken_qwen, backups=[real_ds],
                            primary_retries=0)
llm.generate("用一句话说明什么是保健食品。")
```

### 4.2 实测输出

```
LLM 端点失败: endpoint=qwen(坏Key), action=generate, attempt=1/1,
  error=HTTP 401 Unauthorized | {"error":{"message":"Incorrect API key provided...",
  "code":"invalid_api_key"},"request_id":"6af2c437-..."}

LLM 已降级到 deepseek，前序端点失败原因: qwen(坏Key): HTTP 401 Unauthorized | ...

返回 model = deepseek-flash
text = 保健食品是声称具有特定保健功能或以补充维生素、矿物质为目的的食品，
       适宜特定人群食用，具有调节机体功能的作用，但不以治疗疾病为目的。

last_used = deepseek
last_error = qwen(坏Key): HTTP 401 Unauthorized | {"error":...
stats = {'qwen(坏Key)': {'success': 0, 'failed': 1},
         'deepseek': {'success': 1, 'failed': 0}}
```

**切换链路完整生效**：401 → 分类 → 切换 → 成功返回 → 留痕 → 统计。

### 4.3 一个发现：DeepSeek 回传的 model 名与请求不同

| 项 | 值 |
|---|---|
| 请求 `model` | `deepseek-chat` |
| 响应 `model` | `deepseek-flash` |

**这不是 bug**——DeepSeek 服务端会把 `deepseek-chat` 路由到当前的实际模型并回传真实名。
所以 `GenerationResult.model` 取的是**响应里的值**（实际服务的模型），不是请求值，
这对成本核算与排障更有意义。

---

## 5. 可观测性设计

切换是**静默降级**——如果没留痕，出故障时根本不知道切过。

| 字段 | 用途 |
|---|---|
| `last_used` | 最后一次成功的端点（问「这次答案是谁给的」） |
| `last_error` | 最后一次切换的原因（问「为什么切了」） |
| `stats()` | 每个端点成功/失败次数（问「主模型健康吗」） |

### 5.1 一个设计取舍：`stats()` 要包含未调用的端点

初版 `stats()` 只返回调用过的端点，导致：

```python
stats["deepseek"]["success"]   # KeyError
```

**问题**：监控代码会崩。而且「备用从未被用过」本身是**重要信号**
（说明主模型一直健康），缺了反而看不出。

修正：遍历所有配置的端点，未调用的返回全 0。

### 5.2 一个设计取舍：切换成功时也要记 `last_error`

初版只在**全部失败**时记 `last_error`。但最常见的场景恰恰是
**「切了并且成功了」**——这时如果不记，运维看到的就是「一切正常」，
完全不知道千问挂过。

修正：只要有端点失败就记，不论最终是否成功。

---

## 6. 为什么不用 `openai` SDK

| 维度 | 用 SDK | 用标准库 urllib（当前） |
|---|---|---|
| 依赖 | 多一个包，版本兼容问题 | 零额外依赖 |
| 错误分类 | SDK 异常层级随版本变 | 直接判 HTTP 状态码，稳定 |
| 调试 | 多一层封装 | 请求/响应全可见 |
| 并发 | 有连接池 | 无（本项目调用量小） |

**本项目场景**：培训系统，问答并发低（几十人在线），不需要连接池与异步。
用 urllib 换来的是**错误分类的可控性**——这恰恰是主备切换的判据。

---

## 7. 测试结果

| 项 | 结果 |
|---|---|
| 全量测试（跳过集成） | **482 passed / 1 skipped / 11.8s** |
| 集成测试（含真实 API） | **79 passed / 33.9s** |
| LLM 切换测试（假替身） | **19 条 / 0.1s** |
| LLM 真实调用测试 | **16 条 / 14.4s** |
| CI 门禁 | 3 条全过 |
| `main.py` | exit=0，容器端口 **6/7** |

### 7.1 新增测试覆盖

**`test_llm_fallback.py`（19 条，假替身）**

| 测试类 | 覆盖 |
|---|---|
| `TestPrimaryWorks` | 主模型正常时不调备用 |
| `TestWhenToFailover` | 5 类异常都切换、重试次数、重试成功不切 |
| `TestBackupChain` | 多备用按序、无备用直接抛 |
| `TestObservability` | `last_used` / `last_error` / `stats` |
| `TestDoNotSwallow` | 全部失败时保留每个端点的原始错误 |

**`test_llm_live.py`（16 条，真实 API）**

| 测试类 | 覆盖 |
|---|---|
| `TestProtocol` | 返回非空、token 用量、`system_prompt` 生效、`temperature=0` 确定性、分类合法 |
| `TestLiveFailover` | **真实 401 触发切换**、全坏时抛异常 |
| `TestAssemblyLive` | 容器里的 LLM 真的是主备结构且能调通 |

### 7.2 关键测试：安全意图分类正确

```python
def test_classify_security_intent_correct():
    """「我爸有高血压在吃降压药，能吃这个鱼油吗」应分到「禁忌咨询」。"""
    label = llm.classify(..., ["禁忌咨询", "产品功效", "业务流程", "开放问答"])
    assert label == "禁忌咨询"
```

**为什么重要**：P3 的权重表按意图查。若安全类查询被分到「产品功效」，
图谱权重会从 0.6 降到 0.5，禁忌召回变弱。这条测试护住了 P3 的入口。

---

## 8. 遗留问题

| # | 问题 | 影响 | 计划 |
|---|---|---|---|
| 1 | 未实现 `AnswerQuestionUseCase` | 问答链路未打通 | 下一步 |
| 2 | 未做「分区组装」 | 事实卡/文档/规范/经验未分层 | 下一步 |
| 3 | 意图分类只有 7 类候选 | 未定完整标签体系 | 下一步需对齐 D005 分类 |
| 4 | 未测超时（真实） | 慢网络下的切换行为未验 | 可用本地代理模拟 |
| 5 | `is_available` 只查 Key 非空 | 不做真实探测（省费用） | 可接受，自检已覆盖 |
| 6 | 无 API 成本统计 | token 已有但未聚合 | P7 日报可加 |

### 8.1 未验证的部分（明确声明）

| 未验证项 | 原因 |
|---|---|
| 千问与 DeepSeek 回答质量差异 | 需要业务方做盲评 |
| 高频并发下的限流行为 | 培训场景并发低，未压测 |
| 长时间运行的 Key 过期处理 | 需长时间观察 |

---

## 9. 结论

**P4 第二阶段（LLM 主备双通道）完成。**

| 验收项 | 结果 |
|---|---|
| 千问 / DeepSeek 通用适配 | ✅ 实测两家都通 |
| 主备切换 | ✅ **真实 401 验证通过** |
| 错误分类 | ✅ 5 类该切 + 1 类不该切 |
| 可观测性 | ✅ 留痕 + 统计 |
| 自检报告主备 | ✅ `主=qwen(qwen-plus) → 备=deepseek` |
| 测试 | ✅ 482 日常 + 79 集成 |

**过程产出**：修正 2 个设计缺陷（`stats()` 缺未调用端点、切换成功不记 `last_error`），
两者都属于「平时看不出、出故障时最需要」的可观测性问题。

**下一步**：实现 `AnswerQuestionUseCase` + `QaAgent`（6 节点 Workflow），
重点是「召回为空 → 拒答，不调 LLM」的红灯测试。
