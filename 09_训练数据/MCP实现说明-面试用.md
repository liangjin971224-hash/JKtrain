# MCP 接入实现说明（面试用）

> 项目：健康顾问培训与辅助考核系统（`D:\项目\某健康公司培训系统\`）
> 实现日期：2026-11-06 ｜ MCP SDK 版本：**mcp 2.3.0**
> 代码：`interfaces/mcp/`（1331 行）｜ 测试：`tests/test_mcp/`（639 行，45 个用例）
> 全量测试：**1654 passed**

---

## 一、这个 MCP 是干什么的（30 秒回答）

**本项目的 31 个 HTTP 接口全部只服务前端页面，而项目没有前端。**
实际状态是：接口都做好了，只有开发者能在 Swagger 里 curl。

MCP 补的是「谁来调」这个入口：

```
学员在手机上打开 AI 助手 「孕妇能吃鱼油吗」
        ↓ AI 自己判断该调 kb_search
        ↓
    返回已过合规护栏的话术
```

**一句话**：不是「把接口包一层」，而是「在 MCP 这个新入口上，把原本由 JWT 承担的身份和权限重新建立一遍」。

| | HTTP 入口 | MCP 入口 |
|---|---|---|
| 谁来调 | 前端页面 | **AI 助手**（WorkBuddy / Claude Desktop / 企业 IM） |
| 谁决定调什么 | 人点按钮 | **LLM 自己判断** |
| 身份从哪来 | 每个请求的 JWT header | **连接建立时注入，工具签名里没有** |

---

## 二、代码在哪（文件清单）

### 2.1 生产代码 `interfaces/mcp/`

| 文件 | 行数 | 职责 |
|---|---|---|
| **`server.py`** | 347 | **入口**。创建 `MCPServer`，注册 5 个能力，统一日志与错误 |
| `identity.py` | 160 | **会话身份**。建连时解析 token → `McpIdentity` |
| `policy.py` | 128 | **工具级授权**。白名单，未登记=拒绝 |
| `logging_.py` | 96 | 调用日志 `data/mcp_tool_calls.jsonl` |
| `resources.py` | 86 | `product://{id}` 产品档案 |
| `prompts.py` | 88 | `expression_polish` 润色模板 |
| `tools/kb.py` | 115 | `kb_search` 知识检索 |
| `tools/graph.py` | 172 | `graph_entity` 图谱实体 |
| `tools/drill.py` | 139 | `my_drill_reviews` 演练评语 |

### 2.2 测试 `tests/test_mcp/`

| 文件 | 行数 | 覆盖 |
|---|---|---|
| `test_identity.py` | 292 | **T1–T5 越权测试**（核心） |
| `test_tools.py` | 252 | 工具返回结构、参数夹取、红线 |
| `test_logging.py` | 95 | 日志可审计性 |

### 2.3 定位：第二个入站适配器

```
用户 / WorkBuddy / 企业 IM
        │
        ▼
interfaces/mcp/          ← 新增（入站适配器 ②）
interfaces/http/         ← 已有（入站适配器 ①）
        │
        ▼
application/use_cases/   ← 完全复用，零改动
domain/ports/            ← 完全复用，零改动
infrastructure/          ← 完全复用，零改动
```

> **架构门禁 15 条全部通过**，尤其 `test_interfaces_does_not_import_infrastructure`：
> MCP 层只依赖 `config.assembly.Container` 和 `domain.ports`，不 import 任何 adapter。

---

## 三、五个能力怎么实现的

### 3.1 为什么是 5 个，不是 31 个

**工具越多，LLM 选中率越低。** 5~8 个时选中率好，超过 15 个明显下降。

所以按「**用户意图**」而不是「HTTP 端点」来设计：

| 做法 | 效果 |
|---|---|
| 裁剪 | 31 端点 → 5 能力 |
| 合并 | `drill/start` + `answer` + `review` + 3 张表 → 1 个 `my_drill_reviews` |
| **故意不暴露 6 类** | 见 3.7 |

### 3.2 `kb_search` —— 知识检索

**复用**：`AnswerQuestionUseCase`（不是 `RetrieveUseCase`）

| 为什么不用 RetrieveUseCase | 说明 |
|---|---|
| 它只返回「检索到什么」 | MCP 的调用方是 AI，它要的是**能直接说出口的话** |
| 后者返回的已过 5 道处理 | 三路 RRF → 意图权重 → 权限过滤 → 护栏 → 输出清洗 |

**关键实现**：

```python
result = uc.execute(
    query,
    top_k=k,
    vector_collection=getattr(settings, "coll_training", ""),   # ★ 漏了就静默降级
    allowed_permissions=identity.retrieval_levels,              # ★ 权限下沉
)
```

**返回结构**：

```json
{
  "answer": "康源牌鱼油软胶囊批准的功能是「有助于维持血脂…」",
  "refused": false,
  "cited_chunk_ids": ["..."],
  "intent": "产品功效",
  "retrieval_levels": ["public", "internal"],
  "elapsed_ms": 8930.5
}
```

### 3.3 `graph_entity` —— 图谱实体

**为什么单独做一个图谱工具**：图谱能给出**排除法**，向量给不出。

```
问：血脂高的人能吃什么？
  向量：返回「血脂」相关段落，可能包含旧内容
  图谱：明确查出批准了 EFF_19 的产品 + 各自排除哪些人群
```

后者是**集合运算**，前者是**文本相似**。「谁能吃谁不能吃」必须靠集合运算。

**只暴露 4 个关系**（`GraphQueryPort` 有 10 个方法）：

| 关系 | 用途 |
|---|---|
| `ALIAS_OF` | 别名归一（「鱼油」→ P001） |
| `CONTRAINDICATED_FOR` | **谁能吃 —— 最核心** |
| `CLAIMS` | 批准功能原文 |
| `SUITABLE_FOR` | 适合谁 |

**关键：禁忌带严重度和原因**

```json
{
  "name": "出血倾向者和出血性疾病患者",
  "severity": "critical",
  "reason": "鱼油可能影响凝血功能，与抗凝药同服增加出血风险",
  "must_block": true
}
```

`reason` 是给 AI 解释「为什么不能吃」用的，不能空。

### 3.4 `my_drill_reviews` —— 演练评语

**★ 签名里没有 `employee_id`，这是本方案的核心设计。**

```python
def my_drill_reviews(container, identity, *, days: int = 7) -> str:
    #                           ↑ 身份来自会话，不是参数
    trainee_id = identity.employee_no
```

**如果写成 `my_drill_reviews(employee_id)` 会怎样**：
Agent 传别人的工号就能读到别人的评语，而 `drill_review` 表有 **183 行真实评语**，包含「客户自述在吃阿司匹林」这类内部记录。

**遵守红线**：不给分数、不给评级

```python
# 返回只有：drill_id / case_type / element_hit / risk_count
# 没有 score / verdict / grade
```

### 3.5 `product://{product_id}` —— 产品档案（Resource）

**Resource 与 Tool 的区别**（面试常问）：

| | Tool | Resource |
|---|---|---|
| 谁触发 | AI **主动调用** | AI **按 URI 读取** |
| 有参数 | 有 | URI 模板参数 |
| 副作用 | 可能 | **纯读** |
| 用途 | 执行操作 | 挂进上下文 |

选 Resource 因为：纯读、可被客户端并列挂载、多个产品可同时进上下文。

### 3.6 `expression_polish` —— 润色模板（Prompt）

**为什么提示也要做成 MCP 能力**：项目有「组织表达」红线，原本写在文档里靠人记。做成 Prompt 后**规则固化在模板字符串里**。

三条约束的来历（都不是凭空写的）：

| 约束 | 反面案例 |
|---|---|
| 不改数字 | 「命中率 4/6」被改成「大部分命中」，无法审计 |
| 不加事实 | 「客户自述血脂偏高」→「客户血脂偏高」，丢掉「自述」 |
| 不加强语气 | 「建议咨询医生」→「建议立即就医」，责任边界变了 |

### 3.7 故意不暴露的 6 类（这部分是面试加分项）

| 不暴露 | 理由 |
|---|---|
| `POST /auth/login` | MCP 不管登录，否则 Agent 可自换身份 |
| `POST /accounts/role` | 改权限是高危，Agent 不可碰 |
| `POST /employees/import` | 批量导入，误调即脏数据 |
| `GET /audit-logs` | 审计日志给 Agent 看 = 开后门 |
| **`POST /drill/answer`** | 演练是状态机 + 台词 100% 来自脚本，Agent 介入破坏可复现性 |
| `GET /papers/{no}/export` | 文件导出，超出问答助手范围 |

---

## 四、身份与权限（技术核心）

### 4.1 为什么这是真问题

**MCP 的工具调用没有「当前用户」概念。** REST 有 JWT header，路由里天然知道谁在调；MCP 是「一次建连，多次调用」，身份只在建连时存在。

### 4.2 身份注入时序

```
① 客户端建连：stdio 用环境变量 MCP_TOKEN，SSE 用 Authorization: Bearer
② Server 解析：校验签名 → **回库取最新角色** → McpIdentity(employee_no, role)
③ 会话对象持有 Identity，生命周期 = 连接生命周期
④ 每次工具调用：从会话取 Identity，**不读参数**
⑤ 工具内部：把 retrieval_levels 传给用例 → 拼进 Milvus filter
```

**为什么回库取角色而不是用 token 快照**：

| 场景 | 用快照的后果 |
|---|---|
| 学员升主管 | 要等 8 小时 token 过期才生效 |
| **主管降学员** | **这 8 小时内他仍是主管，能继续组卷** |

后者是真实安全问题（离职/降职当天仍有权限）。所以每请求回库查一次（FR-048 即时生效）。

### 4.3 三层权限分别在哪落地

| 层 | 需求 | MCP 侧实现 |
|---|---|---|
| 功能权限 | FR-048 | `policy.TOOL_FEATURES` 白名单，**未登记=拒绝** |
| 数据权限 | FR-049 | 会话绑定，**工具签名里没有 employee_id** |
| 检索权限 | FR-050 | `Role.retrieval_levels` → `allowed_permissions` → **Milvus filter** |

### 4.4 T1–T5 越权测试（全部实测通过）

| # | 用例 | 实测结果 |
|---|---|---|
| **T1** | 学员塞 `employee_id=M001` | ✅ **参数被忽略**，仍返回 E001 的数据 |
| **T2** | 学员 token 查知识 | ✅ 级别 `['public','internal']`，**无 restricted/confidential** |
| **T3** | 学员调 `build_paper` | ✅ `Unknown tool`（未登记工具根本不存在） |
| **T4** | 无 token（生产模式） | ✅ `McpAuthError: 未提供身份令牌` |
| **T5** | 角色降级后立即调用 | ✅ 每次回库，降级立即生效 |

**T1 的真实输出**：

```
未传 employee_id  → trainee_id=E001 条数=2
塞 employee_id=M001 → trainee_id=E001   ✅ 参数被忽略
```

---

## 五、外部怎么调用

### 5.1 启动方式一：stdio（本地 / 桌面版 AI 助手）

```bash
cd D:\项目\某健康公司培训系统\项目代码

# 生产模式（需要 token）
set AUTH_REQUIRED=true
set MCP_TOKEN=<登录拿到的 Bearer 令牌>
python -m interfaces.mcp.server

# 开发模式（不需要 token，按 MCP_DEBUG_ROLE 处理）
set AUTH_REQUIRED=false
python -m interfaces.mcp.server
```

### 5.2 WorkBuddy / Claude Desktop 配置

在 MCP 配置文件里加（**注意 Windows 要用 `python.exe` 绝对路径**）：

```json
{
  "mcpServers": {
    "health-training": {
      "command": "D:\\项目\\某健康公司培训系统\\项目代码\\.venv\\Scripts\\python.exe",
      "args": ["-m", "interfaces.mcp.server"],
      "cwd": "D:\\项目\\某健康公司培训系统\\项目代码",
      "env": {
        "AUTH_REQUIRED": "true",
        "MCP_TOKEN": "<你的 Bearer 令牌>",
        "PYTHONPATH": "D:\\项目\\某健康公司培训系统\\项目代码"
      }
    }
  }
}
```

> **Claude Desktop 的配置文件位置**：
> `%APPDATA%\Claude\claude_desktop_config.json`
> **WorkBuddy**：设置 → MCP → 添加自定义服务器

### 5.3 客户端能看到什么（实测握手结果）

```
server: health-training-mcp v0.1.0
capabilities: tools=True, resources=True, prompts=True

📋 工具 3 个：
   - kb_search(query, top_k)
   - graph_entity(name, with_relations)
   - my_drill_reviews(days)

📦 资源模板 1 个：['product://{product_id}']
💬 提示 1 个：['expression_polish']
```

**注意 `my_drill_reviews` 的参数只有 `days`** —— 客户端根本没有 `employee_id` 可填。

### 5.4 实际调用示例（Python 客户端，实测可跑）

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(
        command=r"D:\项目\某健康公司培训系统\项目代码\.venv\Scripts\python.exe",
        args=["-m", "interfaces.mcp.server"],
        cwd=r"D:\项目\某健康公司培训系统\项目代码",
        env={"AUTH_REQUIRED": "true", "MCP_TOKEN": "<token>",
             "PYTHONPATH": r"D:\项目\某健康公司培训系统\项目代码"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 查知识
            r = await session.call_tool("kb_search", {
                "query": "康源牌鱼油软胶囊批准的功能是什么？", "top_k": 5})
            print(r.content[0].text)

            # 查图谱（禁忌）
            r = await session.call_tool("graph_entity", {"name": "鱼油"})
            print(r.content[0].text)

            # 查自己的评语（没有 employee_id）
            r = await session.call_tool("my_drill_reviews", {"days": 30})
            print(r.content[0].text)

            # 读资源
            r = await session.read_resource("product://P001")
            print(r.contents[0].text)

            # 拿提示模板
            p = await session.get_prompt("expression_polish",
                {"text": "客户命中率 4/6", "scene": "monthly_report"})
            print(p.messages[0].content.text)

asyncio.run(main())
```

### 5.5 自然语言调用示例（用户在 AI 助手里说的话）

| 用户说 | AI 会调 | 返回 |
|---|---|---|
| 「孕妇能吃鱼油吗」 | `kb_search` | 带禁忌说明的合规话术 |
| 「这款产品有什么功能」 | `graph_entity` | 批准功能原文 + 禁忌 |
| 「我上周练得怎么样」 | `my_drill_reviews` | 自己的演练记录（**查不到别人**） |
| 「帮我把这段月报润色下」 | `expression_polish` | 带三条硬约束的模板 |

---

## 六、踩过的坑（4 个，全部实测）

### 6.1 坑 1：`FastMCP` 在 2.x 改名了

```python
# ❌ 网上教程（mcp 1.x）都是这么写的
from mcp.server.fastmcp import FastMCP
server = FastMCP("name")

# ✅ 2.x 实际
from mcp.server.mcpserver import MCPServer
server = MCPServer(name="name", version="0.1.0")
```

官方专门留了个 `fastmcp.py` 只为了在 import 时给提示：

```
No module named 'mcp.server.fastmcp'. This is mcp 2.x, where FastMCP was renamed
to MCPServer (from mcp.server.mcpserver import MCPServer) ... or pin 'mcp<2'
```

### 6.2 坑 2：`inputSchema` → `input_schema`

**协议层（JSON）用 camelCase，Python 对象用 snake_case** —— 两边不一致。

```python
t.inputSchema        # ❌ AttributeError
t.input_schema       # ✅
init.serverInfo      # ❌
init.server_info     # ✅
```

### 6.3 坑 3：Windows 必须装 `pywin32`

`mcp/os/win32/utilities.py:22` 直接 `import pywintypes`，不装就 `ModuleNotFoundError`。

### 6.4 坑 4：模板资源不在 `list_resources()`

| 方法 | 返回 |
|---|---|
| `list_resources()` | 只有**静态**资源 |
| `list_resource_templates()` | **模板**资源（`product://{id}`） |

按「资源数 1」断言会失败。

### 6.5 坑 5（本项目特有）：漏传 `vector_collection` 导致静默降级

**这是最值得讲的一个**，因为它暴露了本项目的老问题。

首次实测日志：

```
召回路 vector 失败，降级继续: ValueError: 未指定 vector_collection
三路召回完成: graph=1, vector=0, keyword=5, fused=5, degraded=['vector']
```

**不报错、不抛异常，只是召回从三路变两路，融合块数从 9 降到 5。**

修复后：

```
三路召回完成: graph=1, vector=5, keyword=5, fused=9, degraded=无
```

**这是「新入口漏透传可选参数」的第 5 次复发**（前 4 次：`fact_hints` / `product_ids` / `history` / 流式 `done.meta`）。

所以加了源码扫描测试守住：

```python
def test_vector_collection_forwarded():
    src = inspect.getsource(kb_mod.kb_search)
    assert "vector_collection=" in src, "漏传会静默降级"
```

> **面试可讲**：同一类 bug 复发 5 次，说明不是个人疏忽，而是**架构缺少约束**。所以第 5 次不再靠「记住」，而是用测试锁住。

---

## 七、验证记录（全部实测，可复现）

| 验证项 | 结果 | 脚本 |
|---|---|---|
| 架构门禁 15 条 | ✅ 全过 | `tests/test_architecture.py` |
| MCP 单元测试 45 个 | ✅ 全过 | `tests/test_mcp/` |
| 全量测试 | ✅ **1654 passed** | `pytest tests/` |
| 函数层五能力 | ✅ 全部返回真实数据 | `09_训练数据/verify_mcp_tools.py` |
| **stdio 协议端到端** | ✅ 握手 + 5 能力 + 越权拦截 | `09_训练数据/verify_mcp_stdio.py` |
| **真实 token 越权 T1/T2/T3** | ✅ 参数被忽略 / 级别收敛 / 工具不存在 | `09_训练数据/verify_mcp_auth.py` |

**真实数据规模**（不是假数据）：

| 库 | 数据 |
|---|---|
| MySQL `jiankangshujuku` | 13 表，10 表有数据（drill_review **183** 行） |
| Neo4j `jiankangshujuku` | 156 节点 / 184 关系 |
| Milvus | 4 collection / 580 块 |

---

## 八、面试问答准备

### Q1：为什么不用普通 REST API，要做 MCP？

**A**：REST 是给人看的（Swagger 文档里写「POST /drill/review，body 含 drill_id」），AI 看了不知道**什么时候该调**。MCP 的 description 写的是「当用户说『我上周练得怎么样』时使用」——**这是给 AI 的使用说明，不是接口文档**。

### Q2：MCP 和「把接口包一层」有什么区别？

**A**：包一层是 10 分钟的事（自动转 31 个工具）。MCP 要解决的是三件事：① 工具裁剪（31→5，否则 LLM 选不准）② 身份注入（工具签名里不能有 employee_id）③ 权限下沉（检索阶段生效，不是查完再筛）。后两件才是 2 天工时的来源。

### Q3：MCP 没有「当前用户」概念，怎么办？

**A**：身份在**连接建立时**注入，存会话对象，工具调用时从会话取。所以 `my_drill_reviews` 的签名里根本没有 `employee_id` —— 客户端想传都没有这个字段。实测传 `employee_id=M001` 会被忽略，仍返回自己的数据。

### Q4：为什么不把 31 个接口都暴露？

**A**：工具越多 LLM 选中率越低。而且有 6 类**故意不暴露**：改权限、批量导入、审计日志、演练答题、导出文件、登录。其中「演练答题」的理由最值得说——演练是状态机 + 台词 100% 来自脚本，Agent 介入会破坏可复现性。

### Q5：怎么保证不破架构？

**A**：项目有 15 条架构门禁，其中 `test_interfaces_does_not_import_infrastructure` 是硬约束。MCP 层只依赖 `config.assembly.Container` 和 `domain.ports`，**不 import 任何 adapter**。改完跑门禁全绿。

### Q6：踩过什么坑？

**A**：讲**坑 5**（vector_collection 静默降级）最有价值。它不报错、不崩溃，只是召回从三路变两路、融合块数从 9 降到 5。而且这是本项目同类 bug 的第 5 次复发，最后用**源码扫描测试**锁住——说明我意识到「靠记性」不管用，得靠约束。

### Q7：这个方案最大的价值是什么？

**A**：不是「技术多新」，而是**「让一个没有前端的系统变得能用」**。31 个接口原本只有开发者能 curl，加了 MCP 后学员在手机上问、主管在微信里查都能用，而且**跨出去之后身份和权限没有降级**。

---

## 九、如果继续做（S3）

| 项 | 说明 | 成本 |
|---|---|---|
| SSE/HTTP 传输 | mount 到 `app.py`，可鉴权可共享（stdio 只能本地） | 0.5 天 |
| `kb_quality_gaps()` | 读 `miss_log`（25 行真实未命中），告诉培训师知识盲区 | 0.3 天 |
| `case_detail(case_id)` | 案例 + 判定点 | 0.2 天 |
| OpenAI 兼容适配器加清洗 | **既有缺口**：云端千问输出带 `**`，Ollama 路径已清洗 | 0.2 天 |

---

## 十、复现命令

```bash
cd D:\项目\某健康公司培训系统\项目代码

# 装依赖（锁版本，2.x 内部仍在变）
.venv\Scripts\python.exe -m pip install "mcp==2.3.0" "mcp-types==2.3.0" pywin32 sse-starlette

# 测试
.venv\Scripts\python.exe -m pytest tests/test_mcp/ -v
.venv\Scripts\python.exe -m pytest tests/test_architecture.py -v
.venv\Scripts\python.exe -m pytest tests/ -q          # 全量 1654 passed

# 启动
set AUTH_REQUIRED=false
.venv\Scripts\python.exe -m interfaces.mcp.server

# 验证脚本
python 09_训练数据\verify_mcp_stdio.py     # 协议端到端
python 09_训练数据\verify_mcp_auth.py      # 真实 token 越权
```

---

_本文档基于 2026-11-06 的实机验证，所有数字与输出均为实测，未做估算。_
