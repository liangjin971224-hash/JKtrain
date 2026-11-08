# Codex 任务提示词（复制给 Codex 执行）

你是数据标注执行者。请在仓库根目录 `D:\AI260803\健康培训移动` 下完成一次**微调数据生成**任务，只产出一个文件，不要改其他任何文件。

## 任务

读取 `train/annotation_wp_v2/targets.jsonl`（225 行，每行一个知识块），为**每个 chunk 生成恰好 2 个中文问题**，写入：

```
train/annotation_wp_v2/outputs/ft_questions.jsonl
```

共 **450 行**，每行一个 JSON 对象：

```json
{"chunk_id": "...", "collection": "...", "question": "...", "difficulty": "keyword|paraphrase|scenario"}
```

字段必须**恰好**是这四个，不要加注释、不要多余字段、不要 Markdown 代码块。UTF-8 无 BOM，`\n` 换行。

## 先读这两份

1. `train/annotation_wp_v2/ANNOTATION_BRIEF_FT.md`（完整口径）
2. `train/annotation_wp_v2/targets.jsonl`（输入）

每个 target 的关键字段：

- `text`：知识块正文，问题必须**仅凭它**就能回答；
- `existing_questions`：该块已被用于评测的问题（可能为空），**不得复述或改写**；
- `required_difficulties`：必须各生成一个的难度，如 `["keyword","paraphrase"]`；
- `bucket`：`single`（原本只有 1 个问法）或 `zero`（原本零覆盖）。

## 难度定义

- `keyword`：关键词式，像搜索框输入，短、直给，但必须是完整问句。
- `paraphrase`：转述式，换一种说法问同一个事实，尽量不出现原文连续措辞。
- `scenario`：门店情景式，员工问培训助教，或转述客户的问题。

## 硬性规则（违反会被校验器打回）

1. **每个 chunk 恰好 2 题**，难度与 `required_difficulties` 一一对应（顺序不限）。
2. 问题必须仅凭该 chunk 的 `text` 可独立回答；不得使用 chunk 之外的知识。
3. 问题与 `text`、以及 `existing_questions` 中任何问题，**不得有连续 ≥8 个相同汉字/字符**。
4. 不得复述或改写 `existing_questions`：它们已用于评测。对 `single` 块要换一个**不同的事实点或角度**。
5. 不问跨产品对比（一个 chunk 只讲一个产品/主题）；不编造 chunk 里没有的数字、成分、结论。
6. 问题长度 8~120 字；不要用「请告诉我」「帮我看看」这类无信息前缀，也不要只写一个词。
7. 450 行，不多不少；每个 `chunk_id` 出现 2 次。
8. 不要修改 `targets.jsonl`、`ANNOTATION_BRIEF_FT.md`、任何脚本或代码。

## 写法建议（提高一次通过率）

- 先读 `text`，在草稿里列出 2 个可独立提问的事实点（如：适用人群 / 用量 / 禁忌 / 流程步骤 / 数值 / 处理方式），每个事实点写 1 题。
- `keyword` 题尽量贴近正文里的关键术语，但**不要把整句抄进问题**；`paraphrase` 题换动词和句式；`scenario` 题加门店语境。
- 写完一题就自检：把问题里任意连续 8 个字拿到 `text` 里搜，若命中就改写。
- 对 `existing_questions` 非空的块，先看已有问题问的是哪个事实点，新问题**避开它**。
- 建议分批写入（例如每 50 题一批追加），最后核对行数 = 450。

## 交付后自检（必须跑）

在仓库根目录执行：

```bash
项目代码/.venv/Scripts/python.exe train/scripts/99_validate_ft_annotations.py --wp-dir train/annotation_wp_v2
```

要求：**硬违规 0，判定 PASS**。若有 HARD 行，按提示逐条修（通常是 8 字重叠、重复、难度不符、数量不对），改完重跑直到 PASS。
`WARN` 不阻塞，但「近重复 ≥0.85」和「与正文 6~7 字贴线」的条目请人工改写，避免训练数据冗余。

## 最终回报

输出以下信息，不要贴完整 450 行：

1. 产出文件路径与总行数；
2. 校验器输出（目标数 / 期望题数 / 实收数 / 硬违规数 / 判定）；
3. 若有 WARN，列出前 10 条；
4. 是否修改了其他文件（应为「否」）。
