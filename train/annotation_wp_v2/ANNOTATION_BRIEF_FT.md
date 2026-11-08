# 微调数据标注任务书（BGE-M3 对比学习，v2）

> 由 `98_ft_workpackage.py` 于 2026-11-07T20:59:08+08:00 生成。
> 执行者：Codex。产出写入本目录 `outputs/ft_questions.jsonl`，逐行合法 JSON，UTF-8。

## 背景

为 bge-m3 的对比学习准备「问题 ↔ 知识块」正例。输入 `targets.jsonl` 每行一个知识块：
`chunk_id / collection / title / section / text / existing_questions / required_difficulties / bucket`。

## 任务

对**每个 chunk 恰好生成 2 个新问题**，难度按 `required_difficulties`
各取一个（顺序不限），写入 `outputs/ft_questions.jsonl`：

    {"chunk_id": "...", "collection": "...", "question": "...", "difficulty": "keyword|paraphrase|scenario"}

难度定义：

- `keyword`   关键词式：像搜索框输入，短、直给，但必须是完整问句。
- `paraphrase` 转述式：换一种说法问同一个事实，尽量不出现原文的连续措辞。
- `scenario`  情景式：门店场景口吻（员工问培训助教，或转述客户的问题）。

## 硬性规则

1. 问题必须**仅凭该 chunk 的 `text` 就能独立回答**；`title/section` 只辅助理解主题。
2. 问题与 `text`、以及 `existing_questions` 里的任何问题，
   **都不得有连续 ≥8 字相同的片段**（99 号校验会打回）。
3. **不得复述或改写 `existing_questions`**：它们已用于评测，必须换角度/换事实点。
   若某块只有 1 个已有问题，新问题要问该块里**不同的事实点**。
4. 不编造 chunk 里没有的事实；不问跨产品对比（chunk 只讲一个产品/主题）。
5. 问题长度建议 8~80 字；不要用「请告诉我」「帮我看看」这类无信息前缀。
6. 每行一个 JSON 对象，字段名固定，不要加注释或多余字段。
7. 目标数 225 块 × 2 = **450 行**，不多不少。

## 产出

- `outputs/ft_questions.jsonl`（450 行）
- 不要修改 `targets.jsonl`；不要动其他目录。

校验命令（交付前自跑）：

    python train/scripts/99_validate_ft_annotations.py
