# 标注任务书（检索评测集 v1）

> 由 82_build_codex_workpackage.py 于 2026-11-06T23:50:16+08:00 生成。
> 执行者：Codex。产出写入本目录 `outputs/`，逐行合法 JSON，UTF-8，无多余字段。

## 背景

为「健康顾问培训系统」构建检索级评测集：给定问题，判断正确知识块是否被检回。
你负责两类标注，对应两个输出文件。

## 任务 1：从知识块生成问题（策略 B）

输入：`gen_b_targets.jsonl`（每行一个 chunk：chunk_id / collection / title / section / text）。

对**每个 chunk 恰好生成 2 个问题**，写入 `outputs/gen_questions.jsonl`：

    {"chunk_id": "...", "collection": "...", "question": "...", "difficulty": "..."}

难度取值（每个 chunk 的两问各取其一，`required_difficulties` 字段指定了要哪两档）：

- `keyword`   关键词式：像搜索框输入，短、直给（如「鱼油 成分」风格但要是完整问句）
- `paraphrase` 转述式：换一种说法问同一个事实，不出现原文措辞
- `scenario`  情景式：门店场景口吻，参考「客户问这款能不能改善视力，我该怎么答？」

**硬性规则**：

1. 问题必须**仅凭该 chunk 的 text 就能独立回答**（title/section 只辅助理解主题）。
2. **禁止问题与原文存在连续 ≥8 字相同的片段**（校验脚本 83 号会打回）。
3. 情景式问题口吻遵循培训语境：员工向培训助教提问，或转述客户的问题。
4. 不编造 chunk 里没有的事实；不问跨产品对比（chunk 只讲一个产品/主题）。
5. 每行一个 JSON 对象，字段名固定，不要加注释或多余字段。

## 任务 2：候选相关性判定（策略 A）

输入：`judge_a_queries.jsonl`（qid / question）+ `candidates/<qid>.jsonl`
（每行 candidate_id / chunk_id / title / section / text，顺序已随机打乱）。

对每个 qid 的**每条候选**判一次，写入 `outputs/judge_results.jsonl`：

    {"qid": "...", "candidate_id": "...", "verdict": "..."}

verdict 取值与判定口径：

- `relevant`    候选含回答该问题所需的核心事实（去掉它答案就立不住）→ 计入 gold
- `partial`     只沾边（同主题但不含核心事实）→ 不计入 gold
- `irrelevant`  与问题无关 → 不计入 gold

**判定纪律**：

1. 只依据候选文本内容判断，不臆测「库里可能有更好的块」。
2. 候选顺序已打乱，与检索排名无关，不要猜测排序意图。
3. 一个 qid 通常有 0~3 个 relevant；如果超过 5 个，重新审视口径是否放太宽。

## 完成定义

- `outputs/gen_questions.jsonl` 行数 = gen_b_targets.jsonl 行数 × 2
- `outputs/judge_results.jsonl` 行数 = Σ 每个qid的候选数
- 两个文件都能通过 `python ../../scripts/83_validate_annotations.py` 校验
## 已知范围裁剪（不必疑问，均有人工评审兜底）

- `question_bank` 的 140 条由脚本自动生成「题干原句」查询，**不在你的任务里**。
- `self_test` 的 30 条不进评测集（全 restricted，问答链路检索不到）。
- heldout2 的 12 条纯人工金标准不参与任何标注。
