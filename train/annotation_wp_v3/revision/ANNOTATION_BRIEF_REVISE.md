# 案例 rubric 修订任务书（A1b）

> 由 `103_build_rubric_revision.py` 于 2026-11-08T13:40:43+08:00 生成。
> 执行者：Codex。产出写入本目录 `outputs/answer_rubrics_revised.jsonl`。

## 背景

当前 31 条案例 rubric 在 100 题答案评测里出现了**判词形误报**：
回答语义正确，但因为没有逐字命中 `must_include` 而判失败。例子：

- 回答写「孕妇、乳母」，rubric 只认「孕妇和乳母」；
- 回答写「致命错误」，rubric 只认「一票否决」；
- 回答写「免疫低下的成人」，rubric 只认「免疫力低下者」。

本任务不是放宽标准，而是把**同义/格式变体**收进 `any_of`，让评分看语义而不是看字面。

## 输入

- `targets.jsonl`：31 题 + gold 正文
- `current_rubrics.jsonl`：当前 rubric
- `failures.jsonl`：100 题基线里案例题的失败证据（question / answer / missed / forbidden）

## 修订规则

1. 对每个 `must_include` 短语：
   - 若它**有合法的同义/格式变体**（顺序、单复数、简称、同义词），把它从
     `must_include` **移出**，改写成 `any_of` 里的一个组（2~5 个变体）；
   - 若它是**不可替代的硬事实**（编号、数值、专有名词），保留在 `must_include`。
2. 修订后：`must_include` **1~5 条**，`any_of` 每组 2~5 条，
   `must_include + any_of 组数` 合计 **2~5 个必答点**。
3. `any_of` 组内每个变体都必须语义等价、且至少一个能在 gold 正文里找到。
4. `must_not_include` 保持 0~3 条；**仍不得**是正确否定句的子串
   （「不可以X」→ 不能写「可以X」）。
5. 不得新增 gold 里没有的事实；不得为了通过而删除核心必答点。
6. 输出 31 行，qid 与 targets 一一对应。

## 参考失败证据

`failures.jsonl` 里每条包含 `missed`（未命中的 must_include）与 `answer`（模型实际回答）。
优先处理 `missed` 里属于同义/格式变体的项；如果 `answer` 确实漏了核心事实，
**不要**把它加进 any_of，保留严格项。

## 交付后自检

```bash
项目代码/.venv/Scripts/python.exe train/scripts/104_validate_revised_rubrics.py --wp-dir train/annotation_wp_v3
```

要求硬违规 0、判定 PASS；有 HARD 按提示修，改完重跑。
