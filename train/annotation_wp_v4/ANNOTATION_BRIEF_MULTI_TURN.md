# 多轮问答评测 rubric 任务书（20 组）

> 由 `105_build_multi_turn_eval.py` 于 2026-11-08T16:25:36+08:00 生成。
> 执行者：Codex。产出写入本目录 `outputs/multi_turn_rubrics.jsonl`。

## 任务

输入 `targets.jsonl`（20 组，每组含 `turns`、`final_question`、`expected_standalone`
与 `reference_chunks`）。为**每组最后一轮**写机器可判的评分要点：

    {"group_id": "...", "expect": {"must_include": [...], "any_of": [...], "must_not_include": [...]}}

- `must_include`：2~5 条，必须是 reference_chunks 里能逐字找到的短语（编号/数值/术语/步骤），
  长度 2~20 字；
- `any_of`：同义表达分组，每组 2~5 个，至少命中一个；
- `must_not_include`：0~3 条明确错误表述；**不得**是正确否定句的子串
  （「不可以单独退」→ 不要写「可以单独退」）。

## 硬性规则

1. 20 行，group_id 与 targets 一一对应。
2. 只评**最后一轮**的答案（前面几轮用于建立上下文，不评分）。
3. 不编造 reference_chunks 里没有的数字/结论。
4. 输出 UTF-8 无 BOM，一行一个 JSON；字段恰好 `group_id` 与 `expect`。

## 交付后自检

```bash
项目代码/.venv/Scripts/python.exe train/scripts/106_validate_multi_turn_rubrics.py
```

要求硬违规 0、判定 PASS；有 HARD 按提示修，改完重跑。
