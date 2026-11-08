# 答案评测 rubric 标注任务书（案例库，v3）

> 由 `100_answer_eval_workpackage.py` 于 2026-11-08T12:45:42+08:00 生成。
> 执行者：Codex。产出写入本目录 `outputs/answer_rubrics.jsonl`，逐行合法 JSON，UTF-8。

## 任务

输入 `targets.jsonl`（31 行）：每行一道案例题 + 它的 gold 知识块正文。
为**每道题**写一份机器可判的评分要点，写入：

    {"qid": "...", "expect": {"must_include": [...], "any_of": [...], "must_not_include": [...]}}

字段说明：

- `must_include`：回答**必须包含**的短语（2~5 个）。必须是 gold 正文里能核实的事实
  （编号、数值、术语、步骤名），不要写整句。
- `any_of`：同义表达分组；每组至少命中一个即算通过。用于「遵医嘱／听医生／由医生决定」
  这类等价说法。没有同义表达就给空数组。
- `must_not_include`：回答**不得出现**的短语（0~3 个）。只写明确的错误表述。

## 硬性规则

1. 每个 `must_include` 必须是 **gold 正文可核实的短语**，长度 2~20 字；不要写
   「要回答得准确」这种不可判的句子。
2. `must_not_include` **不能是某个正确否定句的子串**。
   反例（v3 踩过）：「可以单独退」会命中正确回答「**不**可以单独退」。
   如果必须写，写成完整错误句（如「赠品可以单独退换，无需按促销规则」），
   或者改用 `must_include` 约束正确说法。
3. 不要写跨产品对比、不要编造 gold 里没有的数字/结论。
4. 每题 `must_include` 2~5 条；`any_of` 每组 2~5 个同义短语；`must_not_include` 0~3 条。
5. 输出 31 行，`qid` 与 targets 一一对应，不多不少。
6. 不要修改 targets.jsonl、任务书或任何脚本。

## 交付后自检

```bash
项目代码/.venv/Scripts/python.exe train/scripts/101_validate_answer_rubrics.py --wp-dir train/annotation_wp_v3
```

要求硬违规 0、判定 PASS；有 HARD 按提示逐条修，改完重跑。
