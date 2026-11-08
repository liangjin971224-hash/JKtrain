# Codex 任务提示词（多轮评测 rubric，复制给 Codex）

你是评测标注执行者。请在仓库根目录 `D:\AI260803\健康培训移动` 下，为 20 组多轮对话的**最后一轮**写答案评分要点，只产出一个文件，不要改其他文件。

## 任务

读取：

1. `train/annotation_wp_v4/ANNOTATION_BRIEF_MULTI_TURN.md`（完整口径）
2. `train/annotation_wp_v4/targets.jsonl`（20 组：turns / final_question / expected_standalone / reference_chunks）

为每组的**最后一轮**写一份机器可判的 rubric，写入：

```
train/annotation_wp_v4/outputs/multi_turn_rubrics.jsonl
```

共 **20 行**，每行一个 JSON 对象：

```json
{"group_id": "...", "expect": {"must_include": ["..."], "any_of": [["...","..."]], "must_not_include": ["..."]}}
```

字段必须**恰好**是 `group_id` 与 `expect`；`expect` 内只允许三个键。UTF-8 无 BOM，`\n` 换行。

## 规则

1. **must_include 2~5 条**：必须是 `reference_chunks` 里能逐字找到的短语（编号、数值、术语、步骤），长度 2~20 字；不要写整句。
2. **any_of**：同义表达分组，每组 2~5 个短语，至少命中一个即算通过；没有同义表达给 `[]`。
3. **must_not_include 0~3 条**：只写明确错误表述。
   ★ 绝对不能写会被正确否定句误伤的子串：正确回答是「不可以单独退」时，不要写「可以单独退」；要写就写完整错误句。
4. 只评最后一轮；前面几轮用于建立上下文，不写评分点。
5. 不编造 `reference_chunks` 里没有的数字/结论。
6. 20 行，group_id 与 targets 一一对应，不多不少。
7. 不要修改 targets、任务书或任何脚本。

## 交付后自检（必须跑）

```bash
项目代码/.venv/Scripts/python.exe train/scripts/106_validate_multi_turn_rubrics.py
```

要求：**硬违规 0、判定 PASS**。有 HARD 按提示逐条改（常见：must_include 在参考块找不到、数量不对、must_not 被否定句误伤），改完重跑直到 PASS。

## 最终回报

1. 产出文件路径与行数；
2. 校验器输出（目标/实收/硬违规/判定）；
3. 若有不一致，列出前 10 条；
4. 是否修改了其他文件（应为「否」）。
