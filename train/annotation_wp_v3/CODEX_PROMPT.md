# Codex 任务提示词（答案 rubric，复制给 Codex）

你是评测标注执行者。请在仓库根目录 `D:\AI260803\健康培训移动` 下，为 31 道案例题补答案评分要点，只产出一个文件，不要改其他文件。

## 任务

读取：

1. `train/annotation_wp_v3/ANNOTATION_BRIEF_ANSWER_RUBRIC.md`（完整口径）
2. `train/annotation_wp_v3/targets.jsonl`（31 行：qid / question / gold_chunks[]）

为每道题写一份机器可判的评分要点，写入：

```
train/annotation_wp_v3/outputs/answer_rubrics.jsonl
```

共 **31 行**，每行一个 JSON 对象：

```json
{"qid": "...", "expect": {"must_include": ["..."], "any_of": [["...","..."]], "must_not_include": ["..."]}}
```

字段必须**恰好**是 `qid` 与 `expect`；`expect` 内只允许 `must_include` / `any_of` / `must_not_include` 三个键。UTF-8 无 BOM，`\n` 换行。

## 规则

1. **must_include 2~5 条**：必须是 gold 正文里能逐字找到的短语（编号、数值、术语、步骤名），长度 2~20 字；不要写整句、不要写「回答准确」这类不可判的话。
2. **any_of**：同义表达分组，每组 2~5 个短语，至少命中一个即算通过；没有同义表达就给 `[]`。
3. **must_not_include 0~3 条**：只写明确的错误表述。
   ★ 绝对不能写会被正确否定句误伤的子串：例如正确回答是「**不**可以单独退」，就不要写 `可以单独退`；要写就写完整错误句（如「赠品可以单独退换，无需按促销规则」）。
4. 不编造 gold 里没有的数字/结论；不问跨产品对比。
5. 31 行，qid 与 targets 一一对应，不多不少。
6. 不要修改 targets、任务书或任何脚本。

## 交付后自检（必须跑）

```bash
项目代码/.venv/Scripts/python.exe train/scripts/101_validate_answer_rubrics.py --wp-dir train/annotation_wp_v3
```

要求：**硬违规 0、判定 PASS**。有 HARD 按提示逐条改（常见：must_include 在 gold 里找不到、数量不对、must_not 会被否定句误伤），改完重跑直到 PASS。

## 最终回报

1. 产出文件路径与行数；
2. 校验器输出（目标/实收/硬违规/判定）；
3. 若有不一致，列出前 10 条；
4. 是否修改了其他文件（应为「否」）。
