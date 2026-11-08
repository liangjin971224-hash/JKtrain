# Codex 任务提示词（案例 rubric 修订，复制给 Codex）

你是评测标注修订者。请在仓库根目录 `D:\AI260803\健康培训移动` 下，修订 31 条案例 rubric，只产出一个文件，不要改其他文件。

## 背景

当前 rubric 出现了**判词形误报**：回答语义正确，但因为没有逐字命中 `must_include` 而判失败。例如：

- 回答写「孕妇、乳母」，rubric 只认「孕妇和乳母」；
- 回答写「致命错误」，rubric 只认「一票否决」；
- 回答写「免疫低下的成人」，rubric 只认「免疫力低下者」。

修订目标：把**合法的同义/格式变体**收进 `any_of`，让评分看语义；**不是放宽事实标准**。

## 先读这四份

1. `train/annotation_wp_v3/revision/ANNOTATION_BRIEF_REVISE.md`（完整规则）
2. `train/annotation_wp_v3/revision/targets.jsonl`（31 题 + gold 正文）
3. `train/annotation_wp_v3/revision/current_rubrics.jsonl`（当前 rubric）
4. `train/annotation_wp_v3/revision/failures.jsonl`（100 题基线里的失败证据：question / answer / missed）

## 输出

```
train/annotation_wp_v3/revision/outputs/answer_rubrics_revised.jsonl
```

31 行，每行：

```json
{"qid": "...", "expect": {"must_include": ["..."], "any_of": [["...","..."]], "must_not_include": ["..."]}}
```

字段恰好 `qid` 与 `expect`；`expect` 只允许三个键。

## 硬性规则

1. 对每个 `must_include`：
   - 有合法同义/格式变体（顺序、单复数、简称、同义词）→ **移出 must_include**，写成 `any_of` 的一个组（2~5 个变体）；
   - 是不可替代的硬事实（编号、数值、专有名词）→ 保留在 `must_include`。
2. 修订后：`must_include` **1~5 条**；`any_of` 每组 2~5 条；`must_include + any_of 组数` 合计 **2~5 个必答点**。
3. `any_of` 组内每个变体必须语义等价，且**至少一个**能在 gold 正文里找到。
4. `must_not_include` 0~3 条；仍不得是正确否定句的子串（正确说法是「不可以单独退」时，不要写「可以单独退」）。
5. 不得为通过而删除核心必答点；`failures.jsonl` 里 `answer` 确实漏掉的事实，**不要**加进 any_of，保留严格项。
6. 31 行，qid 与 targets 一一对应；不要修改任何输入文件。

## 交付后自检（必须跑）

```bash
项目代码/.venv/Scripts/python.exe train/scripts/104_validate_revised_rubrics.py --wp-dir train/annotation_wp_v3
```

要求：**硬违规 0、判定 PASS**。有 HARD 按提示修（常见：any_of 组在 gold 找不到任一变体、总必答点不在 2~5、must_not 被否定句误伤），改完重跑直到 PASS。WARN 里「必答点从 N 降到 M」要人工确认是否放宽过度。

## 最终回报

1. 产出文件路径与行数；
2. 校验器输出（目标/实收/硬违规/判定）；
3. 列出你从 must_include 移到 any_of 的短语（qid + 短语），便于人工复核；
4. 是否修改了其他文件（应为「否」）。
