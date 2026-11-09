# Agent 真实 E2E 评测

> 通过 **9/10**；禁用 0；P95 10.5s；门禁 **PASS**

| 任务 | 期望工具 | 实际工具 | 拒答 | 结果 | 耗时 |
|---|---|---|---|---|---|
| E01 | graph_facts,kb_answer | graph_facts,kb_answer | 否 | PASS | 5.28s |
| E02 | kb_answer | kb_answer | 否 | PASS | 7.45s |
| E03 | kb_answer | kb_answer | 否 | PASS | 10.5s |
| E04 | kb_answer | graph_facts,kb_answer | 否 | FAIL | 4.4s |
| E05 | graph_facts,kb_answer | graph_facts,kb_answer | 否 | PASS | 6.07s |
| E06 | graph_facts,kb_answer | graph_facts,kb_answer | 否 | PASS | 3.52s |
| E07 | my_drill_reviews,weakness_summary | my_drill_reviews,weakness_summary | 否 | PASS | 0.23s |
| E08 | my_drill_reviews,weakness_summary | my_drill_reviews,weakness_summary | 否 | PASS | 0.25s |
| E09 | build_exam | build_exam | 否 | PASS | 0.15s |
| E10 | submit_monthly_report | submit_monthly_report | 否 | PASS | 0.12s |
