# 100 题答案评测归因（自动生成）

> 生成时间：2026-11-09T10:17:22
> 尺子：`eval_questions_v4_100_revised2.jsonl`；旧尺子：`eval_questions_v4_100_revised.jsonl`

## 1. 轮次得分

| 跑次 | 旧尺子+旧评分器 | 新尺子+旧评分器 | 新尺子+新评分器 |
|---|---:|---:|---:|
| a_ab_r1_20261108_200158 | 73 | 74 | 78 |
| a_ab_r2_20261108_202017 | 70 | 72 | 76 |
| a_ab_r3_20261108_204000 | 72 | 75 | 76 |
| **中位** | **72** | **74** | **76** |

## 2. 失败分类（按轮次计）

| 原因 | 次数 |
|---|---:|
| missed_points | 43 |
| gold_not_in_context | 26 |
| refused | 1 |

## 3. 三轮全挂（15 道）

| qid | 主因 | 缺要点 | 禁用命中 | gold 不在上下文 |
|---|---|---|---|---|
| `ev2-04` | missed_points×3 | 疾病治疗类／疾病治疗／禁用词／不得作为保健功能声称／禁止使用／不得使用；防癌 |  | 0 |
| `ev2-25` | gold_not_in_context×3 | 400g；20 份 |  | 3 |
| `ev2-35` | gold_not_in_context×3 | 权威暗示／不能直接采信／不能采信 |  | 3 |
| `gen-C01_s06-paraphrase` | missed_points×3 | 95 分；必需要素缺 4 项；3 处 critical |  | 0 |
| `gen-C02_s01-paraphrase` | gold_not_in_context×3 | 32 岁；路过进店，看到货架上的蛋白粉；好奇 |  | 3 |
| `gen-C04_s11-paraphrase` | missed_points×3 | 给时间承诺；吃几个月就能停；一年左右 |  | 0 |
| `gen-C05_s14-scenario` | missed_points×3 | 陈述人群范围／告知不适宜人群／提供不适宜人群清单／说明产品的人群范围；建议咨询医生／建议客户带长辈咨询主诊医生／建议先问医生／建议问医生；不判断个人能否吃／不替 |  | 0 |
| `gen-C06_s23-paraphrase` | gold_not_in_context×3 | 不推荐；其他产品 |  | 3 |
| `gen-C08_s41-paraphrase` | missed_points×3 | 未给对比信息；无效回答；未陈述自身参数 |  | 0 |
| `gen-C10_s53-scenario` | missed_points×3 | critical |  | 0 |
| `gen-C11_s59-paraphrase` | gold_not_in_context×3 | E2；具体核实路径；30 |  | 3 |
| `gen-C13_s72-paraphrase` | missed_points×3 | 我看别人都一起吃 | 学员回答 F2 | 不套用他人／我看别人都一起吃，考察不套用他人／别人都一起吃，不能套用他人的情况／别人一起吃，考察不套用他人；效果更好； |  | 0 |
| `gen-C16_s102-scenario` | missed_points×3 | D001–D015；培训文档；P001–P020 |  | 0 |
| `gen-C16_s92-scenario` | missed_points×3 | 不急于决定 |  | 0 |
| `hq-03` | gold_not_in_context×3 | 哪款／哪一款／哪一件／产品名／指一下 |  | 3 |

## 4. 评分器/尺子翻转

| qid | 旧尺子旧评分器 | 新尺子旧评分器 | 新尺子新评分器 |
|---|---:|---:|---:|
| `ev2-43` | 0/3 | 1/3 | 1/3 |
| `gen-C05_s19-scenario` | 0/3 | 0/3 | 1/3 |
| `h2-04` | 0/3 | 0/3 | 1/3 |
| `h2-08` | 1/3 | 1/3 | 2/3 |
| `hq-10` | 1/3 | 1/3 | 2/3 |
| `ev2-14` | 1/3 | 1/3 | 3/3 |
| `ev2-20` | 1/3 | 1/3 | 3/3 |
| `ev2-29` | 1/3 | 3/3 | 3/3 |
| `gen-C13_s75-paraphrase` | 0/3 | 3/3 | 3/3 |
| `h2-01` | 2/3 | 2/3 | 3/3 |

## 5. 判读

- `gold_not_in_context` 是检索构图问题，先看该题 gold 是否真的在库/是否 gold 本身错。
- `missed_points` 是生成侧问题，优先看数字、单位、警示语、遵医嘱/就医等必答项。
- `forbidden` 若来自否定/引述语境，应继续修评分器；若是真实宣称，属产品护栏问题。
- `refused` 是产品护栏拒答，需区分真实违规与引述误拒。
