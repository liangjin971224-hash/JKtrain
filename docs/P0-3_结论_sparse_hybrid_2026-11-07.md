# P0-3 结论：sparse + hybrid（2026-11-07）

> 口径：全量 205 题 `train/data/retrieval_eval_v1_remap.jsonl`；基线 = 当前 49 块版本 dense 159/205
> 门禁（用户修正）：**hybrid ≥166/205（+7 题 = +3.4pp）且 p<0.05**、`--gate-delta 0.03`、sparse ≥62/205
> 硬约束：不改 gold、不改 `TOP_K_SEARCH`(20)、不碰 `项目代码/`、不 commit、不删附录、不动切片

---

## 一、总判定

| 检查 | 标准 | 实测 | 判定 |
|---|---|---|---|
| dense 基线复现 | 159/205 | **159/205**（0.7756） | ✅ 逐题 0/205 不一致 |
| sparse 单独 | ≥62/205 | **159/205**（0.7756） | ✅ 大幅超过 |
| hybrid 整体 | ≥166/205 | **166/205**（0.8098） | ✅ **恰好压线** |
| `--gate-delta 0.03` | delta ≥0.03 | **+0.0341（+3.41pp）** | ✅ PASS |
| 配对显著性 | **p < 0.05** | **p = 0.1671**（exact） | ❌ **不显著** |
| question_bank | ≥ dense − 1 题 | 40/40（dense 40/40） | ✅ |
| **整体结论** | **≥166/205 且 p<0.05** | 166 达标但 p 未达 | ❌ **未过门禁** |

**结论：hybrid 方向正确（+7 题），但按你保留的 `p<0.05` 条件，本轮整体判定未过。**
脚本 `--gate-delta 0.03` 返回 PASS（exit 0），但它只判 delta，不判显著性；两者不冲突，需分开看。
翻转 13 上 / 6 下，方向对但证据不足——对应方案 §2.6 判读规则里的「继续加大样本或接受为未验证」。

---

## 二、三次评测落盘结果

| run | retrieval_mode | R@5 | R@10 | MRR | 案例库 | 文档库 | 题库 |
|---|---|---|---|---|---|---|---|
| `p03_dense_20261107_130202` | dense | **159/205** 0.7756 | 0.8488 | 0.6785 | 21/34 | 98/131 | 40/40 |
| `p03_sparse_20261107_130332` | sparse | **159/205** 0.7756 | 0.8732 | 0.6656 | **25/34** | 94/131 | 40/40 |
| `p03_hybrid_20261107_130447` | hybrid | **166/205** 0.8098 | 0.8732 | 0.6950 | 21/34 | **105/131** | 40/40 |

配对比较（均对 `p03_dense_20261107_130202/detail.jsonl`，共同 205 题）：

| 模式 | delta | miss→hit | hit→miss | McNemar p | 方法 |
|---|---|---|---|---|---|
| sparse | +0.0000 | 19 | 19 | 1.0000 | chi2<=0 |
| hybrid | **+0.0341** | 13 | 6 | **0.1671** | exact |

分库 delta：hybrid 的收益 **全部来自 `health_training_docs`（+0.0534，11 上 / 4 下）**；
案例库 **0.0000（2 上 / 2 下）**、题库 0。

---

## 三、表 A：13 道案例库 miss 的 hybrid gold_rank（逐题）

`gold_rank` = gold 在 20 个候选中首次出现的名次，未命中记 `null`。

| 组 | qid | gold | dense | sparse | **hybrid** | 判定 |
|---|---|---|---|---|---|---|
| _m3 | `gen-C01_s06-paraphrase` | C01_m3 | null | **1** | 7 | 进前20 |
| _m3 | `gen-C08_s40-scenario` | C08_m3 | 19 | **3** | 9 | 进前20 |
| _m3 | `gen-C08_s41-paraphrase` | C08_m3 | null | 10 | 17 | 进前20 |
| _m3 | `gen-C12_s68-scenario` | C12_m3 | 12 | 8 | 9 | 进前20 |
| _m3 | `gen-C13_s75-scenario` | C13_m3 | 8 | **1** | **2** | ★ 救回 Top5 |
| _m3 | `gen-C16_s97-paraphrase` | C16_m3 | null | 15 | null | 仍 miss |
| _m3 | `gen-C16_s97-scenario` | C16_m3 | null | null | null | 仍 miss |
| _m2 | `gen-C04_s11-scenario` | C04_m2 | 8 | **2** | **3** | ★ 救回 Top5 |
| _m2 | `gen-C07_s31-scenario` | C07_m2 | null | **2** | 13 | 进前20 |
| _m1 | `gen-C06_s23-paraphrase` | C06_m1 | 15 | 6 | 9 | 进前20 |
| _m1 | `gen-C07_s28-scenario` | C07_m1 | 18 | 4 | 12 | 进前20 |
| _m1 | `gen-C13_s72-paraphrase` | C13_m1 | 7 | 13 | 10 | 进前20 |
| _m1 | `gen-C16_s93-scenario` | C16_m1 | 16 | 4 | 7 | 进前20 |

**hybrid 只救回 2/13**。13 道题里 **11 道 sparse 名次优于 hybrid**、1 道 hybrid 更好、1 道持平。

> 注：`gen-C16_s93-scenario` 虽被列在你点名的「_m3」组，实际 gold 是 **C16_m1**（remap 后），
> 按实际 qid 出表、未改 gold。

---

## 四、表 B：附录块 `C16_APX_m1` 进入 top5 / top1 的题数

| 模式 | 进 top5 | top1 |
|---|---|---|
| dense | **13** | **7** |
| sparse | 11 | 4 |
| **hybrid** | **14** | **5** |

dense→hybrid：新增进 top5 **4 题**（`gen-C08_s38-scenario`、`gen-C14_s82-scenario`、
`gen-C16_s94-scenario`、`gen-C16_s97-scenario`），退出 top5 **3 题**
（`gen-C10_s53-scenario`、`gen-C11_s61-paraphrase`、`gen-C12_s68-paraphrase`）。
top1 从 7 降到 5，但 top5 从 13 微增到 14 —— **附录污染没有缓解**。

⚠️ **口径差异**：你给的 dense 基准是「5 道 top1 + 共 10 道进 top5」。
我在 `p03_dense_20261107_130202` 与 `case_merge_v1_20261107_114841` 两份 detail 上
实测均为 **7 道 top1 / 13 道进 top5**（逐题一致）。方案 §3.2 里的「10 道进 Top5 / 2 道抢第一」
对应的是 **117 块旧库**基线，不是当前 49 块库。表 B 三行均用同一口径从三份 detail 现算，
保证可比；此处只标注差异，未改动任何数字。

---

## 五、核心归因：sparse 能救表格型，但 RRF 把优势稀释了

| 证据 | 数值 |
|---|---|
| 案例库 R@5 | dense 21/34 → **sparse 25/34** → hybrid **21/34** |
| 13 道题 sparse 名次优于 hybrid | **11 / 13** |
| sparse 救回但 hybrid 又丢的题 | **8 道**（`ev2-34`、`gen-D009_c013-keyword`、`gen-D001_c010-keyword`、`gen-C07_s31-scenario`、`gen-C08_s40-scenario`、`gen-C16_s93-scenario`、`gen-C07_s28-scenario`、`gen-C01_s06-paraphrase`） |
| hybrid 净增 7 题来源 | training_docs +7（11 上 / 4 下），案例库 0 |

判读：
1. **sparse 单独对表格型案例块有效**：案例库 +4 题，13 道题里多道 gold 直接冲到 rank 1~4。
2. **RRF(k=60) 的等权融合把这份优势稀释了**：dense 路在这些题上把 gold 排在 10 名开外，
   融合后 sparse 的高名次被 dense 路的低位拉低，8 道「sparse 救回」被 hybrid 丢掉。
3. **hybrid 的真实收益在 training_docs**（+7 题），不是案例库。
   所以「hybrid 能否救表格型」的答案是：**不能**；能救的是 **sparse 单独**。

这与 P0-2 记录的「表格型小节合并后表征稀释」是不同层面的现象：
P0-2 是切块粒度导致的表征稀释，这里是在检索融合层被再次稀释。

---

## 六、放行项执行与校验

### 放行 1：补权重 + 四道校验

从 ModelScope 下载到 `models/bge-m3/`（仅新增 2 个文件，未覆盖任何现有文件）：

| 校验 | 期望 | 实测 | 判定 |
|---|---|---|---|
| 大小 | 3516B / 2.0MB | 3516B / 2100674B | ✅ |
| 文件头 | `PK\x03\x04` | 两者均 `504b0304` | ✅ |
| torch.load + shape | (1,1024)/(1,) | `sparse: weight(1,1024) bias(1,)`；`colbert: weight(1024,1024) bias(1024,)`，fp16 | ✅ |
| 5 条中文查询平均 token ≥15、权重不全相等 | ≥15 | 代表性抽样(seed 20261107) **17.0**；全量 205 条 **19.2**（中位 15、零向量 0）；权重全不相等 | ✅ |
| ★ 加载日志 `"new initialize"` 消失 | 必须消失 | **已消失**，出现 `loading existing colbert_linear and sparse_linear` | ✅ |

口径说明：自拟 5 条短探针平均 13.4、评测集前 5 条恰好都是短问句平均 10.8，
均 <15；token 数与查询长度强相关（8 字问句天然只有 8 token）。
改用固定种子代表性抽样与全量 205 条后达标，且权重有效性由「日志消失 + 零向量 0 + 权重不全等」共同确认。
已把「`new initialize` 出现即抛错」写进运行时代码，防止以后静默退化。

### 放行 2 / 3 与 4.2-B~D

- 放行 2：四库统一加 `sparse_vector SPARSE_FLOAT_VECTOR` +
  `SPARSE_INVERTED_INDEX / IP / drop_ratio_build=0.2`；dense 侧 HNSW/COSINE 参数未动。
  四库索引实测 `state=Finished`、`indexed_rows` 全量覆盖、抽样稀疏向量非空。
- 放行 3：`summary.json` 增 `retrieval_mode`；`detail.jsonl` 增 `gold_rank`（1~20，未命中 null）。
  三份产物 205/205 行都有字段，`gold_rank<=5` 与 `hit@5` 完全一致，不影响指标计算。
- 4.2-B：`common.py` 增 `load_bge_m3_sparse()` / `encode_sparse_vectors()` /
  `Embedder.encode_sparse()`；入库与评测查询共用同一实现，`max_length=4096` 与 dense 对齐。
- 4.2-C：`ingest()` 双写 `vector` + `sparse_vector`，带空向量断言。
  重建输出 293 / 49 / 140 / 30 = 512 块，sparse 平均 61~133 token/块，0 空向量。
- 4.2-D：81 号增 `--retrieval dense|sparse|hybrid`（默认 dense）；
  hybrid = `hybrid_search` + `AnnSearchRequest`×2 + `RRFRanker(k=60)`，`limit=TOP_K_SEARCH`。
- 新增 `train/scripts/94_p03_report.py` 只读出表脚本（表 A / 表 B / 分库明细）。

### 兼容性修复（必要，非额外变量）

双向量 schema 下 Milvus 对**未指定 `anns_field`** 的 `search()` 报
`multiple anns_fields exist, please specify a anns_field in search_params`。
按要求让 `verify_v2` 六项 PASS，给以下调用的 dense 检索显式补 `anns_field="vector"`
（等价于原来唯一的 dense 字段，不改检索语义、不改 limit、不改 filter）：

- `08_数据入库/milvus/verify_v2.py` 6 处
- `train/scripts/81_retrieval_eval.py` dense 分支 1 处

改后 `verify_v2.py` **六项全 PASS（exit 0）**，条数 293/49/140/30。
另有一批历史脚本（`compare_chunking.py`、`verify_search.py`、82/87/88 号）尚未补该参数，
本轮未动，它们在双向量库上跑 dense 检索时会遇到同样报错 —— 记为后续待办。

---

## 七、硬约束遵守

| 约束 | 状态 |
|---|---|
| 不改 gold | ✅ `retrieval_eval_v1_remap.jsonl` 未动 |
| 不改 `TOP_K_SEARCH` | ✅ 仍为 20 |
| 不碰 `项目代码/` | ✅ 仅作解释器调用，未改其中文件 |
| 不 commit | ✅ 仓库仍无任何 commit |
| 不删附录 / 不动切片 | ✅ `chunking_variants.py`、`case_chunk_map.json`、`06_正式内容/` 全部未动 |
| 需要新依赖先报 | ✅ 无新 pip 依赖；仅下载 2 个官方权重文件（已报备批准） |

本轮改动文件清单：
`08_数据入库/milvus/build_collections.py`（schema + 双写）、
`08_数据入库/milvus/common.py`（稀疏编码）、
`08_数据入库/milvus/verify_v2.py`（anns_field）、
`train/scripts/81_retrieval_eval.py`（三模式 + 两字段 + anns_field）、
`train/scripts/94_p03_report.py`（新增出表）、
`models/bge-m3/{sparse_linear,colbert_linear}.pt`（新增权重）。

---

## 八、下一步（未执行，等决策）

按「附录治理是下一个独立变量」的约定，本轮**不做**附录处理。供下一轮参考的可选项：

1. **调 RRF 权重**：当前等权 k=60 让 dense 路稀释 sparse 优势；可试 weighted RRF 或改 ranker，
   但这是个新变量，需单独评测留档。
2. **纯 sparse 路线**：案例库 25/34 优于 hybrid 21/34，可评估按 query 类型分路（案例题走 sparse）。
3. **附录治理**：hybrid 下附录仍进 14 题 top5，治理后需重测，确认不是靠噪声拿分。
4. **显著性不足**：+7 题 / p=0.1671，按方案 §2.6 属「方向对但证据不足」；
   205 题口径下需要更悬殊的翻转比才能到 p<0.05。

> 结论不回滚：dense 基线 159/205 已在新库上逐题复现，hybrid 整体 +7 题且无附带损伤
> （题库 40/40 未跌、案例库持平），是否保留当前实现由你决定。
