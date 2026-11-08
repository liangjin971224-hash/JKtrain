# 进度汇报：P0 检索质量三件套 / 当前停在 P0-3 §4

> 汇报时间：2026-11-07 12:30　执行范围：`docs/实施方案_P0检索质量三件套_2026-11-07.md`
> 口径：全量 205 题（`train/data/retrieval_eval_v1_remap.jsonl`），所有数字只取 `train/eval_runs/` 落盘 `summary.json`
> 当前状态：**P0-3 停在 4.2-B 阻塞点**，等 3 项用户确认；未重建、未跑三次评测

---

## 一、总体进度

| 阶段 | 唯一变量 | 状态 | 主口径 R@5 | 结论 |
|---|---|---|---|---|
| P0-1 | 数据划分（holdout 55） | ✅ 完成 | 156/205（0.7610）不变 | 分层 11/12/9/23 逐格相符，切分未动库 |
| P0-2 | 案例切片粒度 117→49 | ✅ 完成（门禁未达） | **159/205（0.7756）** | 四指标同向改善、无附带损伤，**保留 49 块不回滚** |
| P0-3 | 是否使用稀疏向量 | ⏸ **进行中，停在 4.2-B** | dense 基线已复现 **159/205** | 见第三节 |

P0-2 判读留档：对照同语义旧切片 `case_expand_gold_baseline`（157/205、案例 19/34），
案例库 21/34（门禁 ≥22/34，差 1 题）、整体 159/205（门禁 163/205，差 4 题），
`--gate-delta 0.03` 实测 +0.0098 FAIL，翻转 4 上 / 2 下、McNemar p=0.6875 不显著。
已记录规律：**表格型小节合并后表征稀释**。

### P0-3 分步进度

| 步骤 | 内容 | 状态 | 落盘 / 证据 |
|---|---|---|---|
| 第 1 步 | `p03_dense` 复现基线 | ✅ **通过** | `train/eval_runs/p03_dense_20261107_120943/` |
| 4.2-A | pymilvus 版本检查 + 四库加 `sparse_vector` / 索引 | ✅ **完成，已打印待确认** | `08_数据入库/milvus/build_collections.py` |
| 4.2-B | `Embedder.encode_sparse()` / FlagEmbedding | ⛔ **阻塞** | 见第四节，权重缺失 |
| 4.2-C | 入库写双向量 | ⬜ 未执行 | — |
| 4.2-D | 81 加 `--retrieval dense\|sparse\|hybrid`、summary 加 `retrieval_mode` | ⬜ 未执行 | — |
| 4.3 | 重建 → `p03_dense` / `p03_sparse` / `p03_hybrid` 三次评测 | ⬜ 未执行 | — |
| 判定 | hybrid ≥168/205 且 p<0.05；sparse ≥62/205 | ⬜ 无法判定 | — |
| 汇报 | 8 道关注题 hybrid 逐题名次表 | ⬜ 无数据，未产出 | 见第六节 |

---

## 二、第 1 步：dense 基线复现（通过）

命令：`81_retrieval_eval.py --eval train/data/retrieval_eval_v1_remap.jsonl --label p03_dense`

| 指标 | 实测 | 前置基线 | 判定 |
|---|---|---|---|
| 整体 R@5 | **159/205 = 0.7756** | 159/205 | ✅ |
| `health_case_library` | 21/34 = 0.6176 | 21/34 | ✅ |
| `health_training_docs` | 98/131 = 0.7481 | 98/131 | ✅ |
| `health_question_bank` | 40/40 = 1.0000 | 40/40 | ✅ |
| R@10 / MRR | 0.8488 / 0.6785 | — | 与 P0-2 定稿一致 |

额外做了逐题比对：与 `case_merge_v1_20261107_114841/detail.jsonl` 的 205 个 qid
**命中状态与 `ranked_head` 前 10 全部一致，0/205 不一致**。
排除「总数巧合」，确认库与代码自 P0-2 定稿后无变动，可作为 P0-3 唯一基线。

---

## 三、4.2-A：版本检查 + 新 schema（完成，待确认）

### 3.1 前置门禁

| 检查 | 实测 | 要求 | 判定 |
|---|---|---|---|
| pymilvus | **2.5.18** | ≥2.4.4 | ✅ |
| `MilvusClient.hybrid_search` | 可用 | 需存在 | ✅ |
| `AnnSearchRequest` / `RRFRanker` | 可导入 | 需存在 | ✅ |
| `DataType.SPARSE_FLOAT_VECTOR` | 104 | 需存在 | ✅ |

运行环境：项目虚拟环境 `项目代码/.venv`（Python 3.11.16）。
全局 `python` 为 3.13，缺 pymilvus / torch / sentence-transformers，不可用于本任务。
同环境版本：`torch 2.8.0+xpu`、`transformers 5.19.0`、`sentence-transformers 6.1.0`、`FlagEmbedding 1.4.2`。

### 3.2 代码改动（仅 1 个文件）

`08_数据入库/milvus/build_collections.py`：
把原 `create_collection()` 里的 schema 构造抽成只读函数 `build_schema_and_index(client, with_kp)`，
`create_collection()` 调用它（drop → create 顺序与行为不变）。这样可只读打印新 schema 而不落库。

### 3.3 新 schema（四个 collection 统一；question 另加 KP 字段）

```
字段：
  id                  INT64            primary, auto_id
  vector              FLOAT_VECTOR     dim=1024                    ← 未动
  sparse_vector       SPARSE_FLOAT_VECTOR                          ★ 新增
  chunk_id VARCHAR64  | text VARCHAR12000 | source_dir VARCHAR64
  source_file VARCHAR128 | doc_id VARCHAR32 | title VARCHAR256
  category VARCHAR32  | permission_level VARCHAR32 | review_status VARCHAR32
  data_origin VARCHAR32 | section VARCHAR256 | chunk_index INT32 | extra VARCHAR2000
  [仅 health_question_bank] knowledge_point VARCHAR64 / kp_name VARCHAR128
                            question_type VARCHAR16 / difficulty INT32

索引：
  vector              HNSW / COSINE / M=16, efConstruction=200     ← 未动
  sparse_vector       SPARSE_INVERTED_INDEX / IP / drop_ratio_build=0.2  ★ 新增
  permission_level / category / doc_id   INVERTED                  ← 未动
  [仅 health_question_bank] knowledge_point INVERTED               ← 未动
```

四库分别打印核对过：`health_training_docs` / `health_case_library` / `health_self_test` 各 16 字段 + 5 索引，
`health_question_bank` 20 字段 + 6 索引。**dense 侧参数一字未动**，保证 `--retrieval dense` 与历史基线可比。

### 3.4 现场状态（重要）

- 打印用的是真实构造对象，**未 create_collection、未 drop、未写入任何数据**。
- 线上四库仍是 P0-2 的旧 schema（无 `sparse_vector`）：293 / **49** / 140 / 30。
- `verify_v2.py` 无 schema 字段断言，`EXPECT` 保持 293/49/140/30 即可，无需修改。
- 当前代码处于「schema 已加字段、ingest 尚未写值」的中间态，**此时不能直接跑重建**，否则会写出缺 `sparse_vector` 的行。

---

## 四、4.2-B：⛔ 阻塞详情（停止原因）

### 4.1 先排除「装不上 / torch 冲突」

| 检查 | 结果 |
|---|---|
| `FlagEmbedding` 是否已装 | ✅ 1.4.2，无需新 pip 依赖 |
| `from FlagEmbedding import BGEM3FlagModel` | ✅ 成功 |
| `BGEM3FlagModel('models/bge-m3', device='cpu')` | ✅ 4.1s 加载成功，无报错 |
| 与 `torch 2.8.0+xpu` / `transformers 5.19.0` 冲突 | ❌ 无冲突 |
| `encode(..., return_sparse=True)` 签名 | ✅ 有 `return_sparse` 参数 |

**结论：不是安装失败，也不是 torch 冲突。** 但发现了另一类致命问题。

### 4.2 根因：本地模型缺官方稀疏权重

| 证据 | 内容 |
|---|---|
| `models/bge-m3/pytorch_model.bin` | 仅 **391 个 XLMR 主干 key**，`sparse` / `colbert` 相关 key **0 个** |
| 本地目录文件 | 10 个（含 `1_Pooling`），**无 `sparse_linear.pt`、无 `colbert_linear.pt`** |
| 硬链接源 `D:/models/BAAI/bge-m3` | 同样只有 6 个基础文件 |
| FlagEmbedding `finetune/.../m3/runner.py:88-97` | 两个 .pt 不存在 → 直接 `torch.nn.Linear` 随机初始化，日志原文：<br>`The parameters of colbert_linear and sparse linear is new initialize. Make sure the model is loaded for training, not inferencing` |
| 实测 `m.model.sparse_linear` | `Linear(in_features=1024, out_features=1, bias=True)`，随机权重 |

**实测 5 句 `lexical_weights` token 数：`0 / 0 / 0 / 3 / 0`，平均 0.6**，等于 sparse 向量几乎全空。
若照此入库，`sparse ≥ 62/205` 必然崩，hybrid 的 RRF 融合也会退化为单路 dense。
命中方案 §8 提醒 3「**sparse 崩到接近 0，一律停下来报告，不要自行修一下让它过**」，故停止。

### 4.3 补救可行性（只做了只读探测，未下载）

| 项 | 结果 |
|---|---|
| ModelScope 官方 `BAAI/bge-m3` 仓库 | 18 个文件，确认含 `sparse_linear.pt`、`colbert_linear.pt` |
| 体积 | `colbert_linear.pt` ≈ 2.00 MB；`sparse_linear.pt` 极小（1024→1 线性层） |
| 文件真实性 | 跟随重定向取前 8 字节 = `50 4b 03 04`（`PK\x03\x04`），是真 torch zip，非 HTML |
| HF 直连 `huggingface.co` | ❌ `502 CONNECT tunnel failed` |
| ModelScope | ✅ 可达 |
| 加载兼容性 | runner 直接 `load_state_dict`、无 key 转换；官方权重 key 应匹配，但**补后必须实测非空**才能继续 |

**未自行手写 sparse、未下载任何权重、未安装任何新依赖。**

---

## 五、历次评测数字总表（全部来自落盘 `summary.json`）

| eval_run | n | R@5 | R@10 | MRR | 案例库 | 文档库 | 题库 |
|---|---|---|---|---|---|---|---|
| `baseline_v1_final_20261107_032716` | 205 | 0.7610 | 0.8390 | 0.6590 | 0.5294 | 0.7481 | 1.0 |
| `baseline_v1_recheck_20261107_104850` | 205 | 0.7610 | 0.8390 | 0.6590 | 0.5294 | 0.7481 | 1.0 |
| `baseline_holdout55_20261107_104811` | 55 | 0.7273 | 0.8545 | 0.6458 | 0.4444 | 0.7143 | 1.0 |
| `case_expand_gold_baseline_20261107_111654` | 205 | 0.7659 | 0.8537 | 0.6701 | 0.5588 | 0.7481 | 1.0 |
| `case_merge_v1_20261107_114841`（P0-2 定稿） | 205 | **0.7756** | 0.8488 | 0.6785 | **0.6176** | 0.7481 | 1.0 |
| `p03_dense_20261107_120943`（本轮复现） | 205 | **0.7756** | 0.8488 | 0.6785 | 0.6176 | 0.7481 | 1.0 |

`case_merge_v1` 配对比较：对照 `case_expand_gold_baseline`，delta +0.0098、
miss→hit 4 / hit→miss 2、McNemar p=0.6875（exact，不显著）。

---

## 六、8 道关注题：口径已对齐，hybrid 名次待 4.3 产出

你点名的 6 道 `_m3` 失败题与 2 道回退题，在 remap 评测集里都是**带后缀的 qid**，
且部分简写对应多个变体。已核对出的精确口径如下（`dense` 列即 `p03_dense` 基线）：

| # | 类别 | 精确 qid | gold | dense 名次 | hit@5 | hybrid 名次 |
|---|---|---|---|---|---|---|
| 1 | _m3 失败 | `gen-C12_s68-scenario` | C12_m3 | >10 | ✗ | ⬜ 待 4.3 |
| 2 | _m3 失败 | `gen-C16_s97-paraphrase` | C16_m3 | >10 | ✗ | ⬜ 待 4.3 |
| 3 | _m3 失败 | `gen-C08_s40-scenario` | C08_m3 | >10 | ✗ | ⬜ 待 4.3 |
| 4 | 点名题（gold 实为 m1） | `gen-C16_s93-scenario` | **C16_m1** | >10 | ✗ | ⬜ 待 4.3 |
| 5 | _m3 失败 | `gen-C01_s06-paraphrase` | C01_m3 | >10 | ✗ | ⬜ 待 4.3 |
| 6 | _m3 失败 | `gen-C08_s41-paraphrase` | C08_m3 | >10 | ✗ | ⬜ 待 4.3 |
| 7 | 回退题 | `gen-C04_s11-scenario` | C04_m2 | 8 | ✗ | ⬜ 待 4.3 |
| 8 | 回退题 | `gen-C07_s28-scenario` | C07_m1 | >10 | ✗ | ⬜ 待 4.3 |

两点需你确认：

1. **变体歧义**：`gen-C12_s68-paraphrase` 是 **hit(rank 3)**、`gen-C04_s11-paraphrase` 是 **hit(rank 1)**，
   不属于「失败题」。上表按失败变体取；若你要两个变体都列，可补 2 行。
   另外 gold 含 `_m3` 且 dense miss 的还有 `gen-C16_s97-scenario`、`gen-C13_s75-scenario`（rank 8），
   未在你点名的 6 道内。
2. **名次取不到**：现有 `detail.jsonl` 只存 `ranked_head`（前 10），上表 6 道 >10 的题无法给确切名次。
   建议 4.2-D 给 detail 增一个 `gold_rank` 字段（gold 在 20 个候选里首次出现的名次，未命中记 null），
   **不改 `TOP_K_SEARCH`、不改指标计算、不影响已有基线**。

这张表是判断「hybrid 能否救表格型」的唯一证据，将在 4.3 跑完后逐题补齐。

---

## 七、硬约束遵守情况

| 约束 | 状态 | 说明 |
|---|---|---|
| 不改 gold | ✅ | 未触碰 `retrieval_eval_v1_remap.jsonl` 及任何 gold 字段 |
| 不改 `TOP_K_SEARCH` | ✅ | 仍为 20（`81_retrieval_eval.py:53`） |
| 不碰 `项目代码/` | ✅ | 仅改 `08_数据入库/milvus/build_collections.py`；`项目代码/.venv` 只作为解释器调用 |
| 不 commit | ✅ | 仓库无任何 commit，`git log` 为空 |
| 需要新依赖先报 | ✅ | FlagEmbedding 本已安装；补权重属新增模型文件，已停报、未下载 |
| 不自行手写 sparse | ✅ | 未实现任何手写稀疏编码 |

此外：未重建 Milvus、未 drop collection、未跑 sparse/hybrid 评测、未改 `verify_v2.py`、未改 81 号脚本。

---

## 八、待确认 3 件事（确认后才继续）

1. **是否允许从 ModelScope 补下载 `sparse_linear.pt` + `colbert_linear.pt` 到 `models/bge-m3/`？**
   这是新增 2 个模型权重文件（非 pip 依赖）。允许的话先校验文件完整性与可加载性，
   再实测 `lexical_weights` 非空，才继续 4.2-C。若不允许，P0-3 只能停在 dense，需回滚 4.2-A 改动。
2. **4.2-A 的新 schema 是否确认？**（四库统一加字段与索引、dense 参数未动）
3. **是否允许给 `detail.jsonl` 增 `gold_rank` 字段？**以及第六节的 8 个精确 qid 是否就是你要的那 8 道？

> 口径备注：本轮判定门禁按你的指令记为 **hybrid ≥168/205 且 p<0.05、sparse ≥62/205**。
> 方案 §4.3 表格把 168 写成「dense 156/205 + 5 题」自相矛盾，且 156 是 P0-2 之前的旧基线
> （当前 dense 基线为 159/205）。此处只如实记录，未擅自改门禁数值，请一并确认。

### 确认后的执行顺序

```
补权重 → 校验 → 4.2-B encode_sparse() → 4.2-C 双向量入库
       → 4.2-D 81 加 --retrieval / retrieval_mode / gold_rank
       → 4.3 重建四库 → verify_v2
       → p03_dense（须复现 159/205）
       → p03_sparse --compare p03_dense/detail.jsonl
       → p03_hybrid --compare p03_dense/detail.jsonl --gate-delta 0.03
       → 判定 + 8 题逐题名次表
```

---

## 九、本轮产物清单

| 类型 | 路径 |
|---|---|
| 评测入档 | `train/eval_runs/p03_dense_20261107_120943/`（summary.json + detail.jsonl） |
| 代码改动 | `08_数据入库/milvus/build_collections.py`（新增 `build_schema_and_index()`） |
| 本汇报 | `docs/进度汇报_P0-3_sparse_hybrid_2026-11-07.md` |
| 工作日志 | `.workbuddy/memory/2026-11-07.md`（已追加 P0-3 段） |
| 方案依据 | `docs/实施方案_P0检索质量三件套_2026-11-07.md` §4 / §8 |
