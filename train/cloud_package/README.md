# 云端训练包（BGE-M3 600 对，2026-11-07）

## 0. 环境（首次实例必须装）

```bash
python -m pip install -q -r requirements.txt
python -c "import FlagEmbedding, torch; print(FlagEmbedding.__file__, torch.__version__, torch.cuda.is_available())"
```

> 若 pip 试图**替换/卸载 torch**，立刻 Ctrl-C，改用：
> `python -m pip install -q --no-deps -r requirements.txt`
> 然后按报错单独补缺的包。实例自带的 CUDA torch 不要动。

上传整个目录到 /mnt/workspace/cloud_package，然后：

```bash
cd /mnt/workspace/cloud_package
bash run_cloud.sh            # 训练 → 合并 → sanity → bge-m3-ft-v1.tar.gz
```

> 本包数据：547 train / 53 val（600 对 = 450 新增 + 150 现有）。
> `passage_max_len=1024`（案例块 42/80 超过 512 字，512 会截掉尾部）；
> `per_device_batch=2 × accum=8`，每步 token 数与 4×4 相同，显存不变。

## 下载后本地门禁（任一失败即回滚）

> ★ **顺序**：换了 embedding 模型后，query 向量与库里旧 doc 向量不在同一空间，
> **重建 collection 必须在检索门禁之前**，否则得到接近 0 的假 FAIL。

| 顺序 | 门禁 | 命令 |
|---|---|---|
| 1 | 维度=1024 | `python -c "from sentence_transformers import SentenceTransformer; print(SentenceTransformer('models/bge-m3-ft-v1').get_embedding_dimension())"` |
| 2 | **用新模型重建 4 个 collection** | `EMBED_MODEL_DIR=models/bge-m3-ft-v1 python 08_数据入库/milvus/build_collections.py` → `verify_v2.py` 核对 **293/49/140/30=512** |
| 3 | **holdout 检索门禁（主判据）** | `python train/scripts/92_app_retrieval_eval.py --eval train/data/retrieval_eval_v1_holdout_remap.jsonl --label ft_v1_holdout --compare train/eval_runs/app_holdout_baseline_20261107_205548/detail.jsonl` → **R@5 ≥ 50/55**，案例 ≥5/9、文档 ≥31/35、题库 11/11 |
| 4 | 205 全量 sanity | `python train/scripts/92_app_retrieval_eval.py --label ft_v1_205 --compare train/eval_runs/app_path_head_v1_20261107_194150/detail.jsonl` → ≥177（参考；训练集含其中 150 题，防大幅回退） |
| 5 | 阈值 | 分数分布位移检查；`MIN_VECTOR_SCORE` 不盲目改，拒答率不得上升 |
| 6 | 答案级 + 工程 | `95_answer_ab.py --prompt-variant v3`（主集 ≥41/45、held-out ≥19/24） + `pytest` 全绿 + `ci_gate.py` 3 PASS |
| 7 | 回滚演练 | 指回旧模型重建，holdout 复现 **47/55** |

**基线（旧模型 + 旧库）**：holdout 55 = **47/55（0.8545）**，案例 5/9、文档 31/35、
题库 11/11，MRR 0.7016（`train/eval_runs/app_holdout_baseline_20261107_205548`）；
205 全量 = 177/205。

⚠️ **口径**：训练对含 150 条「非 holdout 的评测题」——这是既有数据，holdout 55
已把这 150 题排除在外。**门禁只看 holdout 55**；205 全量会因训练集重叠而虚高，
只作 sanity。
