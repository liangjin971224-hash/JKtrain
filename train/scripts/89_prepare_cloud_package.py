"""打包云端训练自包含包：数据（FlagEmbedding 格式）+ 脚本 + 门禁。

    python scripts/89_prepare_cloud_package.py

产出 train/cloud_package/（整目录上传 /mnt/workspace 即可）：
    data/train.jsonl / data/val.jsonl   FlagEmbedding m3 finetune 格式
    merge_lora.py                       adapter → 完整模型目录（保持 ST 加载格式）
    quick_val_eval.py                   云端合并后先跑 sanity R@5（CPU 即可）
    run_cloud.sh                        一条命令跑完 训练→合并→sanity→打包
    README.md                           五道门禁清单（对应实施方案 2.4–2.8）

★ 数据格式说明：FlagEmbedding embedder finetune 接受
  {"query": str, "pos": [str...], "neg": [str...]}，与 86 号产出一致，
  这里只做 train/val 拆分与字段裁剪。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = TRAIN_ROOT / "data"
PKG = TRAIN_ROOT / "cloud_package"

RUN_SH = """#!/usr/bin/env bash
# 云端一键脚本：全量微调 → 导出 ST 目录 → sanity → 打包
# 用法：bash run_cloud.sh [BASE_MODEL_DIR] [OUTPUT_DIR]
set -e
BASE=${1:-/mnt/workspace/models/bge-m3}
OUT=${2:-/mnt/workspace/output/bge-m3-ft-v1}
cd "$(dirname "$0")"

# ★ 模块名是 encoder_only（下划线）。官方路径 finetune/embedder/encoder_only/m3，
#   写成 encoder-only 会在第一条命令就 ModuleNotFoundError（2026-11-07 实机踩到）。
# ★ 本版 FlagEmbedding 的 embedding 训练入口**不支持 LoRA**：没有 --finetuning_type /
#   --lora_rank / --target_modules；`AbsEmbedderDataArguments` 也没有 --eval_data，
#   且 --warmup_ratio 被拒。因此这里走**全量微调**，参数只用实测被接受的集合
#   （2026-11-07 A10 实例上验证）。
# ★ 2026-11-07 云端实机：单卡加 --negatives_cross_device 会报
#   "Distributed training has not been initialized for representation all gather"
#   （FlagEmbedding 1.4.2），必须去掉；单卡 in-batch 负例本来就在同一设备内。
# ★ 2026-11-07 调整：600 对里有 54 条正例 >512 字（案例库 42/80 超过 512，
#   最长 2110），512 会把案例块尾部截掉。改为 passage_max_len 1024；
#   同时 batch 4→2、accum 4→8，保持每步 token 数与原来相同（8×1024 = 16×512），
#   显存与训练时长基本不变。
python -m FlagEmbedding.finetune.embedder.encoder_only.m3 \\
    --model_name_or_path "$BASE" \\
    --train_data data/train.jsonl \\
    --output_dir "$OUT" \\
    --train_group_size 4 \\
    --query_max_len 128 \\
    --passage_max_len 1024 \\
    --temperature 0.02 \\
    --learning_rate 1e-5 \\
    --num_train_epochs 3 \\
    --per_device_train_batch_size 2 \\
    --gradient_accumulation_steps 8 \\
    --save_steps 20 --logging_steps 5 \\
    --bf16 true

# 取最后一个 checkpoint（FlagEmbedding 按 step 存目录）
CKPT=$(ls -td "$OUT"/checkpoint-* 2>/dev/null | head -1)
[ -n "$CKPT" ] || { echo "[FAIL] $OUT 下没有 checkpoint-*，训练没跑成功"; exit 1; }
echo "checkpoint = $CKPT"

python merge_lora.py --base "$BASE" --ckpt "$CKPT" --out ./merged
python quick_val_eval.py --model ./merged --data data/val.jsonl

tar czf bge-m3-ft-v1.tar.gz -C merged .
echo "DONE → $(pwd)/bge-m3-ft-v1.tar.gz"
"""

MERGE_PY = '''"""训练产物 → 完整模型目录（sentence-transformers 可直接加载）。

全量微调（本轮用法）：
    python merge_lora.py --base /mnt/workspace/models/bge-m3 \\
                         --ckpt .../checkpoint-33 --out ./merged

将来若改用 LoRA，额外给 --adapter 走 peft 合并：
    python merge_lora.py --base ... --ckpt .../checkpoint-XX --adapter ... --out ./merged

为什么不能直接用 checkpoint：FlagEmbedding 的 Trainer.save_model 存的是纯 HF 模型
（config.json + 权重 + tokenizer），没有 1_Pooling/ 和 modules.json，
SentenceTransformer 加载不了。这里借 base 的 ST 结构，把权重换进去再整体 save。
"""
import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="原始 bge-m3 目录（提供 ST 结构、pooling 与 tokenizer）")
    ap.add_argument("--ckpt", default=None,
                    help="训练输出的 checkpoint 目录（全量微调的权重来源）")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter 目录（可选；给了就在 --ckpt 权重上合并）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if not a.ckpt and not a.adapter:
        raise SystemExit("--ckpt 与 --adapter 至少给一个")

    from transformers import AutoModel

    st = SentenceTransformer(a.base, device="cpu")
    hf = AutoModel.from_pretrained(a.ckpt or a.base, trust_remote_code=True)
    if a.adapter:
        from peft import PeftModel
        hf = PeftModel.from_pretrained(hf, a.adapter).merge_and_unload()
        print(f"peft merged: {a.adapter}")

    # 把（合并后的）transformer 换进 ST 模型，保留 pooling/归一化模块与 tokenizer
    # ★ 必须赋给 `.model`，不能赋 `.auto_model`：后者是只读 property，
    #   赋值会静默失败，st.save() 仍然保存基座权重（2026-11-07 本地实测：
    #   合并后与基座逐权重相同、cos=1.0）。
    st._first_module().model = hf
    Path(a.out).mkdir(parents=True, exist_ok=True)
    st.save(a.out)

    dim = st.get_sentence_embedding_dimension()
    print(f"merged → {a.out}, dim={dim}")
    assert dim == 1024, f"维度门禁失败: {dim} != 1024"


if __name__ == "__main__":
    main()
'''

QUICK_EVAL_PY = '''"""云端 sanity 评测（非正式门禁）：val 集上 pos 是否排进 Top-5。

    python quick_val_eval.py --model ./merged --data data/val.jsonl
★ 参考水位：未训练的 bge-m3 本地实测 ≈0.78（val 集难负例密集，2026-11-07
  用 B=0 恒等合并干跑标定）。<0.50 才说明合并/数据大概率出了问题；
  0.50–0.90 之间不阻塞打包，正式结论以本地 81 号门禁为准。
"""
import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    st = SentenceTransformer(a.model, device="cpu")

    hit5, n = 0, 0
    for r in rows:
        cand_texts = list(r["pos"]) + list(r["neg"])
        gold_set = set(range(len(r["pos"])))
        qv = st.encode([r["query"]], normalize_embeddings=True)
        cv = st.encode(cand_texts, normalize_embeddings=True)
        order = np.argsort(-(qv @ cv.T)[0])
        n += 1
        if any(int(i) in gold_set for i in order[:5]):
            hit5 += 1

    r5 = hit5 / n
    flag = "OK" if r5 >= 0.50 else "疑似异常——检查合并与数据，暂缓打包"
    print(f"sanity R@5 = {r5:.4f}（{n} 条）[{flag}]，"
          f"参考水位 0.78（未训练），正式门禁在本地 81 号")


if __name__ == "__main__":
    main()
'''

README_MD = """# 云端训练包（BGE-M3 600 对，2026-11-07）

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
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="打包云端训练自包含包")
    parser.add_argument("--pairs", default="embed_pairs_v1.jsonl",
                        help="训练对文件名或绝对路径（默认 embed_pairs_v1.jsonl；"
                             "切 holdout 后按 150 条重打请用 embed_pairs_v1_train150.jsonl）")
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.is_absolute():
        pairs_path = DATA_DIR / pairs_path
    rows = [json.loads(l) for l in pairs_path.open(encoding="utf-8") if l.strip()]
    if not rows:
        raise SystemExit(f"[FAIL] {pairs_path} 为空，先跑 86 号。")

    (PKG / "data").mkdir(parents=True, exist_ok=True)
    n_train = n_val = 0
    with (PKG / "data" / "train.jsonl").open("w", encoding="utf-8") as ft, \
         (PKG / "data" / "val.jsonl").open("w", encoding="utf-8") as fv:
        for r in rows:
            slim = {"query": r["query"], "pos": r["pos"], "neg": r["neg"]}
            line = json.dumps(slim, ensure_ascii=False) + "\n"
            if r.get("split") == "val":
                fv.write(line)
                n_val += 1
            else:
                ft.write(line)
                n_train += 1

    (PKG / "run_cloud.sh").write_text(RUN_SH, encoding="utf-8", newline="\n")
    (PKG / "merge_lora.py").write_text(MERGE_PY, encoding="utf-8")
    (PKG / "quick_val_eval.py").write_text(QUICK_EVAL_PY, encoding="utf-8")
    (PKG / "README.md").write_text(README_MD, encoding="utf-8")
    # 训练配置原样随包（云端对照用）
    shutil.copy(TRAIN_ROOT / "configs" / "embed_lora_v1.yaml", PKG / "embed_lora_v1.yaml")

    print("=" * 68)
    print(f"  云端训练包 → {PKG}")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 68)
    print(f"  [OK] data/train.jsonl {n_train} 条 / data/val.jsonl {n_val} 条")
    print(f"  [OK] run_cloud.sh / merge_lora.py / quick_val_eval.py / README.md")
    print(f"  下一步：整目录上传 /mnt/workspace/cloud_package → bash run_cloud.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
