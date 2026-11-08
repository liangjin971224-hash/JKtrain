#!/usr/bin/env bash
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
python -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
    --model_name_or_path "$BASE" \
    --train_data data/train.jsonl \
    --output_dir "$OUT" \
    --train_group_size 4 \
    --query_max_len 128 \
    --passage_max_len 1024 \
    --temperature 0.02 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_steps 20 --logging_steps 5 \
    --bf16 true

# 取最后一个 checkpoint（FlagEmbedding 按 step 存目录）
CKPT=$(ls -td "$OUT"/checkpoint-* 2>/dev/null | head -1)
[ -n "$CKPT" ] || { echo "[FAIL] $OUT 下没有 checkpoint-*，训练没跑成功"; exit 1; }
echo "checkpoint = $CKPT"

python merge_lora.py --base "$BASE" --ckpt "$CKPT" --out ./merged
python quick_val_eval.py --model ./merged --data data/val.jsonl

tar czf bge-m3-ft-v1.tar.gz -C merged .
echo "DONE → $(pwd)/bge-m3-ft-v1.tar.gz"
