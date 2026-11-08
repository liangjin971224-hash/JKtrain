#!/usr/bin/env bash
# LLaMA-Factory SFT。默认走 AMD/ROCm 的 BF16 纯 LoRA；CONFIG 可切到 CUDA 的 4bit QLoRA。
set -euo pipefail

WORKSPACE=${WORKSPACE:-/mnt/workspace}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG=${CONFIG:-$HERE/../configs/sft_qlora_cuda.yaml}
DATA_DIR=${DATA_DIR:-$WORKSPACE/data}
RUN=${RUN:-r1-8b-lora-r8-lr2e-5}
OUT="$WORKSPACE/output/$RUN"

source "$WORKSPACE/venv/bin/activate"
[ -f "$CONFIG" ] || { echo "配置不存在：$CONFIG"; exit 1; }
[ -f "$DATA_DIR/dataset_info.json" ] || { echo "缺少 $DATA_DIR/dataset_info.json（先跑 convert_to_alpaca.py）"; exit 1; }
[ -f "$DATA_DIR/train.jsonl" ] || { echo "缺少 $DATA_DIR/train.jsonl"; exit 1; }

N=$(grep -c . "$DATA_DIR/train.jsonl" || true)
echo "训练样本 $N 条"
if [ "$N" -lt 20 ] && [ "${ALLOW_SMALL:-0}" != "1" ]; then
  echo "样本少于 20 条，这组训练不具备解释力。"
  echo "只验证链路时用 ALLOW_SMALL=1 重跑，并把产物标记为链路验证而非实验组。"
  exit 1
fi

FREE_GB=$(df -BG "$WORKSPACE" | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "可用磁盘 ${FREE_GB}GB"
[ "$FREE_GB" -ge 40 ] || { echo "磁盘不足 40GB，检查点可能写不下"; exit 1; }

mkdir -p "$OUT"
sed "s|__OUTPUT_DIR__|$OUT|" "$CONFIG" > "$OUT/run.yaml"

if [ ! -s "$DATA_DIR/val.jsonl" ]; then
  echo "验证集为空：本次训练无法按验证表现挑检查点，只在结束时保留 Adapter。"
  sed -i '/^eval_dataset:/d; /^per_device_eval_batch_size:/d' "$OUT/run.yaml"
fi

if grep -q "quantization_bit" "$OUT/run.yaml" && [ "$(cat "$WORKSPACE/logs/bnb-status.txt" 2>/dev/null)" = "fail" ]; then
  echo "配置要求 4bit 量化，但环境自检记录 bitsandbytes 不可用。"
  echo "改用 CONFIG=configs/sft_lora_cuda_bf16.yaml 重跑，不要在这里试参数。"
  exit 1
fi

date -Iseconds | tee "$OUT/started_at.txt"
grep -E "model_name_or_path|template|quantization_bit|lora_rank|learning_rate|num_train_epochs|cutoff_len|per_device_train_batch_size" "$OUT/run.yaml" | tee "$OUT/effective-config.txt"

# 后台跑：断掉浏览器连接也不中断。日志同时在 logs/ 留一份
nohup llamafactory-cli train "$OUT/run.yaml" > "$OUT/train.log" 2>&1 &
PID=$!
echo "训练已在后台启动，pid=$PID，日志：$OUT/train.log"
echo "tail -f $OUT/train.log"

wait "$PID"
RC=$?
date -Iseconds | tee "$OUT/finished_at.txt"
echo "退出码 $RC"
[ "$RC" -eq 0 ] || { echo "训练失败，先看 train.log 尾部"; tail -40 "$OUT/train.log"; exit "$RC"; }
echo "Adapter/检查点：$OUT"
