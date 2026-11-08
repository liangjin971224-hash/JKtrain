#!/usr/bin/env bash
# 在魔搭 PAI-DSW 实例内执行一次：建持久化虚拟环境、装 LLaMA-Factory、自检 4bit 可用性。
#
# 两条截然不同的路：
#   CUDA ：优先保留镜像自带 torch（TORCH_MODE=system，装得快），立刻自检 bitsandbytes；
#          自检不过再加 TORCH_MODE=fresh 重装一份 LF 指定的稳定 torch。
#   ROCm ：**只能** TORCH_MODE=system。让 pip 装一份 CUDA 版 torch 会直接废掉 ROCm 轮子。
set -euo pipefail

WORKSPACE=${WORKSPACE:-/mnt/workspace}
export MODELSCOPE_CACHE="$WORKSPACE/models"
LOG="$WORKSPACE/logs"
mkdir -p "$WORKSPACE"/{models,data,output,logs}
STAMP=$(date -Iseconds)
echo "[$STAMP] env setup start" | tee -a "$LOG/session.log"

if command -v rocm-smi >/dev/null 2>&1; then
  VENDOR=amd; GPUINFO=$(rocm-smi --showproductname | head -5)
elif command -v nvidia-smi >/dev/null 2>&1; then
  VENDOR=cuda; GPUINFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv)
else
  echo "未检测到 GPU"; exit 1
fi
echo "--- 加速卡：$VENDOR ---"
echo "$GPUINFO" | tee "$LOG/gpu-info.txt"

if [ "$VENDOR" = "amd" ]; then
  TORCH_MODE=system
  echo "ROCm 环境强制 TORCH_MODE=system"
else
  TORCH_MODE=${TORCH_MODE:-system}
fi

VENV="$WORKSPACE/venv"
if [ ! -d "$VENV" ]; then
  if [ "$TORCH_MODE" = "system" ]; then
    python -m venv --system-site-packages "$VENV"
  else
    python -m venv "$VENV"
  fi
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip setuptools wheel

if [ "$TORCH_MODE" = "system" ]; then
  echo "--- dry-run：确认 pip 不动 torch ---"
  pip install --dry-run "llamafactory==0.9.5" > "$LOG/pip-dryrun.txt" 2>&1 || {
    echo "dry-run 失败，日志：$LOG/pip-dryrun.txt"; tail -40 "$LOG/pip-dryrun.txt"; exit 1;
  }
  grep -E "^Would install" "$LOG/pip-dryrun.txt" | tee -a "$LOG/session.log" || true
  if grep -E "^Would install" "$LOG/pip-dryrun.txt" | grep -qE "(^| )torch-[0-9]"; then
    cat <<'EOF' | tee -a "$LOG/session.log"

!!! dry-run 计划安装/替换 torch。
!!!   CUDA：改用 TORCH_MODE=fresh 重跑（会装一份 LF 指定的稳定 torch，约 2.5GB 下载）。
!!!   ROCm：停在这里，把 pip-dryrun.txt 带回来，不要现场拼依赖。
EOF
    exit 2
  fi
  pip install "llamafactory==0.9.5" matplotlib 2>&1 | tee "$LOG/pip-install.txt"
else
  echo "--- TORCH_MODE=fresh：由 LLaMA-Factory 自带依赖决定 torch 版本 ---"
  pip install "llamafactory[torch,metrics]==0.9.5" 2>&1 | tee "$LOG/pip-install.txt"
fi

if [ "$VENDOR" = "cuda" ]; then
  echo "--- bitsandbytes（24GB 卡上 QLoRA 的必需品）---"
  pip install bitsandbytes 2>&1 | tee -a "$LOG/pip-install.txt" || true
  if python - <<'PY'
import torch, bitsandbytes as bnb
lin = bnb.nn.Linear4bit(64, 64, quant_type="nf4", compute_dtype=torch.bfloat16).cuda()
y = lin(torch.randn(2, 64, dtype=torch.bfloat16, device="cuda"))
print("bnb4bit ok", tuple(y.shape))
PY
  then echo "ok" > "$LOG/bnb-status.txt"; else echo "fail" > "$LOG/bnb-status.txt"; fi
  echo "bitsandbytes 状态：$(cat "$LOG/bnb-status.txt" | tee -a "$LOG/session.log")"
fi

echo "--- 环境指纹（安装后）---"
{
  python - <<'PY'
import torch, transformers
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None), "cuda", torch.version.cuda)
print("device available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__)
PY
  llamafactory-cli version || echo "llamafactory-cli 不可用"
} 2>&1 | tee "$LOG/env-fingerprint-$(date +%Y%m%d-%H%M%S).txt"

pip freeze > "$LOG/pip-freeze-$(date +%Y%m%d-%H%M%S).txt"
echo "[$(date -Iseconds)] env setup done"
echo "下次开机只需： source $VENV/bin/activate"
