"""3B LoRA 合并：低内存策略。

★ 为什么需要专门的低内存策略
  本机总内存 13.7GB，其中 vmmemWSL 占 1.5GB（Docker 里跑着 Milvus，不能停），
  实际可用仅约 3.2GB。

  3B 模型 bf16 = 5.8GB，合并时 peft 会：
    1. 加载基座（5.8GB 内存）
    2. 加载 LoRA（120MB）
    3. merge_and_unload() 把 B@A*scaling 加到 W 上 —— **原地修改，不额外分配**
    4. save_pretrained 逐 tensor 写出（低峰值）

  所以峰值约等于「基座大小 + LoRA + 一点点」，即 ~6GB。
  3.2GB 不够 → 必须先腾内存，或用 mmap 方式加载。

★ 实测策略
  - low_cpu_mem_usage=True：从 safetensors 直接 mmap，避免中间 copy
  - 不用 .to("cuda")：合并全程在 CPU，GPU 只留着不动（省显存也给内存腾空间）
  - 分片保存：max_shard_size 设小一点，降低单次写盘的峰值
"""
import sys, os, time, gc
import torch
import torch.distributed.tensor   # peft 必需

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = r"D:\ai260803\xiangmudaima\pytorch\model\models\Qwen--Qwen2.5-3B-Instruct"
LORA = r"D:\项目\某健康公司培训系统\09_训练数据\lora_out_3b"
OUT = r"D:\项目\某健康公司培训系统\09_训练数据\merged_3b"


def rss_gb() -> float:
    """当前进程内存占用（GB）。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**3
    except Exception:
        return -1.0


def avail_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return -1.0


def main() -> None:
    print(f"起始: 进程 {rss_gb():.2f}GB | 系统可用 {avail_gb():.2f}GB")
    os.makedirs(OUT, exist_ok=True)

    # ---- 1. 只用 tokenizer（极轻）----
    print("\n[1/4] 加载 tokenizer …")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

    # ---- 2. 加载基座（CPU，bf16，mmap）----
    print("[2/4] 加载基座（CPU / bf16 / low_cpu_mem_usage）…")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,     # ★ 从 safetensors 直接映射，避免双份内存
        device_map=None,            # ★ 留在 CPU，不用 GPU
    )
    print(f"      完成 {time.time()-t0:.1f}s | 进程 {rss_gb():.2f}GB")

    # ---- 3. 挂 LoRA 并合并 ----
    print("[3/4] 挂载 LoRA 并合并 …")
    t0 = time.time()
    model = PeftModel.from_pretrained(model, LORA)
    print(f"      挂载完成 {time.time()-t0:.1f}s | 进程 {rss_gb():.2f}GB")

    t0 = time.time()
    merged = model.merge_and_unload()   # 原地加到 W，不额外分配大块
    print(f"      合并完成 {time.time()-t0:.1f}s | 进程 {rss_gb():.2f}GB")

    del model
    gc.collect()

    # ---- 4. 保存（分片，降低单次写盘峰值）----
    print("[4/4] 保存合并模型（分片 2GB）…")
    t0 = time.time()
    merged.save_pretrained(
        OUT,
        safe_serialization=True,
        max_shard_size="2GB",           # ★ 分片，避免一次序列化整个 5.8GB
    )
    tok.save_pretrained(OUT)
    print(f"      完成 {time.time()-t0:.1f}s | 进程 {rss_gb():.2f}GB")

    print(f"\n✅ 已保存到 {OUT}")


if __name__ == "__main__":
    main()
