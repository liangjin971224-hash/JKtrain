"""训练产物 → 完整模型目录（sentence-transformers 可直接加载）。

全量微调（本轮用法）：
    python merge_lora.py --base /mnt/workspace/models/bge-m3 \
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
