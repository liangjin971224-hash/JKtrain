"""不占 GPU 的 tokenize 自检：模板、长度、答案段边界。

在 CPU 环境或有 tokenizer 的本机运行：
    python scripts/30_tokenize_check.py \
        --model-path /mnt/workspace/models/R1-0528-Qwen3-8B \
        --dataset /mnt/workspace/data/train.jsonl \
        --max-length 2048

只依赖 tokenizer，不需要权重。入参是 convert_to_alpaca.py 产出的 Alpaca 文件。
检查项：
  1. chat_template 存在（缺失说明 tokenizer 与模型版本不匹配）
  2. 提示（system+user）渲染结果确实是完整渲染的前缀 —— 否则答案段边界不可靠
  3. 每条样本的 token 长度，以及超过 max_length 的条数
  4. answer 段长度分布，抓「答案为空」的样本
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--show", type=int, default=1)
    args = ap.parse_args()

    model_dir = Path(args.model_path)
    if not model_dir.is_dir():
        raise SystemExit(
            f"模型目录不存在：{model_dir}\n"
            "先运行 10_download.py（CPU 侧加 --tokenizer-only），再用它末尾打印的"
            "「模型目录」作为 --model-path。"
        )
    if not (model_dir / "tokenizer_config.json").exists():
        subdirs = sorted(p.name for p in model_dir.iterdir() if p.is_dir())
        raise SystemExit(
            f"{model_dir} 里没有 tokenizer_config.json。\n"
            f"该目录下的子目录：{subdirs or '无'}\n"
            "如果 --model-path 指到了缓存父目录，请再往下进一层到具体模型目录。"
        )

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if not tok.chat_template:
        raise SystemExit("tokenizer 没有 chat_template，模板名与模型版本不匹配")

    # 往返自检：这一步能挡住"类名配错导致中文被吞"的隐性故障。
    # 本权重实测踩过：tokenizer_config.json 声明 LlamaTokenizerFast，中文编码结果为空数组，
    # 模型输出全是乱码，而长度统计看起来"正常"。
    probe = "康源牌鱼油软胶囊"
    probe_ids = tok(probe, add_special_tokens=False).input_ids
    probe_back = tok.decode(probe_ids)
    if not probe_ids or probe_back.strip() != probe:
        raise SystemExit(
            f"tokenizer 往返失败（{tok.__class__.__name__}）：\n"
            f"  「{probe}」→ {probe_ids} → {probe_back!r}\n"
            "常见原因：tokenizer_config.json 的 tokenizer_class 写错。"
            "把该字段改成 Qwen2TokenizerFast 后重试。"
        )

    rows = load_jsonl(args.dataset)
    if not rows:
        raise SystemExit("数据集为空")

    lengths: list[int] = []
    answer_lengths: list[int] = []
    over_limit: list[str] = []
    prefix_fail: list[str] = []
    empty_answers: list[str] = []

    for idx, row in enumerate(rows, 1):
        user_text = row["instruction"]
        if row.get("input"):
            # 与 LLaMA-Factory 官方口径一致：用户提示 = instruction + "\n" + input
            user_text = f"{user_text}\n{row['input']}"
        full_msgs = [
            {"role": "system", "content": row.get("system", "")},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": row["output"]},
        ]
        prompt_msgs = full_msgs[:-1]

        rendered_full = tok.apply_chat_template(full_msgs, tokenize=False)
        rendered_prompt = tok.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )

        if not rendered_full.startswith(rendered_prompt):
            prefix_fail.append(f"#{idx}")

        n_prompt = len(tok(rendered_prompt, add_special_tokens=False).input_ids)
        n_full = len(tok(rendered_full, add_special_tokens=False).input_ids)
        n_answer = n_full - n_prompt
        lengths.append(n_full)
        answer_lengths.append(n_answer)

        if n_answer <= 0:
            empty_answers.append(f"#{idx}")
        if n_full > args.max_length:
            over_limit.append(f"#{idx}({n_full})")

        if idx <= args.show:
            print(f"=== 样本 #{idx} ===")
            print(f"prompt={n_prompt}tok  total={n_full}tok  answer_labels={n_answer}tok")
            print("--- 渲染后的提示尾部 ---")
            print(rendered_prompt[-300:])
            print("--- 答案段 ---")
            # 前缀已验证一致，字符边界用 len(rendered_prompt)；不能用 n_prompt（那是 token 数）
            print(rendered_full[len(rendered_prompt):][:300])
            print()

    print(f"样本数: {len(rows)}")
    print(f"总长度: 中位 {statistics.median(lengths):.0f} / 最大 {max(lengths)} / 最小 {min(lengths)}")
    print(f"答案段: 中位 {statistics.median(answer_lengths):.0f} / 最大 {max(answer_lengths)}")
    print(f"超过 max_length={args.max_length}: {len(over_limit)} 条 {over_limit[:10]}")
    print(f"提示前缀不匹配(模板边界不可信): {len(prefix_fail)} 条 {prefix_fail[:10]}")
    print(f"答案段为空的样本: {len(empty_answers)} 条 {empty_answers[:10]}")

    if prefix_fail:
        raise SystemExit(
            "提示渲染不是完整渲染的前缀，说明模板对 system/user 的处理与预期不同，"
            "训练前必须换用框架自带的模板渲染，不要依赖本脚本的边界计算"
        )


if __name__ == "__main__":
    main()
