"""对比「LLaMA-Factory 训练用的模板」与「tokenizer 自带 chat_template」的渲染结果。

训练走 LLaMA-Factory 代码里的模板，推理若走 tokenizer 自带模板，两边输入格式就不是一回事。
本脚本把两种渲染并排打出来，必要时人工判断哪个与预训练分布一致。

    python scripts/25_template_probe.py \
        --model-path /mnt/workspace/models/R1-0528-Qwen3-8B \
        --template deepseekr1
"""

from __future__ import annotations

import argparse
import json
import sys

from transformers import AutoTokenizer

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

SYSTEM = "你是一名保健品门店健康顾问，回答要简洁、只讲有依据的内容。"
USER = "鱼油那款批准的是啥功能来着？"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--template", default="deepseekr1")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
    ]

    print(f"tokenizer: {tok.__class__.__name__}")
    print("=" * 70)

    print("\n【A】tokenizer 自带 chat_template")
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(repr(text))
    except Exception as exc:
        print(f"渲染失败：{type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print(f"【B】LLaMA-Factory 模板 {args.template}")

    try:
        from llamafactory.data.template import TEMPLATES
    except Exception as exc:
        print(f"导入 llamafactory 失败：{type(exc).__name__}: {exc}")
        return

    names = sorted(n for n in TEMPLATES if "deepseek" in n or "nothink" in n)
    print("注册表里可用的相关模板：", names)
    if args.template not in TEMPLATES:
        print(f"模板 {args.template} 不存在，请从上面名单里选一个")
        return

    print("\n模型自带的思考标记 token id：")
    for tokstr in ("<｜begin▁of▁thinking｜>", "<｜end▁of▁thinking｜>"):
        tid = tok.convert_tokens_to_ids(tokstr)
        print(f"  {tokstr} -> {tid}")
    print(f"  tokenizer 有 chat_template: {bool(tok.chat_template)}")

    for name in names:
        template = TEMPLATES[name]
        print("\n" + "-" * 70)
        print(f"{name} | class={type(template).__name__} "
              f"| enable_thinking={template.enable_thinking} "
              f"| thought_words={template.thought_words} "
              f"| default_system={template.default_system!r}")
        for flag in (False, True):
            template.enable_thinking = flag
            try:
                template.fix_special_tokens(tok)
                template.fix_jinja_template(tok)
                prompt_ids, response_ids = template.encode_oneturn(tok, messages)
                print(f"  enable_thinking={flag}")
                print(f"    prompt  = {tok.decode(prompt_ids)!r}")
                print(f"    response= {tok.decode(response_ids)!r}")
            except Exception as exc:
                print(f"  enable_thinking={flag} 渲染失败：{type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print("判读要点")
    print("  1. 训练用 enable_thinking=false 的那一行；推理必须用同一行。")
    print("  2. prompt 里出现哪种思考起始标记，原模型预训练时用的就是哪种；")
    print("     LF 用 LF 的 thought_words，与 tokenizer 模板不一致时以预训练的那个为准。")
    print("  3. 两种渲染若在 user 段之后完全相同，差异只在 assistant 前缀，")
    print("     这一点差异足以让训练分布与推理分布对不上。")


if __name__ == "__main__":
    main()
