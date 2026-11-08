"""推理公用逻辑：加载模型、按线上模板拼提示、生成并记录时延与显存。"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from prompt_template import build_messages

_THINK_CLOSERS = ("<｜end▁of▁thinking｜>", "</think>")
_THINK_OPENERS = ("<｜begin▁of▁thinking｜>", "<think>")


def split_think(text: str) -> dict:
    """切开思考段与最终答案。

    找不到结束标记但存在起始标记时，说明思考被 max_new_tokens 截断，
    此时**不能**把整段当答案——那会让"空答案"检查失效。
    """
    best, marker = -1, ""
    for m in _THINK_CLOSERS:
        i = text.rfind(m)
        if i > best:
            best, marker = i, m
    if best == -1:
        if any(o in text for o in _THINK_OPENERS):
            return {"reasoning": text.strip(), "answer": "", "think_closed": False}
        return {"reasoning": "", "answer": text.strip(), "think_closed": True}
    return {
        "reasoning": text[:best].strip(),
        "answer": text[best + len(marker):].strip(),
        "think_closed": True,
    }


def thinking_is_open(prompt: str) -> bool:
    """提示里的思考块是否处于打开状态。

    空思考块 `<think>\\n\\n</think>` 是「关闭思考」的标准写法，
    不能因为含 `<think>` 就当作已启用思考。
    """
    last_open = max(prompt.rfind(o) for o in _THINK_OPENERS)
    last_close = max(prompt.rfind(c) for c in _THINK_CLOSERS)
    return last_open > last_close


def build_prompt_ids(
    tok, messages: list[dict], *, template_name: str | None = None,
    enable_thinking: bool | None = None,
) -> tuple[list[int], str]:
    """渲染提示并返回 (token ids, 渲染来源)。

    template_name 给了就走 LLaMA-Factory 自己的模板 —— 与训练同一条代码路径，
    训练与推理的输入格式因此必然一致。给不了就用 tokenizer 自带的 chat_template。
    """
    if template_name:
        try:
            from llamafactory.data.template import TEMPLATES
        except ImportError as exc:  # 最常见的坑：忘了激活 venv
            raise SystemExit(
                "当前 Python 里没有 llamafactory。\n"
                "先激活项目环境再跑：source /mnt/workspace/venv/bin/activate\n"
                "或者直接用 venv 的解释器：/mnt/workspace/venv/bin/python <脚本>\n"
                "（40_train.sh 会自己激活，所以训练不受影响。）"
            ) from exc

        if template_name not in TEMPLATES:
            raise SystemExit(
                f"LLaMA-Factory 没有模板 {template_name}；"
                f"先跑 25_template_probe.py 看注册表里有哪些。"
            )
        template = TEMPLATES[template_name]
        template.enable_thinking = enable_thinking
        template.fix_special_tokens(tok)
        template.fix_jinja_template(tok)

        # LF 的 system 走独立参数，混在 messages 里会抛 Unexpected role: system；
        # encode_oneturn 又把最后一条当响应，所以补一条空 assistant，只取 prompt 部分。
        system = None
        turns = [dict(m) for m in messages]
        if turns and turns[0].get("role") == "system":
            system = turns.pop(0)["content"]
        turns.append({"role": "assistant", "content": ""})

        prompt_ids, _ = template.encode_oneturn(tok, turns, system=system)
        prompt_ids = list(prompt_ids)
        tail = tok.decode(prompt_ids[-8:], skip_special_tokens=False)
        if "<｜Assistant｜>" not in tail:
            raise SystemExit(
                f"模板 {template_name} 渲染出的提示没有以助手标记结尾，"
                f"末 8 个 token 解出为 {tail!r}；不要拿它去推理。"
            )
        return prompt_ids, f"llamafactory:{template_name}"

    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        prompt = tok.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        prompt = tok.apply_chat_template(messages, **kwargs)
    return list(tok(prompt, add_special_tokens=False).input_ids), "tokenizer"


def load_model(model_path: str, *, adapter: str | None = None):
    model_dir = Path(model_path)
    if not (model_dir / "config.json").exists():
        raise SystemExit(
            f"{model_path} 不是有效的模型目录（缺 config.json）。\n"
            "用 10_download.py 末尾打印的「模型目录」作为 --model-path；"
            "没下过就先下载。"
        )
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    if adapter:
        from peft import PeftModel

        adapter_dir = Path(adapter)
        if not (adapter_dir / "adapter_config.json").exists():
            existing = sorted(p.name for p in adapter_dir.parent.glob(f"{adapter_dir.name}*")) \
                if adapter_dir.parent.exists() else []
            hint = f"同级目录里有：{existing}" if existing else "上级目录里没找到同名目录"
            raise SystemExit(
                f"Adapter 目录里没有 adapter_config.json：{adapter}\n"
                f"{hint}\n"
                "训练会按 save_total_limit 清理旧检查点，checkpoint-N 可能已经被删。"
            )
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def run_question(
    model, tok, item: dict, *, max_new_tokens: int, enable_thinking: bool | None = None,
    template_name: str | None = None,
) -> dict:
    messages = build_messages(
        item["question"],
        fact_hints=item.get("fact_hints"),
        facts=item.get("facts"),
        references=item.get("references"),
    )
    prompt_ids, render_source = build_prompt_ids(
        tok, messages, template_name=template_name, enable_thinking=enable_thinking
    )
    prompt = tok.decode(prompt_ids, skip_special_tokens=False)
    thinking_applied = thinking_is_open(prompt)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    enc = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        streamer=streamer,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    chunks: list[str] = []
    t_first = None
    for chunk in streamer:
        if t_first is None:
            t_first = time.perf_counter() - t0
        chunks.append(chunk)
    thread.join()
    t_total = time.perf_counter() - t0

    text = "".join(chunks)
    parts = split_think(text)
    n_new = len(tok(text, add_special_tokens=False).input_ids)
    return {
        "qid": item.get("qid"),
        "question": item["question"],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt": prompt,
        "render_source": render_source,
        "enable_thinking_requested": enable_thinking,
        "thinking_applied": thinking_applied,
        "output_raw": text,
        "reasoning": parts["reasoning"],
        "answer": parts["answer"],
        "empty_answer": not parts["answer"],
        "think_closed": parts["think_closed"],
        "truncated": n_new >= max_new_tokens - 1,
        "n_prompt_tokens": int(enc["input_ids"].shape[1]),
        "n_new_tokens_total": n_new,
        "n_answer_tokens": len(tok(parts["answer"], add_special_tokens=False).input_ids),
        "t_first_token_sec": t_first,
        "t_total_sec": round(t_total, 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available()
        else None,
    }
