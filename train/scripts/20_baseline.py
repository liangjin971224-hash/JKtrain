"""原模型基线推理：同一批题、同一提示、同一生成设置，记录质量与速度。

在 GPU 实例上运行：
    python scripts/20_baseline.py \
        --model-path /mnt/workspace/models/R1-0528-Qwen3-8B \
        --questions data/baseline_questions.demo.jsonl \
        --out /mnt/workspace/output/baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

import infer_lib


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="默认关闭思考；模板不支持时脚本会自动回退并记录")
    ap.add_argument("--template", default=None,
                    help="LLaMA-Factory 模板名（如 deepseekr1）。给就用它渲染提示，"
                         "与训练同一条代码路径；不给则用 tokenizer 自带 chat_template")
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()

    items = load_jsonl(args.questions)
    model, tok = infer_lib.load_model(args.model_path)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB")
    print(f"torch {torch.__version__} / transformers {transformers.__version__}")
    print(f"待测题目 {len(items)} 条，max_new_tokens={args.max_new_tokens}")

    records: list[dict] = []
    for i, item in enumerate(items, 1):
        rec = infer_lib.run_question(
            model, tok, item,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            template_name=args.template,
        )
        rec["tag"] = args.tag
        records.append(rec)
        print(
            f"[{i}/{len(items)}] {rec['qid']} "
            f"prompt={rec['n_prompt_tokens']}tok "
            f"answer={rec['n_answer_tokens']}tok "
            f"first={rec['t_first_token_sec']:.2f}s total={rec['t_total_sec']:.2f}s "
            f"vram={rec['peak_vram_gb']}GB "
            f"{'TRUNCATED ' if rec['truncated'] else ''}"
            f"{'THINK_UNCLOSED ' if not rec['think_closed'] else ''}"
            f"{'EMPTY_ANSWER ' if rec['empty_answer'] else ''}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "tag": args.tag,
        "model_path": args.model_path,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy (do_sample=False, temperature=0)",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "questions": str(args.questions),
        "records": len(records),
        "template": args.template,
        "render_source": records[0]["render_source"] if records else None,
        "enable_thinking_requested": args.enable_thinking,
        "thinking_applied": sum(r["thinking_applied"] for r in records),
        "empty_answers": sum(r["empty_answer"] for r in records),
        "truncated": sum(r["truncated"] for r in records),
        "think_unclosed": sum(not r["think_closed"] for r in records),
        "mean_first_token_sec": round(
            sum(r["t_first_token_sec"] or 0 for r in records) / len(records), 3
        ),
        "mean_total_sec": round(sum(r["t_total_sec"] for r in records) / len(records), 3),
        "peak_vram_gb": max(r["peak_vram_gb"] or 0 for r in records),
    }
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
