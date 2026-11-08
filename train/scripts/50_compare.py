"""加载 Adapter 后跑同一批题，与原模型基线逐题对照。

    python scripts/50_compare.py \
        --model-path /mnt/workspace/models/R1-0528-Qwen3-8B \
        --adapter /mnt/workspace/output/r1-8b-lora-r8-lr2e-5 \
        --questions data/baseline_questions.demo.jsonl \
        --baseline /mnt/workspace/output/baseline.jsonl \
        --out /mnt/workspace/output/adapter.jsonl

Adapter 目录必须是训练产出的 checkpoint 目录（含 adapter_config.json）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import infer_lib


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="与基线保持同一取值，否则对照不成立")
    ap.add_argument("--template", default=None,
                    help="必须与生成基线时用的是同一个（给 = LLaMA-Factory 模板名，"
                         "不给 = tokenizer 自带 chat_template）")
    args = ap.parse_args()

    if not args.baseline.exists():
        raise SystemExit(
            f"基线文件不存在：{args.baseline}\n"
            "先用 20_baseline.py 生成基线，且模板参数要与本次完全相同：\n"
            f"  python train/scripts/20_baseline.py --template {args.template or '<不给>'} "
            "--out " + str(args.baseline)
        )
    if not args.questions.exists():
        raise SystemExit(f"题目文件不存在：{args.questions}")

    baseline_records = load_jsonl(args.baseline)
    base_by_qid = {r["qid"]: r for r in baseline_records}
    items = load_jsonl(args.questions)
    model, tok = infer_lib.load_model(args.model_path, adapter=args.adapter)

    expected_source = f"llamafactory:{args.template}" if args.template else "tokenizer"
    base_sources = {r.get("render_source") for r in baseline_records if "render_source" in r}
    if not base_sources:
        print("警告：基线文件没有 render_source 字段（旧版脚本生成），"
              "无法确认两边渲染路径一致，对照结果仅供参考。")
    elif base_sources != {expected_source}:
        print(f"警告：基线用 {sorted(base_sources)} 渲染，本次用 {expected_source}，"
              "两边输入格式不同，对照不成立。")

    if args.out.exists():
        raise SystemExit(f"{args.out} 已存在，避免覆盖上次结果，请先移走")

    records: list[dict] = []
    for i, item in enumerate(items, 1):
        rec = infer_lib.run_question(
            model, tok, item,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            template_name=args.template,
        )
        rec["tag"] = "adapter"
        rec["adapter"] = args.adapter
        records.append(rec)
        print(
            f"[{i}/{len(items)}] {rec['qid']} answer={rec['n_answer_tokens']}tok "
            f"total={rec['t_total_sec']:.2f}s "
            f"{'EMPTY_ANSWER' if rec['empty_answer'] else ''}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n=== 基线 vs Adapter ===")
    for rec in records:
        base = base_by_qid.get(rec["qid"])
        if base is None:
            print(f"{rec['qid']}: 基线里没有同题记录，跳过对照")
            continue
        same = base["answer"] == rec["answer"]
        print(f"\n--- {rec['qid']} {rec['question']}")
        print(f"原模型({base['n_answer_tokens']}tok): {base['answer'][:160]}")
        print(f"Adapter({rec['n_answer_tokens']}tok): {rec['answer'][:160]}")
        print(f"完全相同: {same}")

    print("\n提醒：逐题人工判读才算对照结论；这段输出只负责把差异摆出来。")


if __name__ == "__main__":
    main()
