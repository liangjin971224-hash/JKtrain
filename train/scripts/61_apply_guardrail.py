"""把模型的裸输出过一遍线上护栏，再交给评分器。

    python scripts/61_apply_guardrail.py \
        --pred eval_runs/.../adapter_eval_v6.jsonl \
        --eval data/eval_questions_v3.jsonl \
        --out  eval_runs/.../adapter_eval_v6_guarded.jsonl

★ 为什么需要：
  `50_compare.py` 只跑模型，不跑护栏；而云端那版是「模型 + 数字护栏 + 禁用词护栏」。
  两边分母不同，直接比分数不成立。这个脚本把 adapter 的输出补齐成同一口径。

★ 与线上 `AnswerQuestionUseCase.execute` 的对齐程度：
  对齐：数字核实（strip_ungrounded_sentences）、禁用词检测、逐句标注后保留安全内容。
  未对齐：
    · 不做类别声称判定（category=None）——需要产品类别，评测题里没有可靠来源；
    · 不做「重试生成」（静态后处理没有第二次生成的机会）；
    · 拒答模板用简化文案，与线上 BLOCKED_TEMPLATE 措辞不同。
  所以这是「近似线上」，用于横向比较，不等于端到端通过率。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "项目代码"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_template import build_evidence  # noqa: E402

from domain.rules.forbidden_claims import check_forbidden_claims  # noqa: E402
from domain.services.output_guardrail import (  # noqa: E402
    OutputGuardrail,
    strip_ungrounded_sentences,
)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BLOCKED = "关于这个问题，我不能按您期待的方式作答。（命中禁用表述，已拦截）"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    items = {i["qid"]: i for i in load_jsonl(args.eval)}
    preds = load_jsonl(args.pred)
    guard = OutputGuardrail()

    stats = {"unchanged": 0, "number_stripped": 0, "blocked": 0, "partial": 0}
    warned = False
    out: list[dict] = []
    for p in preds:
        qid = p["qid"]
        raw = p.get("answer") or p.get("output") or ""
        item = items.get(qid)
        if item is None:
            out.append({**p, "answer": raw, "guard": "no_eval_item"})
            continue

        # ★ 证据必须用「模型当时真正看到的」那一份。
        #   50_compare / 20_baseline 会把渲染后的 prompt 存进记录里，优先用它。
        #   退回静态 references 只适用于「模型就是按这份 references 生成」的场景，
        #   用错会把合法数字误删（实测云端结果被误删 23 条）。
        evidence = p.get("prompt")
        if not evidence:
            evidence = build_evidence(
                fact_hints=item.get("fact_hints"),
                facts=item.get("facts"),
                references=item.get("references"),
            ) + "\n" + item.get("question", "")
            if not warned:
                print("⚠️ 预测文件没有 prompt 字段，改用评测集静态 references 作证据。"
                      "若该批预测来自检索链路（而非 50_compare），结果会误删合法数字。")
                warned = True

        cleaned, dropped = strip_ungrounded_sentences(raw, evidence)
        verdict = check_forbidden_claims(cleaned)

        if not cleaned.strip() and dropped:
            # 整段都在编数字 → 与线上一致，走拒答模板，不能返回空字符串
            final = BLOCKED
            tag = "blocked"
            stats["blocked"] += 1
        elif verdict.has_critical:
            ann = guard.annotate(cleaned, category=None)
            if getattr(ann, "has_safe_content", False) and (ann.safe_text or "").strip():
                final = ann.safe_text
                tag = "partial"
                stats["partial"] += 1
            else:
                final = BLOCKED
                tag = "blocked"
                stats["blocked"] += 1
        else:
            final = cleaned
            tag = "number_stripped" if dropped else "unchanged"
            stats[tag] += 1

        out.append({
            "qid": qid,
            "answer": final,
            "raw_answer": raw,
            "guard": tag,
            "dropped_numbers": list(dropped),
            "guard_rules": [v.rule_id for v in verdict.violations],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(stats, ensure_ascii=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
