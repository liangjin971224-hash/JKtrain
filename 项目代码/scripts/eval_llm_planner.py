"""LLM 多跳规划：50+ 题评测。

★ 与单测的分工：
  单测证明「校验层拦得住非法输出」「触发判定逻辑对不对」；
  本脚本证明「在真实问法上，LLM 的拆解是否真的比规则好」。

用法::

    .venv/Scripts/python.exe scripts/eval_llm_planner.py           # 规则 + LLM
    .venv/Scripts/python.exe scripts/eval_llm_planner.py --dry     # 只看规则与触发率
    .venv/Scripts/python.exe scripts/eval_llm_planner.py --verbose # 逐题打印

输出四块：
  1. 触发率 vs 期望（含**误触发**与**漏触发**两个方向的错）
  2. 采纳率（LLM 输出通过校验的比例）
  3. 结果变化分布
  4. 人工复核清单（只看真的变了的题）
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from planner_cases import all_cases  # noqa: E402


def _fmt(steps) -> str:
    if not steps:
        return "（空）"
    parts = []
    for i, s in enumerate(steps, 1):
        tag = ""
        if s.depends_on_previous:
            tag += "[依赖]"
        if s.kind == "price_rank":
            tag += "[比价]"
        parts.append(f"{i}.{s.question[:24]}{tag}")
    return " | ".join(parts)


def _diff_kind(rule, llm) -> str:
    if len(rule) != len(llm):
        return "步数变化"
    for a, b in zip(rule, llm):
        if a.depends_on_previous != b.depends_on_previous:
            return "依赖变化"
        if a.kind != b.kind:
            return "kind变化"
    for a, b in zip(rule, llm):
        if a.question != b.question:
            return "措辞变化"
    return "完全相同"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只跑规则，不调 LLM")
    ap.add_argument("--verbose", action="store_true", help="逐题打印")
    args = ap.parse_args()

    from config.settings import Settings
    from domain.services.query_planner import dropped_fragments, plan_queries
    from application.services.query_plan_guard import plan_looks_suspicious

    cases = all_cases()

    planner = None
    if not args.dry:
        from config.assembly import build_container
        from application.services.llm_query_planner import LLMQueryPlanner

        container = build_container(Settings())
        if container.llm is None:
            print("LLM 不可用（检查 .env 的 QWEN_API_KEY）")
            return 1
        planner = LLMQueryPlanner(container.llm, max_calls=1)
        print(f"LLM: {getattr(container.llm, 'name', type(container.llm).__name__)}")

    print(f"评测集: {len(cases)} 题\n")

    rows: list[dict] = []
    for case in cases:
        q = case.question
        rule = plan_queries(q)
        dropped = dropped_fragments(q)
        trig, why = plan_looks_suspicious(q, rule, dropped_fragments=dropped)

        rec = {
            "case": case, "rule": rule, "triggered": trig,
            "trigger_reason": why, "llm": rule, "source": "rule",
            "adopt_fail": "", "change": "",
        }

        if planner and trig:
            planner.reset_budget()   # ★ 每题重置，模拟「每请求新建实例」
            try:
                out = planner.plan(q, rule, dropped_fragments=dropped)
                rec["llm"] = out
                rec["source"] = planner.last_decision.get("source", "?")
                if rec["source"] == "rule":
                    rec["adopt_fail"] = str(
                        planner.last_decision.get("validation_reason") or
                        planner.last_decision.get("reason") or "")
                else:
                    rec["change"] = _diff_kind(rule, out)
            except Exception as exc:                   # noqa: BLE001
                rec["adopt_fail"] = f"{type(exc).__name__}: {exc}"

        rows.append(rec)

        if args.verbose:
            print(f"\n【{case.expect}】{q}   — {case.note}")
            print(f"  规则: {_fmt(rule)}")
            print(f"  触发: {'是' if trig else '否'}"
                  + (f"（{why}）" if why else ""))
            if trig and planner:
                print(f"  LLM : {rec['source']}: {_fmt(rec['llm'])}")
                if rec["adopt_fail"]:
                    print(f"        ↳ 未采纳: {rec['adopt_fail'][:80]}")

    total = len(rows)

    # ================= 一、触发率 =================
    print("\n" + "=" * 74)
    print("一、触发率 vs 期望")
    print("=" * 74)

    matrix: Counter = Counter()
    for r in rows:
        act = "triggered" if r["triggered"] else "not_triggered"
        matrix[(r["case"].expect, act)] += 1

    print(f"{'期望':<12}{'实际触发':<12}{'实际未触发':<12}")
    for exp in ("rule_ok", "trigger", "either"):
        t = matrix.get((exp, "triggered"), 0)
        n = matrix.get((exp, "not_triggered"), 0)
        print(f"{exp:<12}{t:<12}{n:<12}")

    trig_total = sum(1 for r in rows if r["triggered"])
    print(f"\n总触发率: {trig_total}/{total} = {trig_total / total:.0%}")

    false_pos = [r for r in rows
                 if r["case"].expect == "rule_ok" and r["triggered"]]
    missed = [r for r in rows
              if r["case"].expect == "trigger" and not r["triggered"]]

    print(f"\n★ 误触发（期望不该调，却调了）: {len(false_pos)}")
    for r in false_pos:
        print(f"   - {r['case'].question}")
        print(f"     原因: {r['trigger_reason']}")

    print(f"\n★ 漏触发（期望该调，却没调）: {len(missed)}")
    for r in missed:
        print(f"   - {r['case'].question}")

    if planner:
        print("\n" + "=" * 74)
        print("二、采纳率")
        print("=" * 74)
        triggered = [r for r in rows if r["triggered"]]
        adopted = [r for r in triggered if r["source"] in ("llm", "cache")]
        print(f"触发 {len(triggered)} 次 → 采纳 {len(adopted)} 次 "
              f"({len(adopted) / max(len(triggered), 1):.0%})")
        if len(adopted) < len(triggered):
            print("\n未采纳的原因：")
            for r in triggered:
                if r["source"] not in ("llm", "cache"):
                    print(f"   - {r['case'].question[:38]}: "
                          f"{r['adopt_fail'][:70]}")

        print("\n" + "=" * 74)
        print("三、结果变化分布（仅采纳的）")
        print("=" * 74)
        changes = Counter(r["change"] for r in adopted)
        for k, v in changes.most_common():
            print(f"  {k:<12}{v}")

        print("\n" + "=" * 74)
        print("四、需要人工复核（结果真的变了的）")
        print("=" * 74)
        for r in adopted:
            if r["change"] in ("完全相同", ""):
                continue
            print(f"\n【{r['change']}】{r['case'].question}")
            print(f"  规则: {_fmt(r['rule'])}")
            print(f"  LLM : {_fmt(r['llm'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
