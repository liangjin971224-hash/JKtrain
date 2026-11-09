"""答案级 A/B：参考资料条数 `context_docs = 5 / 8 / 10`（P1-b 验收器）。

    python train/scripts/95_answer_ab.py --label p1b_ctx_ab --variants 5,8,10

────────────────────────────────────────────────────────────────────
为什么需要它
────────────────────────────────────────────────────────────────────
`_build_prompt` 只放 `context_docs` 条参考资料。用检索评测的 gold_rank
可以直接算出「gold 是否在上下文里」：5 条 = 177、8 条 = 181、10 条 = 185。
但覆盖只是必要条件——多出来的段落可能是噪声，把答案带偏、触发护栏、
甚至让可答问题变成拒答。所以必须走**答案级** A/B，不能只看覆盖率。

题集与判分
    题集：`train/data/eval_questions_v3.jsonl`（45 题，全部文档库，
          带 must_include / any_of / must_not_include 机器可判要点）。
    判分：复用 `60_score_eval.score_one`，不另写一套口径。
    生成：走线上 `AnswerQuestionUseCase`（意图 + 三路召回 + 护栏 + 数字核实），
          `repository=None`（不写未命中日志），temperature=0（端口默认）。

门禁（见 docs/实施方案_P1-b_上下文扩容_2026-11-07.md §4.3）
    ① rubric 通过率不低于 5 条基线（下降 ≥2 题否决）
    ② 禁用表述命中数不增加
    ③ 拒答率不增加

★ 只读脚本：不写库、不改文档，只在 train/eval_runs/ 下落实验产物。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TRAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAIN_ROOT.parent
PROJECT_CODE = PROJECT_ROOT / "项目代码"
DATA_DIR = TRAIN_ROOT / "data"
RUNS_DIR = TRAIN_ROOT / "eval_runs"

DEFAULT_EVAL = DATA_DIR / "eval_questions_v3.jsonl"
DEFAULT_REMAP = DATA_DIR / "retrieval_eval_v1_remap.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_60():
    """复用答案判分口径（60 号的 normalize + score_one）。"""
    path = Path(__file__).with_name("60_score_eval.py")
    spec = importlib.util.spec_from_file_location("score60", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mcnemar_exact(b: int, c: int) -> float:
    """精确 McNemar（双侧二项）。b/c 是两种单向翻转数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _build_uc(container, settings, context_docs: int, prompt_variant: str):
    """按线上装配方式构造用例（与 ask.py 同参数）。"""
    from application.use_cases.answer_question import AnswerQuestionUseCase

    return AnswerQuestionUseCase(
        graph=container.graph,
        vector=container.vector,
        keyword=container.keyword,
        sparse=getattr(container, "sparse", None),
        sparse_collections=frozenset(
            c.strip() for c in settings.sparse_recall_collections.split(",")
            if c.strip()),
        min_sparse_score=settings.min_sparse_score,
        path_head_collections=frozenset(
            c.strip() for c in settings.path_head_collections.split(",")
            if c.strip()),
        context_docs=context_docs,
        prompt_variant=prompt_variant,
        llm=container.llm,
        repository=None,
        rerank=container.rerank,
        rerank_candidate_k=settings.rerank_candidate_k,
        answer_repair_enabled=settings.answer_repair_enabled,
    )


def run_variant(
    *,
    uc,
    items: list[dict],
    remap: dict[str, dict],
    score60,
    context_docs: int,
    use_references: bool = False,
) -> tuple[list[dict], list[dict]]:
    """跑一个 context_docs 变体，返回 (明细, 预测)。"""
    details: list[dict] = []
    preds: list[dict] = []

    for i, item in enumerate(items, 1):
        qid = item["qid"]
        meta_row = remap.get(qid) or {}
        collection = meta_row.get("collection") or "health_training_docs"
        perms = meta_row.get("allowed_permissions")
        # 没有 remap 的题（如独立 held-out 答案集）按生产默认权限检索，
        # 避免 allowed_permissions=None 把受限块也放进上下文。
        perms = tuple(perms) if perms else ("public", "internal")
        golds = set(meta_row.get("gold_chunk_ids") or ())

        # ★ 独立 held-out 集自带 references（证据正文）。开启后把证据作为
        #   `fact_hints` 注入，使 v2/v3 看到完全相同的材料——只比较 prompt。
        hints: list[str] | None = None
        if use_references:
            hints = []
            for ref in item.get("references") or []:
                body = (ref.get("body") or "").strip()
                if not body:
                    continue
                head = " ".join(
                    x for x in (ref.get("title"), ref.get("section")) if x)
                hints.append(f"{head}：{body}" if head else body)
            hints = hints or None

        t0 = time.perf_counter()
        try:
            result = uc.execute(
                item["question"],
                vector_collection=collection,
                allowed_permissions=perms,
                top_k=20,
                fact_hints=hints,
            )
            answer = result.answer or ""
            refused = result.refused
            refusal_reason = result.refusal_reason
            context_ids = list(result.cited_chunk_ids[:context_docs])
            elapsed = time.perf_counter() - t0
            meta = result.meta or {}
            finish_reason = meta.get("finish_reason", "")
            llm_attempts = meta.get("llm_attempts", 0)
            repair_used = bool(meta.get("repair_used", False))
            repair_signals = list(meta.get("repair_signals", []))
        except Exception as exc:            # noqa: BLE001 单题失败不能中断整批
            answer, refused = "", True
            refusal_reason = f"{type(exc).__name__}: {exc}"
            context_ids, elapsed = [], time.perf_counter() - t0
            finish_reason, llm_attempts = "error", 0
            repair_used, repair_signals = False, []

        scored = score60.score_one(item, answer)
        detail = {
            **scored,
            "qid": qid,
            "context_docs": context_docs,
            "refused": refused,
            "refusal_reason": refusal_reason,
            "finish_reason": finish_reason,
            "llm_attempts": llm_attempts,
            "repair_used": repair_used,
            "repair_signals": repair_signals,
            "context_ids": context_ids,
            "gold_in_context": bool(golds & set(context_ids)),
            "elapsed_s": round(elapsed, 3),
            "answer": answer,
        }
        details.append(detail)
        preds.append({"qid": qid, "answer": answer,
                      "refused": refused, "refusal_reason": refusal_reason})

        if i % 10 == 0:
            print(f"    [{context_docs} docs] {i}/{len(items)} ...")

    return details, preds


def summarize(details: list[dict]) -> dict:
    n = len(details)
    passed = sum(1 for d in details if d["passed"])
    forbidden = sum(1 for d in details if d["forbidden_hit"])
    refused = sum(1 for d in details if d["refused"])
    gold_ctx = sum(1 for d in details if d["gold_in_context"])
    chars = [d["answer_chars"] for d in details]
    times = [d["elapsed_s"] for d in details]
    missed_points = sum(len(d["missed"]) for d in details)
    return {
        "n": n,
        "passed": passed,
        "pass_rate": round(passed / n, 4) if n else 0.0,
        "missed_points": missed_points,
        "forbidden_questions": forbidden,
        "refused": refused,
        "refusal_rate": round(refused / n, 4) if n else 0.0,
        "repair_triggered": sum(1 for d in details if d.get("repair_signals")),
        "repair_used": sum(1 for d in details if d.get("repair_used")),
        "gold_in_context": gold_ctx,
        "gold_in_context_rate": round(gold_ctx / n, 4) if n else 0.0,
        "avg_answer_chars": round(statistics.mean(chars), 1) if chars else 0.0,
        "p50_elapsed_s": round(statistics.median(times), 2) if times else 0.0,
    }


def compare(base: list[dict], other: list[dict]) -> dict:
    """逐题配对：rubric pass 翻转 + 精确 McNemar。"""
    b_by = {d["qid"]: d for d in base}
    o_by = {d["qid"]: d for d in other}
    qids = [q for q in b_by if q in o_by]
    up = [q for q in qids if not b_by[q]["passed"] and o_by[q]["passed"]]
    down = [q for q in qids if b_by[q]["passed"] and not o_by[q]["passed"]]
    out = {
        "n_common": len(qids),
        "pass_up": len(up),
        "pass_down": len(down),
        "up_qids": up,
        "down_qids": down,
        "mcnemar_p": round(mcnemar_exact(len(down), len(up)), 4),
        "delta_passed": len(up) - len(down),
    }
    # 新增/消失的禁用命中逐题对账（合规硬门禁要能定位到题）
    out["new_forbidden_qids"] = [
        q for q in qids
        if not b_by[q]["forbidden_hit"] and o_by[q]["forbidden_hit"]]
    out["fixed_forbidden_qids"] = [
        q for q in qids
        if b_by[q]["forbidden_hit"] and not o_by[q]["forbidden_hit"]]
    return out


def gate(base: dict, other: dict, cmp: dict) -> dict:
    """三条硬门禁（见方案 §4.3）。"""
    checks = {
        "pass_not_drop_2": (other["passed"] - base["passed"]) >= -1,
        "forbidden_not_up": (
            other["forbidden_questions"] - base["forbidden_questions"]) <= 0,
        "refused_not_up": (other["refused"] - base["refused"]) <= 0,
    }
    return {"checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL",
            "delta_passed": cmp["delta_passed"]}


def main() -> int:
    ap = argparse.ArgumentParser(description="答案级 A/B：context_docs 变体")
    ap.add_argument("--label", default="p1b_ctx_ab")
    ap.add_argument("--variants", default="5,8,10",
                    help="逗号分隔，第一个是基线，必须含 5")
    ap.add_argument("--eval", default=None, help="题集（默认 eval_questions_v3）")
    ap.add_argument("--remap", default=None,
                    help="collection/gold 映射（默认 retrieval_eval_v1_remap）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试）")
    ap.add_argument("--prompt-variant", default=None,
                    help="系统提示词版本（默认取 settings.prompt_variant）")
    ap.add_argument("--compare-detail", default=None,
                    help="外部基线 detail.jsonl（如上一轮 P1-b 的 detail_5）")
    ap.add_argument("--use-references", action="store_true",
                    help="把题目的 references 正文作为 fact_hints 注入（独立 held-out 集用）")
    args = ap.parse_args()

    variants = [int(x) for x in args.variants.split(",") if x.strip()]
    if not variants or variants[0] != 5:
        raise SystemExit("[FAIL] variants 第一个必须是基线 5。")

    eval_path = Path(args.eval) if args.eval else DEFAULT_EVAL
    remap_path = Path(args.remap) if args.remap else DEFAULT_REMAP
    items = load_jsonl(eval_path)
    if args.limit:
        items = items[:args.limit]
    remap = {r["qid"]: r for r in load_jsonl(remap_path)}
    if not items:
        raise SystemExit("[FAIL] 题集为空")

    score60 = _load_60()

    sys.path.insert(0, str(PROJECT_CODE))
    from config.assembly import build_container
    from config.settings import Settings

    print("=" * 72)
    print(f"  答案级 A/B label={args.label}  variants={variants}")
    print(f"  题集: {eval_path.name}（{len(items)} 题）")
    print(f"  时间: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}")
    print("=" * 72)

    settings = Settings()
    t0 = time.perf_counter()
    container = build_container(settings)
    print(f"  容器就绪 {time.perf_counter() - t0:.1f}s："
          f"{container.ready_ports()}；pending={list(container.pending) or '无'}")
    if container.llm is None:
        raise SystemExit("[FAIL] LLM 未配置，无法做答案级 A/B")

    run_dir = RUNS_DIR / f"{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt_variant = args.prompt_variant or settings.prompt_variant
    print(f"  提示词版本: {prompt_variant}")

    results: dict[int, dict] = {}
    all_details: dict[int, list[dict]] = {}
    for variant in variants:
        print(f"\n--- context_docs={variant} ---")
        uc = _build_uc(container, settings, variant, prompt_variant)
        details, preds = run_variant(
            uc=uc, items=items, remap=remap, score60=score60,
            context_docs=variant, use_references=args.use_references)
        all_details[variant] = details
        results[variant] = summarize(details)

        pred_path = run_dir / f"pred_{variant}.jsonl"
        with pred_path.open("w", encoding="utf-8", newline="\n") as fh:
            for p in preds:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        detail_path = run_dir / f"detail_{variant}.jsonl"
        with detail_path.open("w", encoding="utf-8", newline="\n") as fh:
            for d in details:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("\n" + "=" * 72)
    print("  汇总")
    print("=" * 72)
    for variant in variants:
        r = results[variant]
        print(f"  {variant:2d} docs | 通过 {r['passed']:2d}/{r['n']} "
              f"({r['pass_rate']:.3f}) | 缺要点 {r['missed_points']:2d} | "
              f"禁用 {r['forbidden_questions']} | 拒答 {r['refused']} | "
              f"补全 {r['repair_used']}/{r['repair_triggered']} | "
              f"gold在上下文 {r['gold_in_context']}/{r['n']} | "
              f"平均 {r['avg_answer_chars']:.0f} 字 | P50 {r['p50_elapsed_s']}s")

    comparisons: dict[str, dict] = {}
    gates: dict[str, dict] = {}
    for variant in variants[1:]:
        cmp = compare(all_details[5], all_details[variant])
        comparisons[f"5_vs_{variant}"] = cmp
        gates[f"5_vs_{variant}"] = gate(results[5], results[variant], cmp)
        print(f"\n  5 vs {variant}: 通过 {cmp['pass_up']} 上 / {cmp['pass_down']} 下，"
              f"净 {cmp['delta_passed']:+d}，McNemar p={cmp['mcnemar_p']}")
        if cmp["new_forbidden_qids"]:
            print(f"    ⚠️ 新增禁用: {cmp['new_forbidden_qids']}")
        verdict = gates[f"5_vs_{variant}"]["verdict"]
        print(f"    门禁: {verdict} {gates[f'5_vs_{variant}']['checks']}")

    # ---- 对外部基线（上一轮 run 的 detail）做配对比较 ----
    external: dict | None = None
    if args.compare_detail:
        ext_path = Path(args.compare_detail)
        if ext_path.is_dir():
            ext_path = ext_path / "detail.jsonl"
        ext_rows = load_jsonl(ext_path)
        ext_sum = summarize(ext_rows)
        # 默认拿第一个变体（通常是 5 条）与外部基线比
        other = all_details[variants[0]]
        ext_cmp = compare(ext_rows, other)
        ext_gate = gate(ext_sum, results[variants[0]], ext_cmp)
        external = {"detail": str(ext_path), "summary": ext_sum,
                    "comparison": ext_cmp, "gate": ext_gate}
        print(f"\n  外部基线 {ext_path.name}: 通过 {ext_sum['passed']}/{ext_sum['n']}"
              f" → {results[variants[0]]['passed']}/{results[variants[0]]['n']}"
              f"（{ext_cmp['pass_up']} 上 / {ext_cmp['pass_down']} 下，"
              f"p={ext_cmp['mcnemar_p']}）")
        if ext_cmp["new_forbidden_qids"]:
            print(f"    ⚠️ 新增禁用: {ext_cmp['new_forbidden_qids']}")
        print(f"    门禁: {ext_gate['verdict']} {ext_gate['checks']}")

    summary = {
        "label": args.label,
        "eval_set": str(eval_path),
        "n_queries": len(items),
        "variants": variants,
        "prompt_variant": prompt_variant,
        "results": {str(k): v for k, v in results.items()},
        "comparisons": comparisons,
        "gates": gates,
        "external_baseline": external,
        "llm_ready": container.llm is not None,
        "ready_ports": container.ready_ports(),
        "generated_at": datetime.now(timezone.utc)
        .astimezone().isoformat(timespec="seconds"),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  [OK] 入档 {run_dir}")

    # 有门禁 PASS 就返回 0；全部 FAIL 返回 1，便于脚本化判定
    all_gates = list(gates.values())
    if external is not None:
        all_gates.append(external["gate"])
    # 单变体（只跑基线）没有比较门禁，正常入档即成功。
    if not all_gates:
        return 0
    return 0 if any(g["verdict"] == "PASS" for g in all_gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
