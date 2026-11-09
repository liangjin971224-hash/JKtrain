"""多轮问答评测（B5）：跑 20 组对话，评最后一轮答案 + 改写/产品继承。

    python train/scripts/107_multi_turn_eval.py --label mt_v1

前置：106 号校验 PASS（annotation_wp_v4/outputs/multi_turn_rubrics.jsonl）。
★ 走真实链路：ConversationStore + QueryRewriter + AnswerQuestionUseCase（真检索/LLM/护栏）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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
WP = TRAIN_ROOT / "annotation_wp_v4"
RUNS = TRAIN_ROOT / "eval_runs"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _load_60():
    p = Path(__file__).with_name("60_score_eval.py")
    spec = importlib.util.spec_from_file_location("score60", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="多轮问答评测")
    ap.add_argument("--wp-dir", default=str(WP))
    ap.add_argument("--label", default="mt_v1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    wp = Path(args.wp_dir)
    groups = load_jsonl(wp / "groups.jsonl")
    rubrics = {r["group_id"]: r["expect"]
               for r in load_jsonl(wp / "outputs" / "multi_turn_rubrics.jsonl")}
    if not groups:
        raise SystemExit(f"[FAIL] 找不到 {wp}/groups.jsonl")
    if len(rubrics) < len(groups):
        raise SystemExit("[FAIL] rubric 不全，先跑 106 号校验")
    if args.limit:
        groups = groups[:args.limit]

    sys.path.insert(0, str(PROJECT_CODE))
    from application.services.conversation import ConversationStore
    from application.services.query_rewriter import QueryRewriter
    from application.use_cases.answer_question import AnswerQuestionUseCase
    from config.assembly import build_container
    from config.settings import Settings

    score60 = _load_60()
    s = Settings()
    c = build_container(s)
    uc = AnswerQuestionUseCase(
        graph=c.graph, vector=c.vector, keyword=c.keyword, sparse=c.sparse,
        sparse_collections=frozenset({"health_case_library"}),
        path_head_collections=frozenset({"health_training_docs"}),
        context_docs=s.context_docs, prompt_variant=s.prompt_variant,
        llm=c.llm, repository=None, rerank=c.rerank,
        rerank_candidate_k=s.rerank_candidate_k, llm_max_tokens=s.llm_max_tokens,
        answer_repair_enabled=s.answer_repair_enabled,
    )
    rewriter = QueryRewriter(graph=c.graph, llm=c.llm)
    store = ConversationStore()

    details: list[dict] = []
    for i, g in enumerate(groups, 1):
        sid = f"mt_{g['id']}"
        key = "eval:trainee"
        final = None
        for turn in g["turns"]:
            session = store.get_or_create(sid, key)
            rr = rewriter.rewrite(turn, session)
            r = uc.execute(
                rr.question, vector_collection="health_training_docs",
                product_ids=rr.product_ids,
                allowed_permissions=("public", "internal"), top_k=20)
            store.append(sid, __import__(
                "application.services.conversation", fromlist=["ConversationTurn"]
            ).ConversationTurn(
                question=turn, rewritten=rr.question, answer=r.answer,
                cited_chunk_ids=r.cited_chunk_ids,
                product_ids=r.product_ids, intent=r.meta.get("intent", ""),
                refused=r.refused,
            ))
            final = (rr, r)

        rr, r = final
        scored = score60.score_one({"expect": rubrics[g["id"]]}, r.answer)
        expected_pids = tuple(g.get("pids") or ())
        rewrite_ok = None
        if expected_pids:
            rewrite_ok = all(p in rr.product_ids for p in expected_pids)
        details.append({
            "group_id": g["id"],
            "turns": g["turns"],
            "final_rewritten": rr.question,
            "rewrite_source": rr.source,
            "product_ids": list(rr.product_ids),
            "expected_product_ids": list(expected_pids),
            "rewrite_ok": rewrite_ok,
            "passed": scored["passed"],
            "missed": scored["missed"],
            "forbidden_hit": scored["forbidden_hit"],
            "refused": r.refused,
            "answer": r.answer,
        })
        flag = "OK" if scored["passed"] else "FAIL"
        print(f"  {i:2d}/{len(groups)} {g['id']} {flag} | rewrite={rr.question!r} "
              f"pids={list(rr.product_ids)}")

    n = len(details)
    passed = sum(1 for d in details if d["passed"])
    with_pids = [d for d in details if d["rewrite_ok"] is not None]
    rewrite_ok = sum(1 for d in with_pids if d["rewrite_ok"])
    forbidden = sum(1 for d in details if d["forbidden_hit"])
    refused = sum(1 for d in details if d["refused"])
    summary = {
        "label": args.label,
        "n": n,
        "passed": passed,
        "pass_rate": round(passed / n, 4),
        "rewrite_checked": len(with_pids),
        "rewrite_ok": rewrite_ok,
        "forbidden": forbidden,
        "refused": refused,
        "gate": "PASS" if passed >= 16 else "FAIL",
        "generated_at": datetime.now(timezone.utc).astimezone()
        .isoformat(timespec="seconds"),
    }
    run_dir = RUNS / f"multi_turn_{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "detail.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for d in details:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 68)
    print(f"  多轮通过 {passed}/{n}（门禁 ≥16）｜改写/产品继承 "
          f"{rewrite_ok}/{len(with_pids)}｜禁用 {forbidden}｜拒答 {refused}")
    print(f"  判定 {summary['gate']}｜入档 {run_dir}")
    print("=" * 68)
    return 0 if summary["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
