"""Agent 真实端到端评测（真实容器 / LLM / 检索 / 组卷 / 月报）。

    python train/scripts/111_agent_e2e_eval.py
    python train/scripts/111_agent_e2e_eval.py --only E01,E07

10 个任务：KB 4、图谱 2、演练 2、组卷 1、月报 1（主管）。
组卷用 persist=False，不写试卷；月报走真实队列，评测后任务留在 DB。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAIN_ROOT.parent
PROJECT_CODE = PROJECT_ROOT / "项目代码"
sys.path.insert(0, str(PROJECT_CODE))

from application.agents.tools import build_agent_registry          # noqa: E402
from application.agents.training_agent import TrainingAgent        # noqa: E402
from application.agents.types import AgentIdentity                 # noqa: E402
from application.use_cases.batch_report import BatchReportUseCase   # noqa: E402
from application.use_cases.build_paper import BuildPaperUseCase     # noqa: E402
from config.assembly import build_container                         # noqa: E402
from config.settings import Settings                                # noqa: E402
from domain.value_objects.role import Role                          # noqa: E402
from interfaces.http.deps import (                                  # noqa: E402
    _start_queue_consumer,
    _stop_queue_consumer,
)
from interfaces.http.routers.exam import load_question_bank         # noqa: E402
from interfaces.http.use_case_factory import build_answer_use_case  # noqa: E402

RUNS = TRAIN_ROOT / "eval_runs"
# 只放「肯定式、不会被否定/教学语境误伤」的短语；其余由产品护栏负责。
FORBIDDEN = ("可以治疗糖尿病", "根治高血压")
REPORT_PERIOD = "2026-09"
REPORT_TRAINEE = "E001"

TASKS = [
    {
        "id": "E01", "role": "trainee", "question": "鱼油有什么功能？",
        "tools": ["graph_facts", "kb_answer"], "must": ["鱼油"],
    },
    {
        "id": "E02", "role": "trainee", "question": "有没有适合三高的？",
        "tools": ["kb_answer"], "any": [["血压", "血糖", "血脂"]],
    },
    {
        "id": "E03", "role": "trainee", "question": "普通食品能说增强免疫力吗？",
        "tools": ["kb_answer"], "any": [["普通食品", "不得", "不能"]],
    },
    {
        "id": "E04", "role": "trainee",
        "question": "客户问「纯天然，没有副作用」这种话能不能说？",
        "tools": ["kb_answer"], "any": [["不能", "禁用", "绝对化"]],
    },
    {
        "id": "E05", "role": "trainee", "question": "P002 的禁忌是什么？",
        "tools": ["graph_facts", "kb_answer"], "must": ["辅酶Q10"],
    },
    {
        "id": "E06", "role": "trainee", "question": "孕妇能吃鱼油吗？",
        "tools": ["graph_facts", "kb_answer"],
        "any": [["孕妇", "不适宜"]],
    },
    {
        "id": "E07", "role": "trainee", "question": "我上周练得怎么样？",
        "tools": ["my_drill_reviews", "weakness_summary"],
        "must": ["演练", "4 次"],
    },
    {
        "id": "E08", "role": "trainee", "question": "哪里薄弱？推荐 3 个案例",
        "tools": ["my_drill_reviews", "weakness_summary"],
        "must": ["薄弱", "建议练习"],
    },
    {
        "id": "E09", "role": "manager", "question": "出 5 道题",
        "tools": ["build_exam"], "must": ["试卷", "S1-"],
        "check_exam": True,
    },
    {
        "id": "E10", "role": "manager",
        "question": f"生成 {REPORT_TRAINEE} {REPORT_PERIOD} 月报",
        "tools": ["submit_monthly_report"], "must": ["月度报告", REPORT_TRAINEE],
        "check_report": True,
    },
]


def _identity(role: str) -> AgentIdentity:
    return AgentIdentity(
        employee_no="M001" if role == "manager" else "E001",
        role=Role.MANAGER if role == "manager" else Role.TRAINEE,
    )


def _wait_report(repo, trainee: str, period: str, timeout: float = 90.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            last = repo.get_report_task(trainee, period)
        except Exception:
            last = None
        if last and last.get("status") in ("pending_review", "done"):
            return last
        time.sleep(2)
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔任务 ID，调试用")
    args = ap.parse_args()
    wanted = {x.strip() for x in args.only.split(",") if x.strip()}
    tasks = [t for t in TASKS if not wanted or t["id"] in wanted]

    settings = Settings()
    container = build_container(settings)
    if container.llm is None:
        print("[FAIL] LLM 未配置")
        return 1
    _start_queue_consumer(container)

    answer_uc = build_answer_use_case(container)
    exam_factory = lambda: BuildPaperUseCase(
        repository=container.repository,
        bank_loader=lambda stage: load_question_bank(container, stage),
    )
    report_factory = lambda: BatchReportUseCase(
        repository=container.repository,
        queue=container.task_queue,
        max_attempts=settings.queue_max_attempts,
    )
    registry = build_agent_registry(
        answer_use_case_factory=lambda: answer_uc,
        graph=container.graph,
        repository=container.repository,
        case_catalog=container.case_catalog,
        exam_use_case_factory=exam_factory,
        exam_bank_loader=lambda stage: load_question_bank(container, stage),
        exam_default_persist=False,
        report_use_case_factory=report_factory,
    )
    agent = TrainingAgent(registry)

    rows = []
    try:
        for task in tasks:
            started = time.perf_counter()
            result = agent.run(task["question"], _identity(task["role"]))
            elapsed = time.perf_counter() - started
            executed = [step.tool for step in result.steps]
            plan_ok = executed == task["tools"]
            include_ok = all(p in result.answer for p in task.get("must", []))
            include_ok = include_ok and all(
                any(p in result.answer for p in group)
                for group in task.get("any", []))
            forbidden = [p for p in FORBIDDEN if p in result.answer]
            refused_ok = not result.refused
            task_ok = True
            extra = ""
            if task.get("check_report"):
                report_task = _wait_report(
                    container.repository, REPORT_TRAINEE, REPORT_PERIOD)
                status = (report_task or {}).get("status", "missing")
                task_ok = status in ("pending_review", "done")
                extra = f"report_status={status}"
            if task.get("check_exam"):
                task_ok = "S1-" in result.answer and not result.refused
                extra = "exam_persist=False"
            passed = (
                plan_ok and include_ok and refused_ok
                and not forbidden and task_ok
            )
            rows.append({
                "id": task["id"],
                "role": task["role"],
                "question": task["question"],
                "expected_tools": task["tools"],
                "executed_tools": executed,
                "refused": result.refused,
                "refusal_reason": result.refusal_reason,
                "plan_ok": plan_ok,
                "include_ok": include_ok,
                "forbidden": forbidden,
                "task_ok": task_ok,
                "extra": extra,
                "elapsed_s": round(elapsed, 2),
                "passed": passed,
                "answer": result.answer[:800],
            })
            print(f"  [{task['id']}] {'PASS' if passed else 'FAIL'} "
                  f"tools={executed} {elapsed:.1f}s {extra}")
    finally:
        _stop_queue_consumer(container)
        try:
            container.close()
        except Exception:
            pass

    passed = sum(1 for r in rows if r["passed"])
    forbidden_count = sum(len(r["forbidden"]) for r in rows)
    latencies = sorted(r["elapsed_s"] for r in rows)
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0
    gate = (
        passed >= 8
        and forbidden_count == 0
        and all(not r["refused"] for r in rows)
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS / f"agent_e2e_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n": len(rows),
        "passed": passed,
        "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "forbidden_count": forbidden_count,
        "p95_elapsed_s": p95,
        "gate": "PASS" if gate else "FAIL",
        "gates": {
            "passed>=8": passed >= 8,
            "forbidden==0": forbidden_count == 0,
            "no_unexpected_refusal": all(not r["refused"] for r in rows),
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    with (out_dir / "detail.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Agent 真实 E2E 评测",
        "",
        f"> 通过 **{passed}/{len(rows)}**；禁用 {forbidden_count}；"
        f"P95 {p95:.1f}s；门禁 **{summary['gate']}**",
        "",
        "| 任务 | 期望工具 | 实际工具 | 拒答 | 结果 | 耗时 |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {','.join(row['expected_tools'])} | "
            f"{','.join(row['executed_tools'])} | "
            f"{'是' if row['refused'] else '否'} | "
            f"{'PASS' if row['passed'] else 'FAIL'} | {row['elapsed_s']}s |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"产物 -> {out_dir}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
