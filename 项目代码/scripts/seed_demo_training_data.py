"""给演示环境补 4 条演练评阅 + 2 条日报（E001）。

    python scripts/seed_demo_training_data.py

★ 只写演示数据，使用固定 drill_id / 日期，可重复执行（幂等）。
★ 不创建员工/账号：默认 E001 已存在。
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from application.services.case_catalog import CaseCatalog          # noqa: E402
from config.settings import Settings                               # noqa: E402
from domain.ports.repository_port import (                         # noqa: E402
    DailySummaryRow,
    DrillReviewRow,
    DrillRiskItemRow,
)
from infrastructure.content.case_loader import load_cases_from_dir  # noqa: E402
from infrastructure.persistence.mysql_repository_adapter import (   # noqa: E402
    MySqlRepositoryAdapterFull,
)

TRAINEE = "E001"


def _pick_cases(catalog: CaseCatalog):
    cases = [c for c in catalog.list_all() if c.question_type]
    by_type: dict[str, object] = {}
    for case in cases:
        by_type.setdefault(case.question_type, case)
    picked = list(by_type.values())[:4]
    for case in cases:
        if len(picked) >= 4:
            break
        if case not in picked:
            picked.append(case)
    return picked


def main() -> int:
    settings = Settings()
    repo = MySqlRepositoryAdapterFull(database_url=settings.database_url)
    catalog = CaseCatalog(Path(settings.case_dir), load_cases_from_dir)

    cases = _pick_cases(catalog)
    if not cases:
        print("[FAIL] 案例目录为空，无法生成演示数据")
        return 1

    today = date.today()
    created = 0
    for index, case in enumerate(cases, 1):
        drill_id = f"demo-drill-{index:03d}"
        required = case.required_elements
        total = max(len(required), 1)
        critical = 1 if index == 1 else 0
        risk_items = ()
        if index == 1:
            risk_items = (
                DrillRiskItemRow(
                    drill_id=drill_id, turn_index=2, sentence_index=1,
                    rule_type="DR-03", matched="可以停药", level="critical",
                    reason="诱导停药：可能延误治疗"),
            )
        elif index == 2:
            risk_items = (
                DrillRiskItemRow(
                    drill_id=drill_id, turn_index=1, sentence_index=2,
                    rule_type="TS-01", matched="别人都说有效", level="high",
                    reason="以他人经验为效果背书"),
            )
        review = DrillReviewRow(
            drill_id=drill_id,
            trainee_id=TRAINEE,
            case_id=case.case_id,
            case_type=case.question_type,
            duration_sec=120 + index * 30,
            element_hit=max(total - (index % 3), 0),
            element_total=total,
            element_hit_all=max(total - (index % 3), 0),
            element_total_all=total,
            risk_count=len(risk_items),
            high_count=1 if index == 2 else 0,
            medium_count=0,
            critical_count=critical,
            created_at=(today - timedelta(days=index)).strftime(
                "%Y-%m-%d %H:%M:%S"),
            risk_items=risk_items,
        )
        repo.save_drill_review(review, risk_items)
        created += 1
        print(f"  drill {drill_id}: {case.case_id} {case.title} "
              f"hit={review.element_hit}/{total} critical={critical}")

    summaries = [
        DailySummaryRow(
            trainee_no=TRAINEE,
            summary_date=(today - timedelta(days=1)).isoformat(),
            content="练习了禁忌与停药场景，发现自己容易把「不建议」说成「不能」，"
                    "下次先陈述依据再给结论。",
            review_note="注意不要替客户判断。",
            status="submitted",
            submitted_at=(today - timedelta(days=1)).strftime(
                "%Y-%m-%d %H:%M:%S"),
        ),
        DailySummaryRow(
            trainee_no=TRAINEE,
            summary_date=today.isoformat(),
            content="复习了鱼油的批准功能和不适宜人群，准备再练一次价格异议。",
            status="draft",
        ),
    ]
    for row in summaries:
        repo.save_daily_summary(row)
        print(f"  daily {row.summary_date}: {row.status}")

    print(f"[OK] 演示数据完成：{created} 条演练评阅 + {len(summaries)} 条日报")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
