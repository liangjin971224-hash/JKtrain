"""P1 端到端验证：真实 MySQL + 真实 LLM 生成月度报告。

★ 与单测的分工：
  单测用假仓储假 LLM，只能证明编排对；
  本脚本用**真库真模型**，才能确认：
    · 评阅结果真的能从库里读回来
    · 能力项归并在真实 case_type 上真的能用
    · 生成的建议真的过了护栏
    · 耗时与调用次数符合预期

跑法::

    .venv/Scripts/python.exe scripts/verify_p1_report.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from config.settings import Settings
    from config.assembly import build_container
    from application.use_cases.monthly_report import MonthlyReportUseCase

    settings = Settings()
    container = build_container(settings)
    repo = container.repository
    if repo is None:
        print("仓储不可用")
        return 1

    titles = {s.case_id: s.title for s in container.case_catalog.list_all()}
    # 真实文档清单（封闭词表，防止模型编造资料名）
    rows = container.milvus_client.query(
        collection_name=settings.coll_training, filter="",
        output_fields=["doc_id", "title"], limit=1000)
    docs: dict[str, str] = {}
    for r in rows:
        docs.setdefault(r["doc_id"], r.get("title", ""))
    print(f"文档清单: {len(docs)} 份")

    # ---- 检查已有数据 ----
    reviews = repo.list_drill_reviews(
        trainee_id="E001", with_items=True)
    print(f"库中 E001 的评阅记录: {len(reviews)} 条")

    if len(reviews) < 2:
        print("\n数据不足（P0 验证只写了 1 条），先补 3 条不同能力项的记录…")
        _seed(repo)
        reviews = repo.list_drill_reviews(trainee_id="E001", with_items=True)
        print(f"补完后: {len(reviews)} 条")

    for r in reviews:
        print(f"  {r.drill_id[:10]} case={r.case_id:4s} "
              f"type={r.case_type[:20]:22s} "
              f"要素 {r.element_hit}/{r.element_total} "
              f"违规 high={r.high_count} crit={r.critical_count}")

    # ---- 生成报告 ----
    uc = MonthlyReportUseCase(
        repository=repo, llm=container.llm,
        case_titles=titles, doc_catalog=docs)

    t0 = time.perf_counter()
    result = uc.execute("E001", period="2026-10")
    elapsed = time.perf_counter() - t0

    s = result.summary
    print("\n" + "=" * 64)
    print(f"报告生成完成，耗时 {elapsed:.2f}s")
    print("=" * 64)
    print(f"学员 {s['trainee_id']} | 周期 {s['period']}")
    print(f"演练 {s['overview']['total_drills']} 次 / "
          f"覆盖 {s['overview']['distinct_cases']} 个案例 / "
          f"总时长 {s['overview']['total_duration_sec']}s")
    print(f"红线违规 {s['overview']['total_critical']} 条")

    print("\n【能力项】")
    for c in s["capabilities"]:
        score = f"{c['score']} 分" if c["score"] is not None else "—"
        mark = " ←薄弱" if c["is_weak"] else ""
        print(f"  {c['name']:8s} {score:8s} {c['level_label']:6s} "
              f"({c['samples']}次) {c['note']}{mark}")

    print(f"\n【薄弱项】{s['weak_capabilities'] or '无'}")

    print("\n【案例明细】")
    for c in s["cases"]:
        print(f"  {c['case_id']}《{c['case_title'][:18]}》"
              f" {c['score']} 分（{c['score_note']}）")
        for r in c["risks"][:2]:
            print(f"     · {r['location']}「{r['matched'][:22]}」"
                  f" [{r['level']}]")

    print("\n【改进建议】")
    sug = s["suggestion"]
    if sug["skipped"]:
        print(f"  （跳过：{sug['reason'] or '无薄弱项'}）")
    print(f"  通过护栏: {sug['passed']} | 重写 {sug['retries']} 次")
    if sug["reason"]:
        print(f"  未通过原因: {sug['reason'][:80]}")
    print("  " + (sug["text"] or "（空）").replace("\n", "\n  "))

    # ---- 校验 ----
    failures: list[str] = []
    if s["overview"]["total_drills"] < 1:
        failures.append("演练次数为 0")
    if not s["capabilities"]:
        failures.append("没有任何能力项（归并失败）")
    for c in s["capabilities"]:
        if c["key"] == "UNKNOWN":
            failures.append("存在未分类能力项")
    if sug["text"] and not sug["passed"]:
        failures.append("建议未通过护栏（应在重写后仍然失败才允许）")
    # 报告里的分数不能写回 SummaryCard —— 这条靠代码结构保证，
    # 这里只做一次显式断言：报告对象不应携带 SummaryCard
    if hasattr(result, "card"):
        failures.append("报告对象不应携带 SummaryCard（避免分数回流）")

    print("\n" + "=" * 64)
    if failures:
        print("P1 验证失败:")
        for f in failures:
            print(f"  ✗ {f}")
        container.close()
        return 1
    print("P1 验证通过")
    container.close()
    return 0


def _seed(repo) -> None:
    """补几条不同能力项的历史记录（只为验证归并与评分）。"""
    from domain.ports.repository_port import (
        DrillReviewRow, DrillRiskItemRow,
    )

    rows = [
        # E 异议处理：两次，有 high 违规 → 平均分低
        ("seed_e1", "C10", "价格异议（price_objection）",
         3, 4, 1, 0, 0),
        ("seed_e2", "C09", "疗效质疑（efficacy_challenge）",
         4, 4, 0, 0, 0),
        # C 禁忌：两次
        ("seed_c1", "C07", "禁忌询问（contraindication）",
         2, 3, 1, 0, 0),
        ("seed_c2", "C13", "配伍询问（incompatibility）",
         3, 3, 0, 1, 0),
        # F 红线：一次（样本不足 → 不可评，但若带 critical 则红线）
        ("seed_f1", "C04", "替代药物询问（drug_replacement）",
         3, 3, 0, 0, 0),
        # A 产品信息：两次，表现好
        ("seed_a1", "C01", "产品信息询问（product_existence）",
         4, 4, 0, 0, 0),
        ("seed_a2", "C02", "产品信息询问（product_ingredient）",
         4, 4, 0, 0, 0),
    ]
    for did, cid, ctype, hit, total, high, med, crit in rows:
        review = DrillReviewRow(
            drill_id=did, trainee_id="E001", case_id=cid,
            case_type=ctype, duration_sec=120,
            element_hit=hit, element_total=total,
            element_hit_all=hit, element_total_all=total,
            risk_count=high + med + crit,
            high_count=high, medium_count=med, critical_count=crit,
            created_at="2026-10-15 10:00:00",
        )
        items = ()
        if high or med or crit:
            items = (DrillRiskItemRow(
                drill_id=did, turn_index=1, sentence_index=1,
                rule_type="功效声称", matched="肯定有效果",
                level="high" if high else ("critical" if crit else "medium"),
                reason="不得声称功效",
            ),)
        repo.save_drill_review(review, items)


if __name__ == "__main__":
    sys.exit(main())
