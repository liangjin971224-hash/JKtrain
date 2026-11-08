"""P2 端到端验证：真实 Redis checkpoint + 真实 MySQL + 真实 LLM。

★ 与单测的分工
  单测用 InMemorySaver + 假仓储，只能证明编排对；
  本脚本验证三件只有真环境才能确认的事：

    ① 图的 checkpoint 真的落在 **Redis**（而不是内存）
    ② **跨进程**恢复（进程 A 挂起 → 进程 B 审批）
    ③ 定稿真的写进了 **MySQL**，且中间态确实不在业务库

跑法::

    .venv/Scripts/python.exe scripts/verify_p2_graph.py --stage a   # 进程A：跑到挂起
    .venv/Scripts/python.exe scripts/verify_p2_graph.py --stage b   # 进程B：审批
    .venv/Scripts/python.exe scripts/verify_p2_graph.py             # 全流程（单进程）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: 跨进程验证用的固定 thread_id（两个进程要能对上）
FIXED_THREAD = "E001:2026-10:xproc001"
PERIOD = "2026-10"


def _build(container):
    """构建图（真实仓储 + 真实 LLM + Redis checkpointer）。"""
    from application.graph.monthly_report_graph import build_report_graph
    from infrastructure.graph.cache_checkpoint_saver import CacheCheckpointSaver

    settings = container.settings
    titles = {s.case_id: s.title for s in container.case_catalog.list_all()}

    docs: dict[str, str] = {}
    try:
        rows = container.milvus_client.query(
            collection_name=settings.coll_training, filter="",
            output_fields=["doc_id", "title"], limit=1000)
        for r in rows:
            docs.setdefault(r["doc_id"], r.get("title", ""))
    except Exception as exc:                       # noqa: BLE001
        print(f"（文档清单读取失败，建议将不提资料名）: {exc}")

    saver = CacheCheckpointSaver(container.cache, ttl_sec=7 * 86400)
    graph = build_report_graph(
        repository=container.repository,
        llm=container.llm,
        checkpointer=saver,
        case_titles=titles,
        doc_catalog=docs,
    )
    return graph, saver


def _ensure_data(repo) -> int:
    """确保有足够的评阅记录（不够就补）。"""
    rows = repo.list_drill_reviews(trainee_id="E001", with_items=True)
    if len(rows) >= 2:
        return len(rows)

    from domain.ports.repository_port import DrillReviewRow, DrillRiskItemRow

    seed = [
        ("p2_e1", "C10", "价格异议（price_objection）", 2, 4, 1, 0),
        ("p2_e2", "C09", "疗效质疑（efficacy_challenge）", 3, 4, 0, 0),
        ("p2_c1", "C07", "禁忌询问（contraindication）", 2, 3, 1, 0),
        ("p2_c2", "C13", "配伍询问（incompatibility）", 2, 3, 0, 1),
    ]
    for did, cid, ct, hit, total, high, crit in seed:
        items = ()
        if high or crit:
            items = (DrillRiskItemRow(
                drill_id=did, turn_index=1, sentence_index=1,
                rule_type="功效声称", matched="肯定有效果",
                level="critical" if crit else "high",
                reason="不得声称功效"),)
        repo.save_drill_review(
            DrillReviewRow(
                drill_id=did, trainee_id="E001", case_id=cid,
                case_type=ct, duration_sec=120,
                element_hit=hit, element_total=total,
                element_hit_all=hit, element_total_all=total,
                risk_count=high + crit, high_count=high,
                critical_count=crit, created_at="2026-10-15 10:00:00",
            ), items)
    return len(repo.list_drill_reviews(trainee_id="E001"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b", "all"], default="all")
    args = ap.parse_args()

    from config.settings import Settings
    from config.assembly import build_container

    settings = Settings()
    container = build_container(settings)
    if container.repository is None:
        print("仓储不可用")
        return 1

    print(f"缓存后端: {type(container.cache).__name__}")
    if type(container.cache).__name__ != "RedisCacheAdapter":
        print("⚠️  不是 Redis —— 跨进程验证的前提不成立")
        return 1

    from langgraph.types import Command

    graph, saver = _build(container)
    cfg = {"configurable": {"thread_id": FIXED_THREAD}}
    failures: list[str] = []

    # ================= 阶段 A：跑到挂起 =================
    if args.stage in ("a", "all"):
        n = _ensure_data(container.repository)
        print(f"\n评阅记录: {n} 条")

        # 清理上次残留
        container.repository._conn  # noqa: B018 确保连接可用
        try:
            with container.repository._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM monthly_report WHERE trainee_id=%s",
                        ("E001",))
                conn.commit()
        except Exception as exc:                   # noqa: BLE001
            print(f"（清理旧定稿失败，忽略）: {exc}")
        saver.delete_thread(FIXED_THREAD)

        r = graph.invoke(
            {"trainee_id": "E001", "period": PERIOD,
             "thread_id": FIXED_THREAD}, cfg)

        itf = r.get("__interrupt__") or []
        print(f"① 挂起: {bool(itf)}")
        if itf:
            v = itf[0].value
            print(f"   演练次数: {v.get('total_drills')}")
            print(f"   薄弱项: {v.get('weak')}")
            print(f"   建议前 60 字: {(v.get('suggestion') or '')[:60]}")
        else:
            failures.append("没有触发 interrupt")

        st = graph.get_state(cfg)
        print(f"② next: {st.next}")
        print(f"   下钻条数: {len(st.values.get('drilldown') or [])}")

        # ★ 验证 1：MySQL 里没有定稿
        row = container.repository.get_monthly_report("E001", PERIOD)
        print(f"③ 库中定稿: {'有 ← 错误！' if row else '无（正确，草稿不该入库）'}")
        if row is not None:
            failures.append("挂起阶段就写了定稿（草稿进了业务库）")

        # ★ 验证 2：Redis 里真的有权重
        try:
            keys = container.cache._client.scan_iter(
                match=f"{settings.redis_key_prefix}report:ckpt:{FIXED_THREAD}*",
                count=100)
            key_list = list(keys)
            print(f"④ Redis checkpoint 键数: {len(key_list)}")
            if key_list:
                print(f"   样例: {key_list[0].split('/')[-1][:70]}")
            if not key_list:
                failures.append("Redis 里没有 checkpoint")
        except Exception as exc:                   # noqa: BLE001
            print(f"   （扫描 Redis 失败）: {exc}")

    # ================= 阶段 B：审批 =================
    if args.stage in ("b", "all"):
        st = graph.get_state(cfg)
        print(f"\n⑤ 恢复前 next: {st.next}")
        if not st.next:
            failures.append("没有可恢复的挂起状态")

        # ★ 模拟「驳回 → 重新生成 → 同意」
        graph.invoke(Command(resume={"decision": "reject",
                                     "note": "建议太笼统，要说清怎么练"}),
                     cfg)
        st2 = graph.get_state(cfg)
        print(f"⑥ 驳回后 next: {st2.next}")
        print(f"   建议变了: {st2.values.get('suggestion_text', '')[:40]}")
        row = container.repository.get_monthly_report("E001", PERIOD)
        if row is not None:
            failures.append("驳回阶段写了定稿")

        graph.invoke(Command(resume={"decision": "approve",
                                     "reviewed_by": "M001",
                                     "note": "通过"}), cfg)

        row = container.repository.get_monthly_report("E001", PERIOD)
        print(f"⑦ 审批后定稿: {'有' if row else '无'}")
        if row is None:
            failures.append("审批后没有定稿")
        else:
            print(f"   status: {row.status}")
            print(f"   thread_id: {row.thread_id}")
            print(f"   演练: {row.total_drills} | 红线: {row.total_critical}")
            print(f"   薄弱: {row.weak_capabilities}")
            print(f"   复核人: {row.reviewed_by!r} | 意见: {row.review_note!r}")
            snap = row.report_json or {}
            print(f"   快照层数: capabilities={len(snap.get('capabilities') or [])} "
                  f"cases={len(snap.get('cases') or [])}")
            if row.status != "approved":
                failures.append(f"状态应为 approved，实际 {row.status}")
            if row.reviewed_by != "M001":
                failures.append("复核人没落库")
            if not snap.get("capabilities"):
                failures.append("快照缺能力项")
            print(f"\n   建议全文:\n   {(row.suggestion or '').replace(chr(10), chr(10) + '   ')}")

    container.close()

    print("\n" + "=" * 62)
    if failures:
        print("P2 验证失败:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("P2 验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
