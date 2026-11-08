"""造「成功 / 失败」对比数据，用于验证批量队列的失败处理。

────────────────────────────────────────────────────────────────────
★ 造失败的方式：给它一个不可能成功的时间区间
────────────────────────────────────────────────────────────────────
MySQL 的 `DATETIME` 遇到 `2026-13-45` 会直接抛
`OperationalError: Incorrect DATETIME value`。

这比「把 LLM 关掉」或「改代码注入 raise」更接近真实：
    · 不碰业务代码（不需要临时 if 分支）
    · 是**真的异常**，走完整的 try/except → 重试 → 判死链路
    · 失败原因能在 `last_error` 里看到，便于验证前端展示

────────────────────────────────────────────────────────────────────
造出三组对比
────────────────────────────────────────────────────────────────────
    V201  正常数据 + 正常区间         → 成功（待复核）
    V202  正常数据 + **非法区间**     → 失败 → 重试 3 次 → 判死
    V203  正常数据 + 正常区间         → 成功（待复核）

这样同一批里既有成功也有失败，前端能同时看到两种状态。

────────────────────────────────────────────────────────────────────
跑法
────────────────────────────────────────────────────────────────────
    .venv/Scripts/python.exe scripts/seed_failure_demo.py
    .venv/Scripts/python.exe scripts/seed_failure_demo.py --clean
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREFIX = "V2"
PASSWORD = "trainee123"
PERIOD = "2026-10"

#: 非法区间 —— 用它投递必然失败（MySQL 会拒绝这个 DATETIME）。
BAD_SINCE = "2026-13-45"

#: `(工号, 姓名, 会不会失败)` —— 三组对比
_PEOPLE = [
    ("V201", "对比-正常甲", False),
    ("V202", "对比-必败乙", True),
    ("V203", "对比-正常丙", False),
]

#: 评阅记录模板（跟 V101 一样的「好」形态，保证有内容可算）
_PROFILE = [
    ("C01", 4, 4, 0), ("C02", 4, 4, 0),
    ("C05", 4, 4, 0), ("C06", 3, 4, 1),
    ("C07", 3, 3, 0), ("C13", 3, 3, 0),
    ("C03", 3, 4, 1), ("C03", 4, 4, 0),
    ("C09", 4, 4, 0), ("C10", 4, 4, 0),
]


def _conn(container):
    return container.repository._conn()


def seed(container) -> None:
    from domain.ports.repository_port import DrillReviewRow
    from infrastructure.auth.password import hash_password

    catalog = {s.case_id: s for s in container.case_catalog.list_all()}
    pwd = hash_password(PASSWORD, rounds=12)

    with _conn(container) as conn:
        with conn.cursor() as cur:
            for no, name, _fail in _PEOPLE:
                cur.execute(
                    """INSERT INTO employee
                       (employee_no, name, hire_date, position, team, mentor, active)
                       VALUES (%s,%s,'2026-04-01','健康顾问','对比组','M001',1)
                       ON DUPLICATE KEY UPDATE
                         name=VALUES(name), team='对比组', active=1""",
                    (no, name))
                cur.execute(
                    """INSERT INTO employee_account
                       (employee_no, role, password_hash, active)
                       VALUES (%s,'trainee',%s,1)
                       ON DUPLICATE KEY UPDATE
                         password_hash=VALUES(password_hash), active=1""",
                    (no, pwd))
            conn.commit()

    for no, name, _fail in _PEOPLE:
        for idx, (cid, hit, total, med) in enumerate(_PROFILE):
            spec = catalog.get(cid)
            if spec is None:
                continue
            container.repository.save_drill_review(
                DrillReviewRow(
                    drill_id=f"faildemo_{no}_{cid}_{idx}",
                    trainee_id=no, case_id=cid,
                    case_type=spec.question_type, duration_sec=100 + idx * 10,
                    element_hit=hit, element_total=total,
                    element_hit_all=hit, element_total_all=total,
                    risk_count=med, high_count=0, medium_count=med,
                    critical_count=0,
                    created_at=f"{PERIOD}-{10 + idx:02d} 09:00:00",
                ), ())


def clean(container) -> dict[str, int]:
    out: dict[str, int] = {}
    with _conn(container) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """DELETE FROM drill_risk_item WHERE drill_id IN (
                     SELECT drill_id FROM (
                       SELECT drill_id FROM drill_review
                       WHERE trainee_id LIKE %s) t)""", (PREFIX + "%",))
            out["drill_risk_item"] = cur.rowcount
            for tbl, col in [("drill_review", "trainee_id"),
                             ("monthly_report", "trainee_id"),
                             ("report_task", "trainee_id"),
                             ("employee_account", "employee_no"),
                             ("employee", "employee_no")]:
                cur.execute(
                    f"DELETE FROM {tbl} WHERE {col} LIKE %s", (PREFIX + "%",))
                out[tbl] = cur.rowcount
        conn.commit()
    return out


def publish(container) -> dict:
    """投递三个任务（含一个必然失败的），返回投递结果。"""
    from application.use_cases.batch_report import BatchReportUseCase

    uc = BatchReportUseCase(
        repository=container.repository, queue=container.task_queue,
        max_attempts=container.settings.queue_max_attempts)

    results = {}
    for no, name, will_fail in _PEOPLE:
        res = uc.enqueue_batch(
            period=PERIOD, trainee_ids=[no],
            since=BAD_SINCE if will_fail else "2026-10-01",
            until="" if will_fail else "2026-11-01",
        )
        results[no] = {
            "name": name, "will_fail": will_fail,
            "task_id": res.task_ids[0] if res.task_ids else "",
        }
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="清理对比数据")
    ap.add_argument("--publish", action="store_true",
                    help="造完数据后立刻投递任务（含 1 个必败）")
    args = ap.parse_args()

    from config.settings import Settings
    from config.assembly import build_container

    container = build_container(Settings())
    try:
        if args.clean:
            removed = clean(container)
            print("已清理对比数据（只删 V2 开头）：")
            for k, v in removed.items():
                print(f"  {k:18s} {v} 行")
            return 0

        seed(container)
        print(f"已造 {len(_PEOPLE)} 个对比账号（密码 {PASSWORD}）:")
        for no, name, will_fail in _PEOPLE:
            flag = "★ 必然失败" if will_fail else "正常"
            print(f"  {no}  {name}  {flag}")

        if args.publish:
            print(f"\n投递任务（失败的那个用非法区间 since={BAD_SINCE}）:")
            res = publish(container)
            for no, r in res.items():
                mark = "★" if r["will_fail"] else " "
                print(f" {mark} {no}  {r['name']}  task={r['task_id']}")
            print("\n请等服务端消费者处理（约 10–20 秒后查进度）")
    finally:
        container.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
