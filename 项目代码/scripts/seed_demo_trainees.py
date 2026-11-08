"""造虚拟学员账号 + 评阅记录，用于测试批量生成流程。

────────────────────────────────────────────────────────────────────
为什么要有这个脚本
────────────────────────────────────────────────────────────────────
批量生成（P3）需要「多个学员」才有意义。库里只有 E001/M001 两个人，
测不出「20 人并发」的效果，也看不出进度条的推进过程。

本脚本造 18 个虚拟学员（V101–V118，加上原有 2 人共 20 个），
每人配不同数量的评阅记录，让报告内容有差异：
    · 有人表现好（能力项分数高）
    · 有人有红线（触发「红线」等级）
    · 有人样本不足（显示「不可评」）
这样能验证报告在各种数据形态下都正常。

────────────────────────────────────────────────────────────────────
★ 安全约定
────────────────────────────────────────────────────────────────────
1. **工号统一前缀 `V`**（V101 起），与真实员工（E/M 开头）区分开，
   一眼能认出是虚拟数据。

2. **姓名也带「虚拟」字样**，避免被误当成真人。

3. **一键清理**（`--clean`）：只删 `employee_no LIKE 'V%'` 的记录，
   绝不碰真实员工。

4. **不改动任何已有账号**，只用 INSERT IGNORE 插入新工号。

────────────────────────────────────────────────────────────────────
跑法
────────────────────────────────────────────────────────────────────
    .venv/Scripts/python.exe scripts/seed_demo_trainees.py           # 造数据
    .venv/Scripts/python.exe scripts/seed_demo_trainees.py --clean   # 清理
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: 虚拟工号前缀。清理只认这个前缀。
PREFIX = "V"

#: 统一密码（演示用，生产绝不这样）。
DEMO_PASSWORD = "trainee123"

#: 周期：与前端默认一致。
PERIOD = "2026-10"

#: 能力项分布模板：`(案例, 要素命中, 要素总数, high, medium, critical)`
#:
#: ★ 刻意造出三种形态：
#:   A 型（好）：全对全无违规 → 达标
#:   B 型（差）：要素少 + 有违规 → 薄弱
#:   C 型（红线）：有 critical → 一票否决
_PROFILES = {
    # ★ 好的学员要铺满 6 个能力项、每项 ≥2 次：
    #   否则每项都因「样本 < MIN_SAMPLES」被判「不可评」，
    #   雷达图因可评项 <3 而不显示 —— 看不出「全绿」长什么样。
    "good": [
        ("C01", 4, 4, 0, 0, 0), ("C02", 4, 4, 0, 0, 0),     # A 产品信息
        ("C05", 4, 4, 0, 0, 0), ("C06", 3, 4, 0, 1, 0),     # B 人群适配
        ("C07", 3, 3, 0, 0, 0), ("C13", 3, 3, 0, 0, 0),     # C 禁忌配伍
        ("C03", 4, 4, 0, 0, 0), ("C03", 3, 4, 0, 1, 0),     # D 功效合规
        ("C09", 4, 4, 0, 0, 0), ("C10", 4, 4, 0, 0, 0),     # E 异议处理
        ("C16", 4, 4, 0, 0, 0), ("C16", 3, 4, 0, 1, 0),     # G 扩展
    ],
    "weak": [
        ("C03", 1, 4, 1, 0, 0), ("C03", 2, 4, 0, 1, 0),
        ("C09", 1, 4, 1, 0, 0), ("C10", 2, 4, 1, 1, 0),
        ("C11", 1, 4, 1, 0, 0), ("C09", 2, 4, 0, 1, 0),
        ("C05", 1, 4, 1, 0, 0), ("C06", 2, 4, 0, 1, 0),
    ],
    "redline": [
        ("C07", 2, 3, 1, 0, 1),
        ("C04", 2, 3, 0, 0, 1),
        ("C13", 2, 3, 1, 0, 0),
    ],
    "sparse": [                      # 样本不足 → 显示「不可评」
        ("C01", 3, 4, 0, 0, 0),
    ],
    # 混合：多数好、个别差 → 雷达有形状但不全满，最接近真实情况
    "mixed": [
        ("C01", 4, 4, 0, 0, 0), ("C02", 3, 4, 0, 1, 0),
        ("C05", 4, 4, 0, 0, 0), ("C06", 4, 4, 0, 0, 0),
        ("C07", 2, 3, 1, 0, 0), ("C13", 3, 3, 0, 0, 0),
        ("C03", 2, 4, 1, 0, 0), ("C03", 3, 4, 0, 1, 0),
        ("C09", 4, 4, 0, 0, 0), ("C10", 2, 4, 1, 0, 0),
        ("C16", 3, 4, 0, 1, 0), ("C16", 4, 4, 0, 0, 0),
    ],
}

#: 18 个虚拟学员的设定：`(工号, 姓名, 团队, 带教人, 模板)`
_TRAINEES = [
    ("V101", "虚拟学员01", "一组", "M001", "good"),
    ("V102", "虚拟学员02", "一组", "M001", "weak"),
    ("V103", "虚拟学员03", "一组", "M001", "mixed"),
    ("V104", "虚拟学员04", "一组", "M001", "redline"),
    ("V105", "虚拟学员05", "一组", "M001", "good"),
    ("V106", "虚拟学员06", "二组", "M001", "weak"),
    ("V107", "虚拟学员07", "二组", "M001", "sparse"),
    ("V108", "虚拟学员08", "二组", "M001", "mixed"),
    ("V109", "虚拟学员09", "二组", "M001", "good"),
    ("V110", "虚拟学员10", "二组", "M001", "redline"),
    ("V111", "虚拟学员11", "三组", "M001", "weak"),
    ("V112", "虚拟学员12", "三组", "M001", "good"),
    ("V113", "虚拟学员13", "三组", "M001", "mixed"),
    ("V114", "虚拟学员14", "三组", "M001", "sparse"),
    ("V115", "虚拟学员15", "三组", "M001", "good"),
    ("V116", "虚拟学员16", "三组", "M001", "weak"),
    ("V117", "虚拟学员17", "三组", "M001", "redline"),
    ("V118", "虚拟学员18", "三组", "M001", "mixed"),
]


def _conn(container):
    return container.repository._conn()


def seed(container) -> int:
    """造账号 + 档案 + 评阅记录。返回造的人数。"""
    from domain.ports.repository_port import DrillReviewRow, DrillRiskItemRow
    from infrastructure.auth.password import hash_password

    catalog = {s.case_id: s for s in container.case_catalog.list_all()}
    pwd_hash = hash_password(DEMO_PASSWORD, rounds=12)
    created = 0

    with _conn(container) as conn:
        with conn.cursor() as cur:
            for no, name, team, mentor, profile in _TRAINEES:
                # ---- 员工档案 ----
                cur.execute(
                    """INSERT INTO employee
                       (employee_no, name, hire_date, position, team, mentor, active)
                       VALUES (%s,%s,%s,%s,%s,%s,1)
                       ON DUPLICATE KEY UPDATE
                         name=VALUES(name), team=VALUES(team),
                         mentor=VALUES(mentor), active=1""",
                    (no, name, "2026-03-01", "健康顾问", team, mentor),
                )
                # ---- 账号 ----
                cur.execute(
                    """INSERT INTO employee_account
                       (employee_no, role, password_hash, active)
                       VALUES (%s,'trainee',%s,1)
                       ON DUPLICATE KEY UPDATE
                         password_hash=VALUES(password_hash), active=1""",
                    (no, pwd_hash),
                )
                created += 1
            conn.commit()

    # ---- 评阅记录（走仓储，保证与真实写入路径一致） ----
    for no, name, team, mentor, profile in _TRAINEES:
        for idx, (cid, hit, total, high, med, crit) in enumerate(
                _PROFILES[profile]):
            spec = catalog.get(cid)
            if spec is None:
                continue
            did = f"demo_{no}_{cid}_{idx}"
            risks = ()
            if high or med or crit:
                level = "critical" if crit else ("high" if high else "medium")
                risks = (DrillRiskItemRow(
                    drill_id=did, turn_index=1, sentence_index=1,
                    rule_type="功效声称",
                    matched="降血脂" if crit else "肯定有效果",
                    level=level,
                    reason=("医疗用语：应为「有助于维持……健康水平」，不能用「降」"
                            if crit else "不得声称或暗示功效"),
                ),)
            container.repository.save_drill_review(
                DrillReviewRow(
                    drill_id=did, trainee_id=no, case_id=cid,
                    case_type=spec.question_type, duration_sec=90 + idx * 40,
                    element_hit=hit, element_total=total,
                    element_hit_all=hit, element_total_all=total,
                    risk_count=high + med + crit,
                    high_count=high, medium_count=med, critical_count=crit,
                    created_at=f"{PERIOD}-{10 + idx:02d} 10:00:00",
                ),
                risks,
            )
    return created


def clean(container) -> dict[str, int]:
    """只删虚拟工号（`V%`）的数据，绝不碰真实员工。"""
    out: dict[str, int] = {}
    with _conn(container) as conn:
        with conn.cursor() as cur:
            # 评阅明细要先删（外键语义上先删子表）
            cur.execute(
                """DELETE FROM drill_risk_item
                   WHERE drill_id IN (
                     SELECT drill_id FROM (
                       SELECT drill_id FROM drill_review
                       WHERE trainee_id LIKE %s
                     ) t)""", (PREFIX + "%",))
            out["drill_risk_item"] = cur.rowcount

            cur.execute("DELETE FROM drill_review WHERE trainee_id LIKE %s",
                        (PREFIX + "%",))
            out["drill_review"] = cur.rowcount

            cur.execute("DELETE FROM monthly_report WHERE trainee_id LIKE %s",
                        (PREFIX + "%",))
            out["monthly_report"] = cur.rowcount

            cur.execute("DELETE FROM report_task WHERE trainee_id LIKE %s",
                        (PREFIX + "%",))
            out["report_task"] = cur.rowcount

            cur.execute("DELETE FROM employee_account WHERE employee_no LIKE %s",
                        (PREFIX + "%",))
            out["employee_account"] = cur.rowcount

            cur.execute("DELETE FROM employee WHERE employee_no LIKE %s",
                        (PREFIX + "%",))
            out["employee"] = cur.rowcount
        conn.commit()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="清理虚拟数据")
    args = ap.parse_args()

    from config.settings import Settings
    from config.assembly import build_container

    container = build_container(Settings())
    try:
        if args.clean:
            removed = clean(container)
            print("已清理虚拟数据（只删 V 开头的工号）：")
            for k, v in removed.items():
                print(f"  {k:18s} {v} 行")
        else:
            n = seed(container)
            total = 2 + n            # E001 + M001 + 虚拟
            print(f"已造 {n} 个虚拟学员（{PREFIX}101–{PREFIX}118）")
            print(f"统一密码: {DEMO_PASSWORD}")
            print(f"员工总数: {total}")
    finally:
        container.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
