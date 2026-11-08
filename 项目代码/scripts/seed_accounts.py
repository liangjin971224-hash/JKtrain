"""初始化演示账号（P8）。

用法：
    .venv\\Scripts\\python.exe scripts\\seed_accounts.py
    .venv\\Scripts\\python.exe scripts\\seed_accounts.py --reset

★ 为什么需要这个脚本
────────────────────────────────────────────────────────────────────
P8 之前系统没有登录，所有接口裸奔。加了鉴权后，
如果没有账号就一个人都进不来 —— 连自己都锁在外面。

这个脚本建两个演示账号，便于验收：

    工号        密码          角色
    M001       admin123      主管（全部功能）
    E001       trainee123    学员（只能问答/演练/写日报）

★ 生产环境必须改密码，或删掉这两个账号。
  启动自检会提示「检测到演示账号」。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.assembly import build_container          # noqa: E402
from config.settings import Settings                  # noqa: E402
from domain.value_objects.account import EmployeeAccount  # noqa: E402
from domain.value_objects.employee import EmployeeProfile  # noqa: E402
from infrastructure.auth.password import hash_password     # noqa: E402

#: 演示账号（工号, 密码, 角色, 姓名）
DEMO_ACCOUNTS = [
    ("M001", "admin123", "manager", "李主管"),
    ("E001", "trainee123", "trainee", "张三"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化演示账号")
    parser.add_argument("--reset", action="store_true",
                        help="重置已有账号的密码")
    args = parser.parse_args()

    settings = Settings()
    container = build_container(settings)

    try:
        repo = container.repository
        if repo is None:
            print("[FAIL] 仓储未就绪，请先跑 scripts/init_db.py")
            return 1

        for emp_no, password, role, name in DEMO_ACCOUNTS:
            existing = repo.get_account(emp_no)

            if existing is not None and not args.reset:
                print(f"[跳过] {emp_no} 已存在（改密码请加 --reset）")
                continue

            # 档案（没有就建）
            if repo.get_employee(emp_no) is None:
                repo.save_employee(EmployeeProfile(
                    employee_no=emp_no,
                    name=name,
                    hire_date="2026-10-01",
                    position="健康顾问" if role == "trainee" else "门店主管",
                    team="一组",
                ))

            repo.save_account(EmployeeAccount(
                employee_no=emp_no,
                role=role,
                password_hash=hash_password(password),
                active=True,
            ))
            print(f"[OK] {emp_no} / {password} / {role}")

        print("\n完成。登录后请立即修改密码。")
    finally:
        container.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
