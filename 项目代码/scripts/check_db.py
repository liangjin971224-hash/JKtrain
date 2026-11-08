"""检查 MySQL 库表状态（只读）。

凭据从 .env 的 DATABASE_URL 读，不写在命令行里。

用法：
    .venv\\Scripts\\python.exe scripts\\check_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymysql                                    # noqa: E402

from config.settings import Settings               # noqa: E402

#: 本项目需要的表
EXPECTED_TABLES = {
    "drill_record": "演练记录",
    "daily_summary": "每日总结",
    "miss_log": "未命中日志",
    "exam_paper": "试卷（含题目快照）",
    "exam_paper_question": "试卷题目明细",
    "question_usage": "题目使用记录（防重复）",
}


def parse_db_url(url: str) -> dict:
    """从 SQLAlchemy URL 解析连接参数。"""
    # mysql+pymysql://root:pass@localhost:3306/dbname?charset=utf8mb4
    p = urlparse(url)
    return {
        "host": p.hostname or "localhost",
        "port": p.port or 3306,
        "user": p.username or "root",
        "password": p.password or "",
        "database": (p.path or "/").lstrip("/") or None,
        "charset": "utf8mb4",
    }


def main() -> int:
    settings = Settings()
    url = getattr(settings, "database_url", "")
    if not url:
        print("[FAIL] .env 里没有 DATABASE_URL")
        return 1

    cfg = parse_db_url(str(url))
    print("连接目标: "
          f"{cfg['user']}@{cfg['host']}:{cfg['port']} / {cfg['database']}")

    try:
        conn = pymysql.connect(
            host=cfg["host"], port=cfg["port"], user=cfg["user"],
            password=cfg["password"], charset=cfg["charset"])
    except Exception as exc:                      # noqa: BLE001
        print(f"[FAIL] 连接失败: {type(exc).__name__}: {exc}")
        return 1

    try:
        with conn.cursor() as cur:
            cur.execute("SHOW DATABASES")
            dbs = [r[0] for r in cur.fetchall()]
            print(f"\n已有数据库（{len(dbs)} 个）:")
            for d in dbs:
                mark = "  ← 目标库" if d == cfg["database"] else ""
                print(f"  {d}{mark}")

            target = cfg["database"]
            if not target:
                print("\n[WARN] URL 里没有指定数据库名")
                return 0

            if target not in dbs:
                print(f"\n[结论] 目标库 `{target}` **不存在**，需要创建")
                return 0

            cur.execute(f"USE `{target}`")
            cur.execute("SHOW TABLES")
            tables = {r[0] for r in cur.fetchall()}

            print(f"\n`{target}` 现有表（{len(tables)} 张）:")
            for t in sorted(tables):
                print(f"  {t}")
            if not tables:
                print("  （空库）")

            print(f"\n本项目需要的表:")
            missing = []
            for t, desc in EXPECTED_TABLES.items():
                if t in tables:
                    print(f"  [有] {t:24} {desc}")
                else:
                    print(f"  [缺] {t:24} {desc}")
                    missing.append(t)

            if missing:
                print(f"\n[结论] 缺 {len(missing)} 张表，需要建："
                      f"{', '.join(missing)}")
            else:
                print("\n[结论] 表已齐全")
                for t in sorted(EXPECTED_TABLES):
                    cur.execute(f"SHOW CREATE TABLE `{t}`")
                    print(f"\n--- {t} ---")
                    print(cur.fetchone()[1])
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
