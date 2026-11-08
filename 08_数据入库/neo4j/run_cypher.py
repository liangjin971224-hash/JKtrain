"""
Neo4j Cypher 执行器
用途：把 .cypher 脚本按语句拆分，逐条提交到指定数据库
注意：Neo4j HTTP API 一次事务只能提交一条语句（管理命令限制），所以逐条执行
"""
import json
import os
import re
import sys
import urllib.request
import base64

# ★ 连接参数支持环境变量覆盖（2026-10-23 迁移时改）。
#   换机器时不用改代码，设环境变量或改这里即可。
NEO4J_HTTP = os.getenv("NEO4J_HTTP", "http://127.0.0.1:7474")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
DATABASE = os.getenv("NEO4J_DATABASE", "jiankangshujuku")


def build_auth():
    token = base64.b64encode(f"{NEO4J_USER}:{NEO4J_PASSWORD}".encode()).decode()
    return f"Basic {token}"


def run_cypher(statement, database=DATABASE):
    """提交单条 Cypher"""
    url = f"{NEO4J_HTTP}/db/{database}/tx/commit"
    payload = json.dumps({"statements": [{"statement": statement}]}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": build_auth(),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        return False, f"HTTP 错误: {exc}"

    if data.get("errors"):
        msgs = "; ".join(e.get("message", "") for e in data["errors"])
        return False, msgs

    results = data.get("results", [])
    if results and results[0].get("data"):
        return True, results[0]["data"]
    return True, None


def split_statements(text):
    """按分号拆分语句，跳过注释行"""
    # 去掉 // 行注释
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)

    # 按分号拆
    raw = [s.strip() for s in cleaned.split(";")]
    return [s for s in raw if s]


def main():
    if len(sys.argv) < 2:
        print("用法: python3 run_cypher.py <脚本路径> [数据库名]")
        sys.exit(1)

    path = sys.argv[1]
    database = sys.argv[2] if len(sys.argv) > 2 else DATABASE

    with open(path, encoding="utf-8") as fh:
        content = fh.read()

    statements = split_statements(content)
    print(f"共解析出 {len(statements)} 条语句，目标库: {database}")
    print("=" * 60)

    ok = 0
    fail = 0
    for idx, stmt in enumerate(statements, 1):
        preview = stmt.replace("\n", " ")[:70]
        success, info = run_cypher(stmt, database)
        if success:
            ok += 1
            if idx % 10 == 0 or idx == len(statements):
                print(f"[{idx}/{len(statements)}] OK  {preview}")
        else:
            fail += 1
            print(f"[{idx}/{len(statements)}] FAIL  {preview}")
            print(f"           错误: {info}")

    print("=" * 60)
    print(f"完成：成功 {ok} 条，失败 {fail} 条")


if __name__ == "__main__":
    main()
