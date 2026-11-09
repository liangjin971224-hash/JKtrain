"""问答接口并发冒烟（只读，不写库）。

    python scripts/load_smoke.py --requests 20 --concurrency 5
    python scripts/load_smoke.py --requests 50 --concurrency 10

用途：上线/演示前确认「多个人同时问」不会 500、不会把 LLM 打挂。
不是压测工具——它跑的是真实检索 + 真实 LLM，成本与延迟都是真的。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_QUESTIONS = [
    "鱼油有什么功能？",
    "孕妇能吃破壁灵芝孢子粉吗？",
    "客户说吃了产品头晕要求赔偿，怎么处理？",
    "鱼油每天吃几粒？",
    "普通食品能说增强免疫力吗？",
    "客户在吃抗凝药，能吃鱼油吗？",
]


def _post(url: str, payload: dict, token: str = "", timeout: float = 60.0):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, body, time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body, time.perf_counter() - started
    except Exception as exc:                    # noqa: BLE001 网络错误也要计数
        return 0, f"{type(exc).__name__}: {exc}", time.perf_counter() - started


def main() -> int:
    ap = argparse.ArgumentParser(description="问答接口并发冒烟")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--employee", default="E001")
    ap.add_argument("--password", default="trainee123")
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    login_status, login_body, _ = _post(
        f"{base}/api/auth/login",
        {"employee_no": args.employee, "password": args.password},
        timeout=args.timeout,
    )
    if login_status != 200:
        print(f"[FAIL] 登录失败 status={login_status} body={login_body[:200]}")
        print("       先启动应用：python serve.py")
        return 1
    token = json.loads(login_body)["token"]
    print(f"登录 OK：{args.employee} @ {base}")

    questions = [DEFAULT_QUESTIONS[i % len(DEFAULT_QUESTIONS)]
                 for i in range(args.requests)]
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                _post,
                f"{base}/api/ask",
                {"question": q},
                token,
                args.timeout,
            )
            for q in questions
        ]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - started

    ok = [r for r in results if r[0] == 200]
    status_counts: dict[int, int] = {}
    refused = 0
    for status, body, _ in results:
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == 200:
            try:
                if json.loads(body).get("refused"):
                    refused += 1
            except json.JSONDecodeError:
                pass

    latencies = [r[2] for r in ok]
    print(f"\n请求 {len(results)} | 并发 {args.concurrency} | 总耗时 {elapsed:.1f}s")
    print(f"状态分布: {status_counts}")
    print(f"成功 {len(ok)} | 失败 {len(results) - len(ok)} | 拒答 {refused}")
    if latencies:
        latencies.sort()
        def pct(p: float) -> float:
            idx = min(len(latencies) - 1, int(len(latencies) * p))
            return latencies[idx]
        print(
            f"延迟 P50 {statistics.median(latencies):.2f}s | "
            f"P95 {pct(0.95):.2f}s | max {latencies[-1]:.2f}s"
        )
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
