"""启动 Web 服务。

用法：
    .venv\\Scripts\\python.exe serve.py
    .venv\\Scripts\\python.exe serve.py --port 8080 --reload

启动后打开 http://127.0.0.1:8000/
"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from config.settings import get_settings


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="启动健康顾问培训系统 Web 服务")
    parser.add_argument("--host", default=settings.http_host,
                        help=f"监听地址（默认 {settings.http_host}）")
    parser.add_argument("--port", type=int, default=settings.http_port,
                        help=f"端口（默认 {settings.http_port}）")
    parser.add_argument("--reload", action="store_true",
                        help="代码改动自动重载（开发用；会重复加载模型，慎用）")
    parser.add_argument("--log-level", default="info",
                        choices=["critical", "error", "warning", "info", "debug"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 68)
    print(f"  健康顾问培训系统")
    print(f"  地址: http://{args.host}:{args.port}/")
    print(f"  文档: http://{args.host}:{args.port}/docs")
    print("  ★ 首次启动要加载 bge-m3，约 15 秒，请稍候…")
    print("=" * 68)

    try:
        uvicorn.run(
            "interfaces.http.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        print("\n已停止。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
