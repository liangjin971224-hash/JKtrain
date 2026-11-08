"""程序入口。

P0 验收要求：`python main.py` 能启动，并打印自检结果
（Neo4j / Milvus / 模型三项连接情况，缺哪个报哪个）。

用法：
    python main.py            # 跑自检并退出
    python main.py --serve    # 跑自检后启动 FastAPI（P1 起才有路由）
"""
from __future__ import annotations

import logging
import sys

from config.assembly import build_container
from config.settings import get_settings
from config.startup_check import run_startup_checks, summarize

logger = logging.getLogger("main")

SEP = "=" * 68


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    """跑自检。返回进程退出码：0 = 可启动，1 = 有阻塞项。"""
    settings = get_settings()
    _setup_logging(settings.log_level)

    print(SEP)
    print("健康顾问培训系统 · P0 骨架自检")
    print(SEP)
    print(f"  环境        {settings.app_env}")
    print(f"  Neo4j       {settings.neo4j_uri} / {settings.neo4j_database}")
    print(f"  Milvus      {settings.milvus_uri}")
    print(f"  模型        {settings.embed_model_dir}")
    print(f"  存储根      {settings.storage_root_path}")
    print("-" * 68)

    # 1. 环境自检
    results = run_startup_checks(settings)
    for r in results:
        print(r.format())

    # 2. 容器装配
    print("-" * 68)
    container = build_container(settings)
    try:
        ready = container.ready_ports()
        print(f"  容器端口就绪  {len(ready)}/7  {ready}")
        if container.pending:
            print("  待实现端口:")
            for name, reason in container.pending.items():
                print(f"    - {name:<14} {reason}")
    finally:
        container.close()

    # 3. 合规护栏规则清单
    #    纯领域逻辑，不依赖任何外部服务，所以自检里能直接跑
    print("-" * 68)
    try:
        from domain.services.output_guardrail import OutputGuardrail

        rules = OutputGuardrail().describe_rules()
        print("  合规护栏  " + " / ".join(
            f"{k} {v}" for k, v in rules.items()))
    except Exception as exc:                      # noqa: BLE001
        print(f"  [WARN] 护栏规则加载失败: {type(exc).__name__}: {exc}")

    # 4. 结论
    print("-" * 68)
    ok, summary = summarize(results)
    print(summary)
    print(SEP)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
