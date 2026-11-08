"""MCP 调用日志。

★ 为什么要单独记日志
  MCP 的价值主张之一是「身份可追溯」。审计日志表（`audit_log`，95 行）只记
  HTTP 侧的操作，MCP 是另一个入口——不记就等于开了个不受监控的后门。

  这里写**独立 jsonl**，不改 MySQL 表结构，理由：
    · 不动在用的表（避免和主线开发冲突，方案 10.4）
    · jsonl 便于本地排查（`tail -f` 就能看）
    · 后续若要入库，这份文件就是现成的迁移素材

★ 记什么 / 不记什么
  记：时间、工具、身份、耗时、是否被拒、返回字节数、脱敏后的参数
  不记：完整答案原文（可能含内部内容）、token
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: 日志路径。默认放项目 data/ 下，与运行时数据一起。
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "mcp_tool_calls.jsonl"


def _mask(value: Any, limit: int = 120) -> Any:
    """参数脱敏与截断。

    ★ 参数可能包含用户原话（含姓名、电话），落盘前截断到 120 字符。
      完整原话对排障够用，超长只会让日志难读。
    """
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_mask(v, limit) for v in value][:10]
    if isinstance(value, dict):
        return {k: _mask(v, limit) for k, v in list(value.items())[:10]}
    return str(value)[:limit]


def log_call(
    *,
    kind: str,
    name: str,
    employee_no: str,
    role: str,
    ok: bool,
    elapsed_ms: float,
    args: dict | None = None,
    result_bytes: int = 0,
    error: str = "",
    path: Path | None = None,
) -> None:
    """追加一条调用日志。

    Args:
        kind: tool / resource / prompt。
        name: 工具名。
        employee_no: 调用者工号。
        role: 角色标签。
        ok: 是否成功。
        elapsed_ms: 耗时（毫秒）。
        args: 调用参数（会脱敏）。
        result_bytes: 返回内容字节数。
        error: 失败原因（成功时为空）。
        path: 日志路径（测试可注入临时路径）。
    """
    target = path or _DEFAULT_PATH
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kind": kind,
        "name": name,
        "employee_no": employee_no,
        "role": role,
        "ok": ok,
        "elapsed_ms": round(elapsed_ms, 1),
        "result_bytes": result_bytes,
        "args": _mask(args or {}),
    }
    if error:
        record["error"] = error[:300]

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        # ★ 日志写不进去不能让工具调用失败。
        #   审计是运营素材，不该影响主流程（与 miss_log 的处理一致）。
        logger.warning("MCP 调用日志写入失败（不影响调用）: %s", exc)
