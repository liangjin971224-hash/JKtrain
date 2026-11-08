"""统一时间来源。

领域层不做 IO、不直接读系统时间（时间也是外部依赖）。
需要时间的地方由调用方传入，测试里则可用固定值复现。
"""
from __future__ import annotations

from datetime import UTC, datetime


def now_iso() -> str:
    """当前时间的 ISO 8601 字符串（UTC，带时区）。

    测试中如需固定时间，请直接构造字符串而不是调这个函数。
    """
    return datetime.now(UTC).isoformat(timespec="seconds")
