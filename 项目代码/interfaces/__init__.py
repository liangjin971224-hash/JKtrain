"""协议层出口。

P0 阶段只有错误映射；路由从 P1 起逐步加入。
"""
from __future__ import annotations

from interfaces.http.errors import ErrorBody, map_domain_error

__all__ = ["ErrorBody", "map_domain_error"]
