"""应用层出口。

P0 阶段只有基础类型；用例从 P1 起逐步加入。
"""
from __future__ import annotations

from application.types import QuestionIntent, UseCaseResult

__all__ = ["QuestionIntent", "UseCaseResult"]
