"""案例目录访问（应用层服务）。

────────────────────────────────────────────────────────────────────
为什么这一层存在
────────────────────────────────────────────────────────────────────
门禁 `test_interfaces_does_not_import_infrastructure` 禁止接口层直接
import 基础设施。而「读案例文件」需要 IO，属于 infrastructure。

所以链路是：
    interfaces/http/routers/drill.py
        ↓ 只 import 这个模块（application 层）
    application/services/case_catalog.py      ← 当前文件
        ↓ 需要 IO 时，通过**注入的 loader 函数**，不直接 import 实现
    infrastructure/content/case_loader.py

★ 关键设计：`case_catalog` **不 import** `case_loader`。
  loader 由装配环节（`interfaces/http/drill.py` 的调用方、或测试）注入。
  这样：
    · 应用层不依赖具体实现（门禁要求）
    · 测试可以注入假 loader，不读真实文件
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from domain.value_objects import CaseSpec

logger = logging.getLogger(__name__)

#: loader 的函数签名：接受目录，返回案例列表。
CaseLoader = Callable[[Path], list[CaseSpec]]


class CaseCatalog:
    """案例目录（带进程内缓存）。

    ★ 案例是**只读内容资产**，缓存起来避免每次请求都读盘解析。
      代价：改了案例 markdown 需重启服务才生效。
    """

    def __init__(self, case_dir: Path, loader: CaseLoader) -> None:
        """
        Args:
            case_dir: 案例目录。
            loader: 加载函数（由装配环节注入，避免应用层依赖具体实现）。
        """
        self._dir = case_dir
        self._loader = loader
        self._cache: dict[str, CaseSpec] | None = None

    def _ensure_loaded(self) -> dict[str, CaseSpec]:
        if self._cache is not None:
            return self._cache

        # ★ 不在这里判断「目录是否存在」。
        #   原因：目录检查属于 loader 的职责。
        #   如果这里先判存在性，注入的假 loader（不读文件）就永远拿不到调用机会，
        #   测试里所有用例都会变成「案例目录为空」。
        #   loader 自己会在目录缺失时返回空并记日志（见 case_loader）。
        try:
            cases = self._loader(self._dir)
        except Exception as exc:                  # noqa: BLE001
            logger.error("加载案例失败: %s: %s", type(exc).__name__, exc)
            cases = []

        self._cache = {c.case_id: c for c in cases}
        logger.info("案例目录已缓存: %d 个", len(self._cache))
        return self._cache

    # ---------------- 查询 ----------------
    def list_all(self) -> list[CaseSpec]:
        """全部案例（按 case_id 排序）。"""
        return sorted(self._ensure_loaded().values(),
                      key=lambda c: c.case_id)

    def get(self, case_id: str) -> CaseSpec | None:
        """按 ID 取案例，不存在返回 None。"""
        return self._ensure_loaded().get(case_id)

    def count(self) -> int:
        return len(self._ensure_loaded())

    def reload(self) -> int:
        """清缓存重新加载（改了案例文档后可用）。"""
        self._cache = None
        return self.count()


__all__ = ["CaseCatalog", "CaseLoader"]
