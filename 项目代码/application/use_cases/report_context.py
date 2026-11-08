"""月度报告的上下文准备（案例标题 + 真实文档清单）。

────────────────────────────────────────────────────────────────────
为什么单独放这里，而不是留在路由里
────────────────────────────────────────────────────────────────────
这两项数据被**两处**使用：

    ① 同步接口 `GET /api/reports/monthly`
    ② 批量 worker（`deps._build_report_graph`，每任务新建图）

如果留在 `routers/reports.py`，`deps.py` 就得反过来 import 路由模块 ——
`deps` 是被所有路由依赖的底层模块，被路由反向依赖是**循环依赖的前兆**。
所以抽到 application 层，两边都往下依赖，方向干净。

★ 这也符合项目一贯做法：同一套逻辑只能有一份实现，
  两份就会漂移（比如一边取了文档清单、另一边忘了取，模型又开始编资料名）。
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def case_titles(container: Any) -> dict[str, str]:
    """`case_id → 案例标题`（报告里显示中文标题而不是 C01/C02）。"""
    catalog = getattr(container, "case_catalog", None)
    if catalog is None:
        return {}
    try:
        return {s.case_id: s.title for s in catalog.list_all()}
    except Exception as exc:                       # noqa: BLE001
        logger.warning("读取案例标题失败（报告将显示 case_id）: %s", exc)
        return {}


def doc_catalog(container: Any) -> dict[str, str]:
    """从知识库取真实文档清单（`doc_id → 标题`）。

    ★ 为什么必须取真实清单：不给清单时模型会**编造资料名**
      （实测写过「《禁忌话术禁用词表（V3.2）》第4章」等三份不存在的文件），
      学员按建议根本找不到东西。
      这里把真实清单作为**封闭词表**喂给模型。
    """
    milvus = getattr(container, "milvus_client", None)
    if milvus is None:
        return {}
    settings = getattr(container, "settings", None)
    coll = getattr(settings, "coll_training", "") if settings else ""
    if not coll:
        return {}
    try:
        rows = milvus.query(
            collection_name=coll,
            filter="", output_fields=["doc_id", "title"], limit=1000)
        out: dict[str, str] = {}
        for r in rows:
            out.setdefault(r["doc_id"], r.get("title", ""))
        return out
    except Exception as exc:                       # noqa: BLE001
        logger.warning("读取文档清单失败（建议将不提资料名）: %s", exc)
        return {}


__all__ = ["case_titles", "doc_catalog"]
