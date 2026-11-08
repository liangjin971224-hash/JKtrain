"""演练评语工具：`my_drill_reviews`。

════════════════════════════════════════════════════════════════════
★ 这个工具的签名里**没有 employee_id**，这是刻意的
════════════════════════════════════════════════════════════════════
如果写成 `my_drill_reviews(employee_id: str)`，Agent 只要传别人的工号
就能读到别人的评语 —— 而 `drill_review` 表有 **183 行真实评语**，
包含「客户自述在吃阿司匹林」「客户对价格敏感」这类内部记录。

这是 MCP 与传统 REST 最大的差别：
    REST 侧身份来自 JWT header，路由里天然知道「当前是谁」；
    MCP 的工具调用**没有「当前用户」概念**，身份只在连接建立时存在。

所以设计成：
    · 工具签名只有 `days`（时间范围）
    · 身份从会话对象取
    · `trainee_id` 在服务端写入时就是会话的工号

────────────────────────────────────────────────────────────────────
为什么不用 SummaryCard（摘要卡）
────────────────────────────────────────────────────────────────────
项目红线：摘要卡**不出现 score / verdict / grade**（不给学员打分）。
MCP 返回也遵守同一条——只给「练了几次、错在哪、要素命中情况」，
把「好/不好」留给主管判断（企业终判）。
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta

from config.assembly import Container
from interfaces.mcp.identity import McpIdentity
from interfaces.mcp.policy import require_tool

logger = logging.getLogger(__name__)


def _clamp_days(value: int, *, low: int = 1, high: int = 90) -> int:
    """把 days 夹到合法区间。"""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 7
    return max(low, min(high, n))


def _as_text(value: object) -> str:
    """把 datetime / str / None 统一转成字符串。

    ★ 为什么要这个函数：
      仓储端口声明 `created_at: datetime`，但 MySQL 适配器实际返回 str。
      类型不匹配是「端口契约与实现不一致」，这里做兼容而不是去改适配器——
      改适配器会影响 HTTP 侧已跑通的路径，风险更大。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return str(iso())
    return str(value)


def my_drill_reviews(
    container: Container,
    identity: McpIdentity,
    *,
    days: int = 7,
) -> str:
    """查询当前会话用户自己的演练评语。

    Args:
        container: 容器。
        identity: **会话身份**（不是参数传入的）。
        days: 查最近多少天（1–90）。

    Returns:
        JSON 字符串：{trainee_id, days, count, reviews, note}
    """
    require_tool(identity, "my_drill_reviews")

    repo = container.repository
    if repo is None:
        return json.dumps(
            {"error": "仓储未就绪，无法读取评语"}, ensure_ascii=False)

    # ★ 只用会话身份，绝不读参数。
    #   若调用方在参数里塞 employee_id，MCP 层根本不会读它——
    #   因为函数签名里没有这个参数，SDK 会直接忽略未知字段。
    trainee_id = identity.employee_no

    n = _clamp_days(days)
    since = (date.today() - timedelta(days=n)).isoformat()

    rows = repo.list_drill_reviews(
        trainee_id=trainee_id,
        since=since,
        with_items=True,
    )

    reviews = []
    for r in rows:
        reviews.append({
            "drill_id": r.drill_id,
            "case_id": r.case_id,
            "case_type": r.case_type,
            "duration_sec": r.duration_sec,
            # 要素命中（不给分，只给原始计数）
            "element_hit": r.element_hit,
            "element_total": r.element_total,
            # 风险点计数
            "risk_count": r.risk_count,
            "critical_count": getattr(r, "critical_count", 0),
            # ★ created_at 的类型不固定：仓储实现可能返回 datetime 或 str。
            #   实测 MySQL 适配器返回 str（"2026-10-21 14:30:00"），
            #   直接调 .isoformat() 会 AttributeError。
            "created_at": _as_text(getattr(r, "created_at", "")),
        })

    payload = {
        "trainee_id": trainee_id,
        "days": n,
        "since": since,
        "count": len(reviews),
        "reviews": reviews,
        "note": (
            "只返回演练要素命中与风险点计数，不给出分数或优劣判定。"
            "如需判断训练效果，请结合 case_type 与 risk_count 人工分析。"
        ),
    }
    if not reviews:
        payload["note"] = (
            f"{trainee_id} 在最近 {n} 天没有演练记录。"
            "请确认是否已完成情景演练，或放宽 days 范围。"
        )

    return json.dumps(payload, ensure_ascii=False)
