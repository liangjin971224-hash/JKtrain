"""签名令牌（P8）。

★ 为什么自己写而不引入 PyJWT

    我们只需要「签发 + 校验 + 过期」这一个能力。
    标准库 `hmac` + `hashlib` + `base64` 三十行就能做对，
    而且**签名比较用 `hmac.compare_digest`**（防时序攻击）。

    引入 PyJWT 的代价：
      · 多一个依赖，升级时可能引入 CVE
      · 它的默认行为是「躺在 token 里的东西都可信」，
        而这正是我们要避免的（见下面的 role_is_snapshot）
      · 我们不需要 JWT 的算法协商（`alg: none` 攻击的根源）

★ 结构

    base64url(payload_json) + "." + base64url(hmac_sha256(payload, secret))

    与 JWT 的「header.payload.signature」不同，这里省掉 header ——
    没有算法协商，就不存在算法混淆攻击。

★ ★ 一个重要设计：role 只存快照，不作信任依据

    FR-048 要求「权限变更应即时生效」。
    如果角色只从 token 读，改库后旧 token 仍按旧角色放行，
    直到 token 过期 —— 这就是「变更不即时」。

    所以载荷里加 `role_is_snapshot: True`，
    业务代码必须**回库取最新角色**。token 里的 role 只用于
    「快速看一眼大概是什么身份」，不用于鉴权判断。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import typing

#: 默认有效期：8 小时（一个工作日）。
#:
#: 太短（1 小时）→ 学员练到一半被踢出去，体验差。
#: 太长（30 天）→ 手机丢了就是长期风险。
DEFAULT_TTL_SEC = 8 * 3600


def _b64e(raw: bytes) -> str:
    """base64url 编码（去掉 padding，便于放 URL/header）。"""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    """base64url 解码（补齐 padding）。"""
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def _sign(payload: str, secret: str) -> str:
    """对载荷签名。"""
    mac = hmac.new(secret.encode("utf-8"),
                   payload.encode("ascii"), hashlib.sha256)
    return _b64e(mac.digest())


def issue_token(
    *,
    employee_no: str,
    role: str,
    secret: str,
    ttl_sec: int = DEFAULT_TTL_SEC,
) -> str:
    """签发令牌。

    Args:
        employee_no: 工号。
        role: 角色快照（**不作鉴权依据**，见模块注释）。
        secret: 签名密钥。必须足够长（建议 ≥32 字节随机串）。
        ttl_sec: 有效期（秒）。负数用于测试过期。

    Returns:
        令牌串。
    """
    now = int(time.time())
    payload = {
        "sub": employee_no,
        "role": role,
        # ★ 标记：这个角色只是快照，鉴权必须回库取最新值
        "role_is_snapshot": True,
        "iat": now,
        "exp": now + ttl_sec,
    }
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return f"{body}.{_sign(body, secret)}"


def verify_token(
    token: str,
    *,
    secret: str,
) -> dict[str, typing.Any] | None:
    """校验令牌。

    Args:
        token: 令牌串。
        secret: 签名密钥。

    Returns:
        载荷字典；任何校验失败返回 None。

    ★ 所有失败都返回 None，不抛异常：
      过期、签名错、格式坏 —— 对调用方都是「没通过」。
      抛异常会让调用方不得不到处 try/except。
    """
    if not token or not secret:
        return None

    parts = token.split(".")
    if len(parts) != 2:
        return None

    body, sig = parts

    # ★ 时序安全比较：不能直接用 ==，否则可以通过响应时间差
    #   逐字节猜出正确签名。
    expected = _sign(body, secret)
    if not hmac.compare_digest(sig, expected):
        return None

    try:
        payload = json.loads(_b64d(body).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    # 过期检查
    exp = payload.get("exp")
    if not isinstance(exp, int) or exp < int(time.time()):
        return None

    # 必填字段
    if not payload.get("sub"):
        return None

    return payload


__all__ = ["DEFAULT_TTL_SEC", "issue_token", "verify_token"]
