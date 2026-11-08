"""密码哈希与校验（P8）。

★ 用 bcrypt，不用自己搓 sha256。

    自搓的常见错误：
      · 不加盐 → 彩虹表直接查出来
      · 用 sha256 单次 → GPU 每秒能试几十亿次
      · 盐和哈希拼一起但用固定分隔符 → 解析出错就降级成不安全

    bcrypt 是专门为「存密码」设计的：
      自适应成本因子（默认 12，约 250ms/次）、内置盐、
      哈希串自带算法标识与参数。

★ 为什么不引入 passlib：
    bcrypt 库本身够用，passlib 是一层封装（而且已停止维护）。
    少一层依赖，出问题时排查路径更短。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: bcrypt 成本因子。
#:
#: 12 约 250ms/次。这个值要在「安全性」与「登录体验」之间取：
#:   · 太低（10，约 60ms）→ 离线爆破成本低
#:   · 太高（14，约 1s）  → 登录明显卡顿，且并发时 CPU 打满
#:
#: 12 是业界常用起点。真机实测过登录延迟后再调。
DEFAULT_ROUNDS = 12


def hash_password(plain: str, *, rounds: int = DEFAULT_ROUNDS) -> str:
    """生成密码哈希。

    Args:
        plain: 明文密码。
        rounds: bcrypt 成本因子。

    Returns:
        bcrypt 哈希串（形如 `$2b$12$...`）。

    Raises:
        ValueError: 密码为空。

    ★ 空密码一律拒绝 —— 允许空密码等于给自己挖坑。
    """
    if not plain:
        raise ValueError("密码不能为空")

    import bcrypt

    # bcrypt 只处理前 72 字节。更长的密码会被静默截断，
    # 这会造成「两个不同的长密码哈希一样」的错觉。
    # 这里显式截断到 72 字节，并在超长时记日志。
    raw = plain.encode("utf-8")
    if len(raw) > 72:
        logger.warning("密码超过 72 字节，将被截断（bcrypt 限制）")
        raw = raw[:72]

    return bcrypt.hashpw(raw, bcrypt.gensalt(rounds=rounds)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """校验密码。

    Args:
        plain: 用户输入的明文。
        hashed: 库里的哈希串。

    Returns:
        True 表示匹配。

    ★ 任何异常都返回 False（fail-safe），不抛给上层：
      坏哈希、空哈希、格式不对 —— 都是「校验不通过」，
      不应该让登录接口 500。
    """
    if not plain or not hashed:
        return False

    try:
        import bcrypt

        raw = plain.encode("utf-8")[:72]
        return bcrypt.checkpw(raw, hashed.encode("utf-8"))
    except (ValueError, TypeError) as exc:
        # 哈希串格式坏了（比如被人工改过）
        logger.warning("密码校验异常（按失败处理）: %s", type(exc).__name__)
        return False


__all__ = ["DEFAULT_ROUNDS", "hash_password", "verify_password"]
