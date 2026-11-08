"""认证能力端口（P8）。

★ 为什么要抽端口，而不是在用例里直接 import infrastructure.auth

    门禁 `test_application_does_not_import_infrastructure` 禁止用例层
    依赖具体实现。这条规则的实质意义：

      用例层关心的是「校验密码」「签发令牌」这两件事，
      不关心用的是 bcrypt 还是 argon2、HMAC 还是 RSA。

    抽成端口后：
      · 换哈希算法 → 只改适配器
      · 测试 → 注入假实现，不用跑 250ms 的 bcrypt

────────────────────────────────────────────────────────────────────
两个端口，分开定义
────────────────────────────────────────────────────────────────────
`PasswordHasher`  只管「哈希 / 校验」，纯函数，无状态
`TokenSigner`     只管「签发 / 校验 / 解析」，需要密钥

分开的原因：测试时常需要「校验永远失败」的密码端口，
但令牌还要能正常签发 —— 合在一起就没法单独替换。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PasswordHasherPort(ABC):
    """密码哈希端口。"""

    @abstractmethod
    def hash(self, plain: str) -> str:
        """生成哈希。

        Raises:
            ValueError: 密码为空。
        """

    @abstractmethod
    def verify(self, plain: str, hashed: str) -> bool:
        """校验。

        Returns:
            True 表示匹配。**任何异常都返回 False**，不抛出。
        """


class TokenSignerPort(ABC):
    """令牌签发与校验端口。"""

    @abstractmethod
    def issue(
        self,
        *,
        employee_no: str,
        role: str,
        ttl_sec: int | None = None,
    ) -> str:
        """签发令牌。"""

    @abstractmethod
    def verify(self, token: str) -> dict[str, Any] | None:
        """校验令牌。

        Returns:
            载荷字典；无效/过期/签名不符返回 None。
        """


__all__ = ["PasswordHasherPort", "TokenSignerPort"]
