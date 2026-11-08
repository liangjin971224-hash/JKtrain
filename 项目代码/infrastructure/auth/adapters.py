"""认证端口的具体实现（P8）。

★ 把 `password.py` / `token.py` 的函数包成端口实现。
  函数版本保留（有些地方直接调更顺手），这里是给用例层用的。

★ 为什么不合并成一个大类：
  `PasswordHasher` 和 `TokenSigner` 是两个独立能力，
  合在一起后测试时想只替换其中一个就很别扭。
"""
from __future__ import annotations

from typing import Any

from domain.ports.auth_port import PasswordHasherPort, TokenSignerPort
from infrastructure.auth.password import hash_password, verify_password
from infrastructure.auth.token import DEFAULT_TTL_SEC, issue_token, verify_token


class BcryptPasswordHasher(PasswordHasherPort):
    """bcrypt 实现。"""

    def __init__(self, *, rounds: int | None = None) -> None:
        #: 测试时可传小值（如 4）让哈希快一点
        self._rounds = rounds

    def hash(self, plain: str) -> str:
        if self._rounds is None:
            return hash_password(plain)
        return hash_password(plain, rounds=self._rounds)

    def verify(self, plain: str, hashed: str) -> bool:
        return verify_password(plain, hashed)


class HmacTokenSigner(TokenSignerPort):
    """HMAC-SHA256 实现。"""

    def __init__(self, *, secret: str, ttl_sec: int = DEFAULT_TTL_SEC) -> None:
        self._secret = secret
        self._ttl = ttl_sec

    def issue(
        self,
        *,
        employee_no: str,
        role: str,
        ttl_sec: int | None = None,
    ) -> str:
        return issue_token(
            employee_no=employee_no,
            role=role,
            secret=self._secret,
            ttl_sec=self._ttl if ttl_sec is None else ttl_sec,
        )

    def verify(self, token: str) -> dict[str, Any] | None:
        return verify_token(token, secret=self._secret)


__all__ = ["BcryptPasswordHasher", "HmacTokenSigner"]
