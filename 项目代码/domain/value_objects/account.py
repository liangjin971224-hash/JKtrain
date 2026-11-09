"""账号与审计日志（P8）。

★ 账号与员工档案分开的原因：

    员工档案（`EmployeeProfile`）= 人事数据：姓名、入职日期、团队、带教人
    账号（`EmployeeAccount`）    = 凭据数据：密码哈希、角色、启用状态

    两者生命周期不同：
      · 员工入职 → 建档案；开通系统 → 建账号
      · 员工转岗 → 改档案 team；改权限 → 改账号 role
      · 员工离职 → 档案保留（历史可查）；账号停用（不能登录）

    合在一起会导致「查员工列表」时把密码哈希也带出来 —— 这是很常见的泄露。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from domain.value_objects.role import Role


@dataclass(frozen=True, slots=True)
class EmployeeAccount:
    """员工账号（凭据）。

    ★★ 这个对象**不应被序列化到任何面向用户的响应里**。
      `password_hash` 只能存在于认证链路内部。
      接口层要返回账号信息时，先剥掉哈希。
    """

    employee_no: str
    #: 角色（存字符串，读时用 `Role.parse` 校验）
    role: str = "trainee"
    password_hash: str = ""
    #: 是否启用（离职后停用，但账号记录保留）
    active: bool = True

    def __post_init__(self) -> None:
        if not self.employee_no or not self.employee_no.strip():
            raise ValueError("工号不能为空")

    @property
    def role_enum(self) -> Role:
        return Role.parse(self.role)

    def to_public(self) -> dict:
        """给前端的公开信息（**不含密码哈希**）。"""
        return {
            "employee_no": self.employee_no,
            "role": self.role,
            "role_label": self.role_enum.label,
            "active": self.active,
        }


class AuditAction(StrEnum):
    """审计动作（只记「关键动作」，不记普通查询）。"""

    LOGIN = "login"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    EXAM_BUILD = "exam_build"
    EXAM_EXPORT = "exam_export"
    EXAM_STATUS = "exam_status"
    EMPLOYEE_IMPORT = "employee_import"
    SUMMARY_REVIEW = "summary_review"
    #: 账号管理（改角色 / 启停用）—— 单独一类，
    #: 因为它直接改变权限，审计价值高于普通业务操作
    ACCOUNT_ROLE = "account_role"
    ACCOUNT_ACTIVE = "account_active"
    ACCOUNT_PASSWORD = "account_password"
    ACCOUNT_CREATE = "account_create"
    PERMISSION_DENIED = "permission_denied"


class AuditResult(StrEnum):
    OK = "ok"
    FAILED = "failed"
    DENIED = "denied"


@dataclass(frozen=True, slots=True)
class AuditLogEntry:
    """审计日志。

    要能回答：**谁**在**什么时候**对**什么**做了**什么**，成没成。

    ★★ 刻意**不含**任何敏感字段：
      没有 password、token、secret。审计日志是长期保存的，
      把凭据写进去等于把密码永久留档。

      这与「未命中日志不含评分字段」是同一类设计：
      **能力边界写进数据结构，比写在文档里可靠。**
    """

    actor: str
    role: str
    action: str
    target: str = ""
    result: str = AuditResult.OK.value
    detail: dict = field(default_factory=dict)
    created_at: str = ""
    ip: str = ""

    def __post_init__(self) -> None:
        if not self.actor:
            raise ValueError("操作人不能为空")
        if not self.action:
            raise ValueError("动作不能为空")


__all__ = ["AuditAction", "AuditLogEntry", "AuditResult", "EmployeeAccount"]
