"""员工批量导入解析（FR-044）。

★ 只做「行 → 档案」的转换，不碰 IO。
  CSV/Excel 读取由调用方负责，这里只管解析与校验。

────────────────────────────────────────────────────────────────────
容错策略：坏行跳过 + 如实报告
────────────────────────────────────────────────────────────────────
导入 30 人的名单，里面有 1 行姓名漏填——如果整批失败，
管理员要自己找是哪一行，很烦。

所以：**能导入的照常导入，坏行跳过并报告原因**。

★ 但同批内重复工号要去重（保留第一条），
  因为那多半是复制粘贴重复了，两条内容通常一样。
"""
from __future__ import annotations

import logging
from typing import Any

from domain.value_objects.employee import EmployeeProfile

logger = logging.getLogger(__name__)

#: 中英文表头映射。
#: ★ 为什么两种都支持：Excel 手填常用中文，系统导出常用英文。
_HEADER_MAP: dict[str, str] = {
    "工号": "employee_no",
    "员工编号": "employee_no",
    "编号": "employee_no",
    "employee_no": "employee_no",
    "emp_no": "employee_no",

    "姓名": "name",
    "名字": "name",
    "name": "name",

    "入职日期": "hire_date",
    "入职时间": "hire_date",
    "hire_date": "hire_date",

    "岗位": "position",
    "职位": "position",
    "position": "position",

    "团队": "team",
    "部门": "team",
    "所属团队": "team",
    "team": "team",

    "带教人": "mentor",
    "导师": "mentor",
    "mentor": "mentor",
}


class ImportResult:
    """导入结果（成功 + 失败明细）。

    ★ 单独一个结构，而不是只返回成功列表：
      管理员需要知道「哪几行没进来、为什么」。
    """

    def __init__(self) -> None:
        self.employees: list[EmployeeProfile] = []
        self.errors: list[str] = []
        self.duplicates: list[str] = []

    @property
    def ok(self) -> bool:
        return bool(self.employees)

    def summary(self) -> str:
        parts = [f"成功 {len(self.employees)} 条"]
        if self.duplicates:
            parts.append(f"重复 {len(self.duplicates)} 条")
        if self.errors:
            parts.append(f"失败 {len(self.errors)} 条")
        return "，".join(parts)


def _normalize(row: dict[str, Any]) -> dict[str, str]:
    """把一行按表头映射转成标准字段名。"""
    out: dict[str, str] = {}
    for raw_key, value in row.items():
        key = str(raw_key).strip()
        std = _HEADER_MAP.get(key) or _HEADER_MAP.get(key.lower())
        if std:
            out[std] = str(value).strip() if value is not None else ""
    return out


def parse_employee_rows(rows: list[dict[str, Any]]) -> list[EmployeeProfile]:
    """解析导入行，返回可用档案（坏行跳过）。

    这是给「只关心成功结果」的调用方用的简化接口。
    需要失败明细请用 `import_employees`。

    Args:
        rows: 原始行（键可以是中英文表头）。

    Returns:
        成功解析的员工档案。同批重复工号只保留第一条。
    """
    return import_employees(rows).employees


def import_employees(rows: list[dict[str, Any]]) -> ImportResult:
    """导入员工（带失败明细）。

    Args:
        rows: 原始行。

    Returns:
        `ImportResult`，含成功列表、重复工号、失败原因。
    """
    result = ImportResult()
    seen: set[str] = set()

    for idx, raw in enumerate(rows, start=1):
        data = _normalize(raw)

        emp_no = data.get("employee_no", "")
        if not emp_no:
            result.errors.append(f"第 {idx} 行：工号为空")
            continue

        if emp_no in seen:
            result.duplicates.append(emp_no)
            logger.warning("第 %d 行工号重复，跳过: %s", idx, emp_no)
            continue

        try:
            emp = EmployeeProfile(
                employee_no=emp_no,
                name=data.get("name", ""),
                hire_date=data.get("hire_date", ""),
                position=data.get("position", ""),
                team=data.get("team", ""),
                mentor=data.get("mentor", ""),
            )
        except ValueError as exc:
            result.errors.append(f"第 {idx} 行（{emp_no}）：{exc}")
            continue

        seen.add(emp_no)
        result.employees.append(emp)

    logger.info("员工导入: %s", result.summary())
    return result


__all__ = ["ImportResult", "import_employees", "parse_employee_rows"]
