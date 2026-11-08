"""案例文件加载（基础设施层）。

★ 为什么单独一层：
  解析逻辑是**纯文本处理**（domain），但「读文件」属于 IO。
  门禁 `test_domain_has_no_io` 强制 domain 不做 IO，
  所以把「打开文件 + 读取文本」放在这里，解析仍复用 domain 的函数。

  这样也带来一个好处：解析逻辑可以脱离文件系统测试，
  直接喂字符串即可。
"""
from __future__ import annotations

import logging
from pathlib import Path

from domain.services.case_parser import dedupe_cases, parse_case_text
from domain.value_objects import CaseSpec

logger = logging.getLogger(__name__)


def load_cases_from_dir(case_dir: Path) -> list[CaseSpec]:
    """加载目录下全部案例集。

    Args:
        case_dir: 案例 markdown 所在目录。

    Returns:
        去重后的案例列表（按 case_id 排序）。

    Note:
        单个文件读取失败不影响其他文件——只打日志。
        案例是内容资产，某个文件暂时不可读不应让整个 P5 不可用。
    """
    if not case_dir.exists():
        logger.error("案例目录不存在: %s", case_dir)
        return []

    all_cases: list[CaseSpec] = []
    for path in sorted(case_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:                  # noqa: BLE001
            logger.error("读取案例文件失败 %s: %s: %s",
                         path.name, type(exc).__name__, exc)
            continue

        cases = parse_case_text(text, source_name=path.name)
        logger.info("已解析 %s: %d 个案例", path.name, len(cases))
        all_cases.extend(cases)

    merged = dedupe_cases(all_cases)
    logger.info(
        "案例加载完成: %d 个（去重前 %d 个）", len(merged), len(all_cases))
    return merged


def load_cases_by_id(case_dir: Path) -> dict[str, CaseSpec]:
    """加载为 {case_id: CaseSpec} 字典。"""
    return {c.case_id: c for c in load_cases_from_dir(case_dir)}


__all__ = ["load_cases_by_id", "load_cases_from_dir"]
