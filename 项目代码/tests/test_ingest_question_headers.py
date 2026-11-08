"""题库迁移时，标题匹配不能吞掉下一行的题干。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_ingest(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """加载项目旁的入库入口，不连接数据库或加载模型。"""
    directory = Path(__file__).resolve().parents[2] / "08_数据入库" / "milvus"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "health_training_ingest", directory / "build_collections.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_question_stems_survive_blank_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, newline: str,
) -> None:
    """保留普通题、多选题和末尾题的题干、答案及阶段信息。"""
    module = _load_ingest(monkeypatch)
    folder = tmp_path / "03_题库"
    folder.mkdir()
    source = "\n".join([
        "# 一、KP_PROD_CATEGORY 产品类别辨识（2 道）", "",
        "## Q005 ★☆☆", "", "**以下哪个标识是保健食品的专用标志？**", "",
        "A. 绿色食品标志", "B. 蓝帽子标志", "", "答案：B", "",
        "## S023 ★★☆（多选）", "", "**下列哪些属于产品类别？**", "",
        "A. 保健食品", "B. 普通食品", "", "答案：AB", "",
    ])
    (folder / "题库_S1.md").write_bytes(source.replace("\n", newline).encode("utf-8"))
    monkeypatch.setattr(module, "BASE", tmp_path)

    rows = module.parse_questions()

    assert [r["chunk_id"] for r in rows] == ["Q005", "S023"]
    assert rows[0]["text"].startswith("以下哪个标识是保健食品的专用标志？")
    assert rows[1]["text"].startswith("下列哪些属于产品类别？")
    assert rows[0]["question_type"] == "单选"
    assert rows[1]["question_type"] == "多选"
    assert all(r["section"] == "S1" for r in rows)
    assert all(r["knowledge_point"] == "KP_PROD_CATEGORY" for r in rows)
