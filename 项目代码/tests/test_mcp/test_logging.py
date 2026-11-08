"""MCP 调用日志测试。

★ 为什么日志也要测
  MCP 的价值主张之一是「身份可追溯」。如果日志写不进去、或者记的是别人，
  审计就等于没有。这三条是「能审计」的最低保证：
    1. 记的是**会话身份**
    2. 失败调用也要记（否则越权尝试会静默消失）
    3. 参数要脱敏截断（避免把用户原话整段落盘）
"""
from __future__ import annotations

import json
from pathlib import Path

from interfaces.mcp.logging_ import log_call


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


class TestLogCall:
    def test_writes_record(self, tmp_path: Path) -> None:
        p = tmp_path / "calls.jsonl"
        log_call(
            kind="tool", name="kb_search",
            employee_no="E001", role="学员",
            ok=True, elapsed_ms=123.4,
            args={"query": "孕妇能吃鱼油吗"}, result_bytes=200,
            path=p,
        )
        rows = _read(p)
        assert len(rows) == 1
        r = rows[0]
        assert r["kind"] == "tool"
        assert r["name"] == "kb_search"
        assert r["employee_no"] == "E001"
        assert r["ok"] is True
        assert r["elapsed_ms"] == 123.4
        assert r["result_bytes"] == 200
        assert "ts" in r

    def test_records_failure(self, tmp_path: Path) -> None:
        """★ 被拒的调用必须留痕.

        越权尝试不记日志，就等于没人知道有人试过。
        """
        p = tmp_path / "calls.jsonl"
        log_call(
            kind="tool", name="my_drill_reviews",
            employee_no="E002", role="学员",
            ok=False, elapsed_ms=1.0,
            args={"days": 7}, error="无权使用该工具",
            path=p,
        )
        r = _read(p)[0]
        assert r["ok"] is False
        assert "无权" in r["error"]

    def test_long_arg_truncated(self, tmp_path: Path) -> None:
        """参数超长要截断（避免整段用户原话落盘）。"""
        p = tmp_path / "calls.jsonl"
        log_call(
            kind="tool", name="kb_search",
            employee_no="E001", role="学员",
            ok=True, elapsed_ms=1.0,
            args={"query": "长" * 500},
            path=p,
        )
        r = _read(p)[0]
        assert len(r["args"]["query"]) <= 121
        assert r["args"]["query"].endswith("…")

    def test_appends_not_overwrites(self, tmp_path: Path) -> None:
        p = tmp_path / "calls.jsonl"
        for i in range(3):
            log_call(
                kind="tool", name=f"t{i}",
                employee_no="E001", role="学员",
                ok=True, elapsed_ms=1.0, path=p,
            )
        assert len(_read(p)) == 3

    def test_bad_path_does_not_raise(self, tmp_path: Path) -> None:
        """★ 日志写不进去不能让工具调用失败。

        审计是运营素材，不该影响主流程（与 miss_log 的处理一致）。
        """
        bad = tmp_path / "no_such_dir" / "x" / "y.jsonl"
        # 不应抛异常
        log_call(
            kind="tool", name="kb_search",
            employee_no="E001", role="学员",
            ok=True, elapsed_ms=1.0, path=bad,
        )
