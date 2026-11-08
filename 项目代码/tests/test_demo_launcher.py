"""演示入口必须区分真实就绪、降级服务和占用端口的其他应用。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def launcher() -> ModuleType:
    """只加载启动函数，不启动进程或连接数据库。"""
    path = Path(__file__).resolve().parents[2] / "一键启动.py"
    spec = importlib.util.spec_from_file_location("demo_launcher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("state,expected", [
    ("ready", True), ("partial", False), ("pending", False),
    ("other_app", False), ("offline", False), ("degraded", False),
])
def test_readiness_requires_this_app_and_all_capabilities(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch,
    state: str, expected: bool,
) -> None:
    health = {"status": "ok", "ready_ports": list(launcher.REQUIRED_PORTS), "pending_ports": {}}
    document = {"info": {"title": "健康顾问培训系统"}}
    if state == "partial":
        health["ready_ports"] = ["graph"]
    elif state == "pending":
        health["pending_ports"] = {"repository": "unavailable"}
    elif state == "other_app":
        document["info"]["title"] = "另一个项目"
    elif state == "degraded":
        health["status"] = "degraded"
    monkeypatch.setattr(launcher, "get_json", lambda url: (
        None if state == "offline" else health if url.endswith("/api/health") else document))

    assert launcher.app_ready("http://127.0.0.1:8000") is expected


def test_wait_stops_when_new_process_exits(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = Mock()
    process.poll.return_value = 1
    sleep = Mock()
    monkeypatch.setattr(launcher.time, "sleep", sleep)

    with pytest.raises(launcher.StartupError, match="进程已退出"):
        launcher.wait_for("培训应用", lambda: False, 180, process)

    sleep.assert_not_called()


@pytest.mark.parametrize("encoding", ["utf-8", "gbk"])
def test_windows_command_output_keeps_chinese_readable(
    launcher: ModuleType, encoding: str,
) -> None:
    message = "本项目的后台应用已停止。"
    assert launcher.decode_output(message.encode(encoding)) == message
