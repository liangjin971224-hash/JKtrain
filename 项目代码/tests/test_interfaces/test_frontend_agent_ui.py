"""Agent 前端静态约束：入口、时间线安全渲染、接口接线。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[2] / "interfaces" / "http" / "web"


@pytest.fixture(scope="module")
def html() -> str:
    return (WEB / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def js() -> str:
    return (WEB / "app.js").read_text(encoding="utf-8")


class TestAgentPage:
    def test_tab_exists_for_both_roles(self, html: str) -> None:
        assert re.search(
            r'data-panel="agent"[^>]*data-feature="ask"', html)

    def test_panel_and_timeline_exist(self, html: str) -> None:
        assert 'id="panel-agent"' in html
        assert 'id="agentPlan"' in html
        assert 'id="agentSteps"' in html
        assert 'id="agentAnswer"' in html

    def test_endpoint_wired(self, js: str) -> None:
        assert "/api/agent/chat" in js

    def test_plan_and_steps_use_text_content(self, js: str) -> None:
        body = js[js.find("function renderAgentTimeline"):]
        body = body[:body.find("\nasync function runAgent")]
        assert "textContent" in body
        assert "innerHTML" not in body, "计划/步骤不能用 innerHTML 拼不可信内容"

    def test_answer_uses_safe_markdown(self, js: str) -> None:
        body = js[js.find("async function runAgent"):]
        body = body[:body.find("\n$('agentRunBtn')")]
        assert "mdToHtml" in body

    def test_static_versions_present(self, html: str) -> None:
        assert re.search(r'/static/app\.js\?v=\d{8}', html)
        assert re.search(r'/static/style\.css\?v=\d{8}', html)
