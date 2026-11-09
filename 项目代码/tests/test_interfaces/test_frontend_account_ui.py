"""账号管理前端的硬约束（静态检查）。

密码输入必须用 type=password 对话框，不能用 prompt() 明文回显；
账号页仅主管可见；动态工号/姓名必须转义。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WEB = (Path(__file__).resolve().parents[2]
       / "interfaces" / "http" / "web")
APP_JS = WEB / "app.js"
INDEX_HTML = WEB / "index.html"


@pytest.fixture(scope="module")
def js() -> str:
    return APP_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def html() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


class TestAccountPanel:
    def test_account_tab_is_manager_only(self, html: str) -> None:
        assert re.search(
            r'data-panel="account"[^>]*data-feature="employee_view"', html
        ), "账号页必须挂 employee_view 权限"

    def test_account_panel_exists(self, html: str) -> None:
        assert 'id="panel-account"' in html
        assert 'id="accBody"' in html

    def test_password_input_is_masked(self, html: str) -> None:
        assert re.search(r'id="pwdInput"[^>]*type="password"', html), (
            "密码输入必须 type=password，不能明文回显"
        )

    def test_change_password_button_exists(self, html: str) -> None:
        assert 'id="changePwdBtn"' in html


class TestAccountActions:
    @pytest.mark.parametrize("path", [
        "/api/accounts/create",
        "/api/accounts/reset-password",
        "/api/accounts/active",
        "/api/employees/active",
        "/api/auth/change-password",
    ])
    def test_endpoint_wired(self, js: str, path: str) -> None:
        assert path in js, f"前端没有接 {path}"

    def test_password_not_via_prompt(self, js: str) -> None:
        assert "function askPassword" in js
        assert "prompt(" not in js.split("function askPassword")[1][:800], (
            "密码对话框不能用 prompt() 明文回显"
        )

    def test_dynamic_fields_escaped(self, js: str) -> None:
        body = js[js.find("async function loadAccounts"):]
        body = body[:body.find("\nasync function createAccount")]
        assert "escapeHtml(e.employee_no)" in body
        assert "escapeHtml(e.name" in body

    def test_manager_feature_allowed(self, js: str) -> None:
        assert "'employee_view'" in js, "applyAuthUI 必须放行主管的账号页"
