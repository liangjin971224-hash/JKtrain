"""前端两条合并后新增的硬约束（静态检查）。

────────────────────────────────────────────────────────────────────
1. 多轮会话 ID 必须由前端生成
────────────────────────────────────────────────────────────────────
合并进来的后端契约是「不传 `session_id` = 单轮，后端不会替你新建会话」
（单轮响应必须逐字节不变）。所以前端如果等后端发 id，就永远拿不到 id，
追问永远接不上——2026-11-08 浏览器验收实测：追问「那孕妇能吃吗」被拒答。

约束：提问前必须 `ensureAskSession()`，且 id 格式满足后端校验
`[A-Za-z0-9_-]{8,64}`。

2. 答案里的 Markdown 必须先转义再渲染
────────────────────────────────────────────────────────────────────
模型输出带 `**加粗**`、`* 列表`，直接 `textContent` 会把符号原样显示。
渲染时**绝不能**把模型原文塞进 `innerHTML`——模型输出是不可信内容。

约束：`mdToHtml` 只能对 `escapeHtml(...)` 的结果套标签。
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


def _body(src: str, func: str) -> str:
    start = src.find(f"function {func}")
    assert start != -1, f"{func} 不见了"
    end = src.find("\n}", start)
    assert end != -1
    return src[start:end]


class TestAskSession:
    def test_frontend_generates_session_id(self, js: str) -> None:
        body = _body(js, "ensureAskSession")
        assert "localStorage.setItem" in body, "会话 ID 必须持久化，刷新后还能接上"
        assert "web-" in body, "id 前缀便于排障（后端要求 8~64 位字母数字_-）"

    def test_ask_body_always_carries_session(self, js: str) -> None:
        """★ 关键：不能再写成 `if (ASK_SESSION) body.session_id = ...`。

        那样第一次提问没有 id，后端按单轮处理且不返回 id → 永远是单轮。
        """
        assert re.search(r"body\.session_id\s*=\s*ensureAskSession\(\)", js), (
            "提问请求必须无条件带 session_id（由 ensureAskSession 生成）"
        )
        assert "if (ASK_SESSION) body.session_id" not in js, (
            "旧的「有才带」写法会让首次提问丢失会话"
        )

    def test_new_conversation_still_sent(self, js: str) -> None:
        """「新对话」只清历史、不换 id（后端 new_conversation 语义）。"""
        assert "new_conversation: true" in js


class TestMarkdown:
    def test_escape_before_tags(self, js: str) -> None:
        body = _body(js, "mdToHtml")
        assert "escapeHtml(" in body, "渲染前必须转义 HTML"
        # 每个插值都必须是 mdInline(escapeHtml(...))
        for m in re.finditer(r"\$\{([^}]*)\}", body):
            expr = m.group(1)
            assert "mdInline(escapeHtml(" in expr, (
                f"有未转义的插值：{expr}"
            )

    def test_answer_paths_use_renderer(self, js: str) -> None:
        """流式与非流式都必须走 setAnswerHtml（不能一处渲染一处原样）。"""
        assert js.count("setAnswerHtml(target, text)") == 1, "流式未走渲染器"
        assert js.count("setAnswerHtml(ui.body, r.answer") == 1, "非流式未走渲染器"

    def test_health_text_not_hardcoded_port_count(self, js: str) -> None:
        """端口数 7→10 后，写死 `/7` 会显示「10/7」。"""
        assert "/7 端口" not in js
        assert "ready >= 8" in js


class TestStaticVersioning:
    def test_assets_are_versioned(self) -> None:
        """静态资源必须带版本号，否则浏览器缓存旧 JS（本次实测踩到）。"""
        html = INDEX_HTML.read_text(encoding="utf-8")
        assert re.search(r'/static/app\.js\?v=\d{8}', html), "app.js 缺版本参数"
        assert re.search(r'/static/style\.css\?v=\d{8}', html), "style.css 缺版本参数"
