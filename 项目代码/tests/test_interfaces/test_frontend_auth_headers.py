"""前端请求头守卫（2026-10-23）。

────────────────────────────────────────────────────────────────────
为什么需要这个文件
────────────────────────────────────────────────────────────────────
实测缺陷：勾选「流式输出」后回答区显示 `请求失败：HTTP 401`。

根因不是后端鉴权出问题（带令牌 200、不带令牌 401，行为完全正确），
而是**前端有两条请求路径，只有一条带了令牌**：

    api()        → headers 里有 Authorization   ✅
    askStream()  → 自己拼 headers，只有 Content-Type   ❌

非流式走 `api()`，所以一直正常；流式走另一条，才暴露。
这类缺陷单测测不出来——它不在 Python 里。

★ 所以本文件做的是**静态检查前端源码**，两条硬约束：
    1. 每个 `fetch(` 调用都必须通过 `authHeaders()` 拿请求头
    2. 原始的 `logout(false)` 不能被 401 处理用到
       （`false` 表示「不显示登录框」，用户会卡在空页面上）

★ 为什么用静态检查而不是跑浏览器：
   项目没有前端测试框架（JS 只是静态资源），
   引入 Playwright 等于是为一条 5 行的规则背上一个浏览器。
   而这条规则的本质是「不要手写 headers」，静态检查刚好对口，
   且能在 CI 里 0.1 秒跑完。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

APP_JS = (Path(__file__).resolve().parents[2]
          / "interfaces" / "http" / "web" / "app.js")


@pytest.fixture(scope="module")
def js() -> str:
    return APP_JS.read_text(encoding="utf-8")


# ===================================================================
# 1. 每个 fetch 都必须带令牌
# ===================================================================
class TestFetchCarriesToken:
    def test_every_fetch_uses_auth_headers(self, js: str) -> None:
        """★ 本文件最重要的一条。

        做法：取每个 `fetch(` 之后 400 字符内的片段，
        要求其中出现 `authHeaders(`。
        400 字符是够的（实测最远的调用距离约 120 字符），
        又不会跨到下一个 fetch 造成假通过。
        """
        offenders: list[str] = []
        for m in re.finditer(r"\bfetch\(", js):
            window = js[m.start():m.start() + 400]
            if "authHeaders(" not in window:
                # 截到该调用结束的附近，便于定位
                snippet = js[max(0, m.start() - 120):m.start() + 160]
                offenders.append(snippet.replace("\n", " ⏎ "))

        assert not offenders, (
            "有 fetch 调用没走 authHeaders()，会漏带令牌导致 401：\n"
            + "\n---\n".join(offenders)
        )

    def test_no_handwritten_authorization_header(self, js: str) -> None:
        """令牌只许在 `authHeaders()` 里拼。

        ★ 手写 `'Authorization'` 的地方只要不在 authHeaders 内，
          就是在绕过唯一实现 —— 将来改令牌格式（如加前缀、换 header 名）
          必然漏改。
        """
        # authHeaders 函数体范围
        start = js.find("function authHeaders")
        assert start != -1, "authHeaders 函数不见了"
        end = js.find("\n}", start) + 2
        body = js[start:end]

        others = [
            (i, line.strip())
            for i, line in enumerate(js.split("\n"), 1)
            if "Authorization" in line and line not in body
        ]
        # 允许注释里提到（文档说明）
        others = [(i, t) for i, t in others if not t.lstrip().startswith("*")
                  and not t.lstrip().startswith("//")]
        assert not others, f"authHeaders 之外有手写 Authorization: {others}"

    def test_auth_headers_actually_sets_token(self, js: str) -> None:
        """`authHeaders` 必须真的把令牌拼进去（别被改成空实现）。"""
        start = js.find("function authHeaders")
        end = js.find("\n}", start) + 2
        body = js[start:end]
        assert "AUTH.token" in body
        assert "Bearer" in body


# ===================================================================
# 2. 401 处理必须真的回到登录页
# ===================================================================
class TestUnauthorizedHandling:
    def test_handle_unauthorized_shows_login(self, js: str) -> None:
        """★ 原代码写 `logout(false)` —— 注释说「回登录页」，实际没回。

        `false` = 不显示登录框，用户会停在一个点不动的页面上。
        """
        start = js.find("function handleUnauthorized")
        assert start != -1, "handleUnauthorized 函数不见了"
        end = js.find("\n}", start) + 2
        body = js[start:end]

        assert "logout(true)" in body, (
            "401 处理必须 logout(true) 把登录框显示出来；\n"
            "logout(false) 只清令牌不显示登录框，用户无法重新登录"
        )
        assert "logout(false)" not in body

    @pytest.mark.parametrize("path_marker", ["/api/auth/login"])
    def test_login_401_not_treated_as_expiry(self, js: str,
                                            path_marker: str) -> None:
        """登录接口的 401 是「密码错」，不能当成「令牌失效」。

        ★ 若当成令牌失效：会先清掉登录态并弹登录框，
          用户看到「表单被重置」而不是「密码不对」。
        """
        assert path_marker in js
        assert "!path.includes(" in js, (
            "401 处理必须排除登录接口 —— 否则密码输错会先清空表单"
        )


# ===================================================================
# 3. 结构约束
# ===================================================================
class TestStructure:
    def test_stream_uses_api_auth_path(self, js: str) -> None:
        """流式入口必须存在且带令牌（这是本次缺陷的原始位置）。"""
        start = js.find("async function askStream")
        assert start != -1
        window = js[start:start + 500]
        assert "authHeaders()" in window, (
            "askStream 又自己拼 headers 了 —— 这就是本次 401 的成因"
        )
        assert "401" in window, "askStream 也要处理 401"

    def test_no_duplicate_header_builders(self, js: str) -> None:
        """只允许一个地方拼 `Content-Type` + 令牌。

        多于一处就说明有人在另起一套 —— 必然漂移。
        """
        builders = re.findall(
            r"Content-Type['\"]?\s*:\s*['\"]application/json", js)
        assert len(builders) == 1, (
            f"发现 {len(builders)} 处手写 Content-Type，应只有 authHeaders() 一处"
        )
