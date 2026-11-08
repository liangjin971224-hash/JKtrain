"""问答多轮会话存储测试（2026-10-24）。

────────────────────────────────────────────────────────────────────
要守的三件事
────────────────────────────────────────────────────────────────────
1. **只留 3 轮**：再多会撑 prompt，再少追问就接不上
2. **只存答案摘要**：绝不存 `annotation`（含违规原文，回灌等于教模型说错话）
3. **owner 隔离**：知道 session_id 就能读别人对话（可能含过敏史、用药），这是隐私

第 3 条是演练模块踩过的同类问题（只按 id 取、不校验归属）。
"""
from __future__ import annotations

import pytest

from interfaces.http.ask_session_store import (
    ANSWER_HEAD_CHARS,
    AskSessionStore,
    AskTurn,
    make_turn,
)


def _turn(q: str = "问题", a: str = "答案", **kw) -> AskTurn:
    return make_turn(q, a, **kw)


# ===================================================================
# 1. 轮次上限
# ===================================================================
class TestTurnLimit:
    def test_keeps_only_three(self) -> None:
        store = AskSessionStore(max_turns=3)
        s = store.create("E001")
        for i in range(5):
            store.append(s.session_id, "E001", _turn(f"q{i}", f"a{i}"))

        got = store.get_or_create(s.session_id, "E001")
        assert len(got.turns) == 3
        # 保留的必须是最新 3 轮，不是最早 3 轮
        assert [t.question for t in got.turns] == ["q2", "q3", "q4"]

    def test_default_is_three(self) -> None:
        assert AskSessionStore().max_turns == 3


# ===================================================================
# 2. 摘要裁剪（不存全量，更不存 annotation）
# ===================================================================
class TestAnswerHead:
    def test_answer_is_truncated(self) -> None:
        long_answer = "啊" * 1000
        t = make_turn("q", long_answer)
        assert len(t.answer_head) == ANSWER_HEAD_CHARS

    def test_answer_head_constant(self) -> None:
        """200 字是权衡值：够判断话题，又不会把整段话术塞进 prompt。"""
        assert ANSWER_HEAD_CHARS == 200

    def test_short_answer_untouched(self) -> None:
        t = make_turn("q", "短答案")
        assert t.answer_head == "短答案"

    def test_make_turn_has_no_annotation_field(self) -> None:
        """★ 结构上就不给存 annotation 的口子。

        annotation.segments 含违规原文（供前端标红）。若进了历史，
        下一轮 prompt 会把它当"上一轮说过的话"，等于教模型说错话。
        所以 AskTurn 里**没有**这个字段，想传也传不进来。
        """
        t = make_turn("q", "a")
        assert not hasattr(t, "annotation")
        assert not hasattr(t, "segments")


# ===================================================================
# 3. owner 隔离（隐私）
# ===================================================================
class TestOwnerIsolation:
    def test_other_owner_gets_new_session(self) -> None:
        """★ 拿别人的 session_id 请求，必须拿不到原内容。

        且**不能报错** —— 报错会暴露「这个 id 存在」这个信息。
        """
        store = AskSessionStore()
        mine = store.create("E001")
        store.append(mine.session_id, "E001",
                     _turn("我有海鲜过敏", "注意避开鱼油"))

        other = store.get_or_create(mine.session_id, "E002")
        assert other.session_id != mine.session_id
        assert other.turns == [], "别人的历史被读到了"

    def test_append_to_other_session_rejected(self) -> None:
        store = AskSessionStore()
        mine = store.create("E001")
        ok = store.append(mine.session_id, "E002", _turn())
        assert ok is False
        assert store.get_or_create(mine.session_id, "E001").turns == []

    def test_reset_by_other_owner_rejected(self) -> None:
        store = AskSessionStore()
        mine = store.create("E001")
        store.append(mine.session_id, "E001", _turn())
        assert store.reset(mine.session_id, "E002") is False
        assert len(store.get_or_create(mine.session_id, "E001").turns) == 1


# ===================================================================
# 4. 新建与清空
# ===================================================================
class TestLifecycle:
    def test_empty_session_id_creates_new(self) -> None:
        store = AskSessionStore()
        s = store.get_or_create("", "E001")
        assert len(s.session_id) == 16

    def test_reset_clears_turns_but_keeps_id(self) -> None:
        """★ 「新对话」只清历史、不换 id。

        这样前端不用改本地存的值，也避免"点了新对话但旧请求还在飞"串话。
        """
        store = AskSessionStore()
        s = store.create("E001")
        store.append(s.session_id, "E001", _turn())
        assert store.reset(s.session_id, "E001") is True

        after = store.get_or_create(s.session_id, "E001")
        assert after.session_id == s.session_id
        assert after.turns == []

    def test_same_id_returns_same_session(self) -> None:
        store = AskSessionStore()
        s = store.create("E001")
        assert store.get_or_create(s.session_id, "E001").session_id == s.session_id


# ===================================================================
# 5. 历史渲染（prompt 用）
# ===================================================================
class TestRender:
    def test_empty_history_renders_empty(self) -> None:
        store = AskSessionStore()
        assert store.create("E001").render_history() == ""

    def test_render_contains_numbers_and_qa(self) -> None:
        store = AskSessionStore()
        s = store.create("E001")
        store.append(s.session_id, "E001",
                     _turn("鱼油有什么功能", "有助于维持血脂健康水平"))
        text = store.get_or_create(s.session_id, "E001").render_history()

        # ★ 必须带「第 N 轮」编号：用户追问会说「第 1 点」，
        #   编号能让模型把指代对上。
        assert "第 1 轮" in text
        assert "鱼油有什么功能" in text
        assert "有助于维持血脂" in text

    def test_refused_turn_is_marked(self) -> None:
        """拒答轮次也要记，否则模型不知道"上一轮没答上来"。"""
        store = AskSessionStore()
        s = store.create("E001")
        store.append(s.session_id, "E001",
                     make_turn("库里没有的东西", "资料里没有", refused=True))
        text = store.get_or_create(s.session_id, "E001").render_history()
        assert "未能回答" in text

    def test_newlines_flattened(self) -> None:
        """历史进 prompt 前要把换行压平，避免破坏分区结构。

        ★ 每轮固定输出两行（客户问 / 已答），所以 1 轮 = 1 个换行符。
          答案内部的 \\n 必须被替换掉，否则 prompt 的分区会被打断。
        """
        store = AskSessionStore()
        s = store.create("E001")
        store.append(s.session_id, "E001",
                     make_turn("q", "第一行\n第二行\n第三行"))
        text = store.get_or_create(s.session_id, "E001").render_history()
        assert text.count("\n") == 1, (
            f"1 轮应只有 1 个换行（两行之间），实际 {text.count(chr(10))} 个：{text!r}"
        )
        assert "第二行" in text, "换行被压平但内容不能丢"


# ===================================================================
# 6. 过期与容量
# ===================================================================
class TestEviction:
    def test_expired_session_replaced(self) -> None:
        store = AskSessionStore(ttl_sec=0)
        s = store.create("E001")
        store.append(s.session_id, "E001", _turn())
        # ttl=0 → 立即过期，应给出新会话
        again = store.get_or_create(s.session_id, "E001")
        assert again.turns == []

    def test_capacity_evicts_oldest(self) -> None:
        store = AskSessionStore(max_sessions=2)
        a = store.create("E001")
        b = store.create("E001")
        c = store.create("E001")
        assert store.count() <= 2
        # 最早的那个应被淘汰（拿到的是新会话）
        assert store.get_or_create(a.session_id, "E001").session_id != a.session_id
        assert store.get_or_create(c.session_id, "E001").session_id == c.session_id
