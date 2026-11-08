"""多轮会话存储单测（B2）。

★ 用假时钟推进 TTL，避免真等 30 分钟；并发用例用真实线程池
  验证「多个请求同时 append 不丢轮次」。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from application.services.conversation import ConversationStore, ConversationTurn


class _Clock:
    """可控时钟。"""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, sec: float) -> None:
        self.now += sec


def _turn(text: str) -> ConversationTurn:
    return ConversationTurn(
        question=text, rewritten=text, answer="（合规正文）", at=1000.0)


def _store(**kwargs) -> ConversationStore:
    kwargs.setdefault("ttl_sec", 1800)
    kwargs.setdefault("max_sessions", 200)
    kwargs.setdefault("max_turns", 3)
    return ConversationStore(**kwargs)


# ===================================================================
# TTL
# ===================================================================
class TestTTL:
    def test_expired_session_starts_from_turn_one(self) -> None:
        clock = _Clock()
        store = _store(ttl_sec=60, clock=clock)
        store.get_or_create("session01", "u1:manager")
        store.append("session01", _turn("鱼油怎么吃"))

        clock.advance(61)
        again = store.get_or_create("session01", "u1:manager")
        assert again.turns == []
        assert store.next_turn("session01") == 1
        assert again.created_at == clock.now

    def test_access_refreshes_last_active(self) -> None:
        clock = _Clock()
        store = _store(ttl_sec=60, clock=clock)
        store.get_or_create("session01", "u1:manager")
        store.append("session01", _turn("q1"))

        clock.advance(40)
        store.get_or_create("session01", "u1:manager")   # 刷新
        clock.advance(40)                               # 距上次访问仅 40s
        assert len(store.get_or_create("session01", "u1:manager").turns) == 1

    def test_append_after_expiry_does_not_raise(self) -> None:
        clock = _Clock()
        store = _store(ttl_sec=10, clock=clock)
        store.get_or_create("session01", "u1:manager")
        clock.advance(11)
        assert store.append("session01", _turn("q1")) == 1
        assert len(store.get_or_create("session01", "u1:manager").turns) == 1


# ===================================================================
# LRU 与身份隔离
# ===================================================================
class TestLRUAndIdentity:
    def test_lru_evicts_least_recently_used(self) -> None:
        clock = _Clock()
        store = _store(max_sessions=2, clock=clock)
        store.get_or_create("session01", "u1:manager")
        store.get_or_create("session02", "u1:manager")
        clock.advance(1)
        store.get_or_create("session01", "u1:manager")   # 触摸 s1 → s2 最久未用
        store.append("session01", _turn("q1"))

        clock.advance(1)
        store.get_or_create("session03", "u1:manager")
        assert store.count() == 2
        # s2 被淘汰；s1 的历史还在
        assert len(store.get_or_create("session01", "u1:manager").turns) == 1
        assert store.get_or_create("session02", "u1:manager").turns == []
        assert store.next_turn("session02") == 1

    def test_user_key_mismatch_is_a_new_session(self) -> None:
        store = _store()
        store.get_or_create("session01", "u1:manager")
        store.append("session01", _turn("鱼油怎么吃"))
        store.append("session01", _turn("那孕妇能吃吗"))

        other = store.get_or_create("session01", "u2:trainee")
        assert other.user_key == "u2:trainee"
        assert other.turns == []                 # 不能读到 u1 的上下文
        assert store.next_turn("session01") == 1
        assert store.append("session01", _turn("新问题")) == 1


# ===================================================================
# 轮次截断 / 并发
# ===================================================================
class TestTurnsAndConcurrency:
    def test_keeps_only_last_three_turns(self) -> None:
        store = _store(max_turns=3)
        store.get_or_create("session01", "u1:manager")
        for i in range(1, 6):
            assert store.append("session01", _turn(f"q{i}")) == i

        turns = store.get_or_create("session01", "u1:manager").turns
        assert len(turns) == 3
        assert [t.question for t in turns] == ["q3", "q4", "q5"]
        assert store.next_turn("session01") == 6   # 响应里的轮次号不截断

    def test_concurrent_append_does_not_lose_turns(self) -> None:
        store = _store(max_turns=50)
        store.get_or_create("session01", "u1:manager")
        n = 24

        with ThreadPoolExecutor(max_workers=8) as pool:
            turns = list(pool.map(
                lambda i: store.append("session01", _turn(f"q{i}")),
                range(n)))

        assert sorted(turns) == list(range(1, n + 1))   # 轮次号唯一且连续
        stored = store.get_or_create("session01", "u1:manager").turns
        assert {t.question for t in stored} == {f"q{i}" for i in range(n)}
        assert store.next_turn("session01") == n + 1

    def test_reset_clears_everything(self) -> None:
        store = _store()
        store.get_or_create("session01", "u1:manager")
        store.append("session01", _turn("q1"))
        store.reset()
        assert store.count() == 0
        assert store.next_turn("session01") == 1
        assert store.get_or_create("session01", "u1:manager").turns == []

    def test_snapshot_is_not_shared(self) -> None:
        store = _store()
        snapshot = store.get_or_create("session01", "u1:manager")
        snapshot.turns.append(_turn("不该写回存储"))
        assert store.get_or_create("session01", "u1:manager").turns == []
