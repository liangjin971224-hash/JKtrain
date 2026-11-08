"""多轮问答会话（docs/TDD_多轮问答_2026-11-08.md §2）。

★ 与演练会话（`interfaces/http/session_store.py`）同样的取舍：
  短会话、低并发、重启可丢，所以放内存 + LRU + TTL，不引入外部服务。

★ 只存「最近 3 轮的问答摘要」，**不存完整历史、不存 annotation.segments**：
  · 第一版改写只读上一轮实体，更多轮收益低、噪声大（P1-b 教训）；
  · `annotation.segments` 含违规原文，按接口层约定不得落日志/入库，
    会话里只保留 `answer`（合规正文或拒答话术）。

★ 线程安全：FastAPI 的同步接口跑在线程池里，`get_or_create` / `append`
  可能被同一会话的多个请求同时调用，所有读写都在同一把锁内完成。
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

#: 每个会话最多保留几轮（FIFO，见 TDD §2.2）。
DEFAULT_MAX_TURNS = 3
#: 进程内最多同时保留多少个会话。
DEFAULT_MAX_SESSIONS = 200
#: 会话空闲多久算过期（秒，30 分钟）。
DEFAULT_TTL_SEC = 1800


@dataclass(frozen=True)
class ConversationTurn:
    """一轮问答。

    `question` 是用户原话，`rewritten` 是实际进检索/生成的问题；
    两者分开存，排障时才看得出「改写有没有把问题带偏」。
    """

    question: str
    rewritten: str
    answer: str
    cited_chunk_ids: tuple[str, ...] = ()
    product_ids: tuple[str, ...] = ()
    intent: str = ""
    refused: bool = False
    at: float = field(default_factory=time.time)


@dataclass
class ConversationSession:
    """一次会话。

    `user_key` 由接口层按「工号:角色」生成；不匹配时按新会话处理，
    保证 A 用户的上下文不会被 B 用户读到。
    """

    session_id: str
    user_key: str
    turns: list[ConversationTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class ConversationStore:
    """进程内会话存储：`OrderedDict` + `threading.Lock`，LRU + TTL。

    ★ 为什么还要单独一个轮次计数器：
      `turns` 会按 `max_turns` 截断，不能再用 `len(turns)` 回答
      「当前是第几轮」。响应里的 `turn` 是 1-based 且单调递增的，
      所以用 `_turn_seq` 单独记，随会话一起淘汰/重置。
    """

    def __init__(
        self,
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_turns: int = DEFAULT_MAX_TURNS,
        ttl_sec: int = DEFAULT_TTL_SEC,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """
        Args:
            max_sessions: 最多保留多少个会话，超出按最久未访问淘汰。
            max_turns: 每个会话最多保留几轮（FIFO）。
            ttl_sec: 空闲多久算过期（秒）。
            clock: 时间源。测试可注入假时钟，避免真的等 30 分钟。
        """
        self._sessions: OrderedDict[str, ConversationSession] = OrderedDict()
        self._turn_seq: dict[str, int] = {}
        self._lock = threading.Lock()
        self._max_sessions = max(1, int(max_sessions))
        self._max_turns = max(1, int(max_turns))
        self._ttl = max(1, int(ttl_sec))
        self._clock = clock

    # ---------------- 读写 ----------------
    def get_or_create(
        self, session_id: str, user_key: str
    ) -> ConversationSession:
        """取会话；不存在 / 已过期 / 身份不匹配则新建。

        Returns:
            会话**快照**（`turns` 是新列表）。调用方拿到的对象不会被
            其他线程并发改写，改写器可以放心读上一轮。

        Note:
            身份不匹配时丢弃旧记录再新建：既不泄露旧会话，也不会把
            两个用户的轮次混在一起（TDD §5）。
        """
        sid = str(session_id or "")
        with self._lock:
            now = self._clock()
            session = self._sessions.get(sid)
            if session is not None and (
                self._expired(session, now) or session.user_key != user_key
            ):
                self._sessions.pop(sid, None)
                self._turn_seq.pop(sid, None)
                session = None

            if session is None:
                session = ConversationSession(
                    session_id=sid, user_key=user_key,
                    created_at=now, last_active=now,
                )
                self._sessions[sid] = session
                self._turn_seq[sid] = 0
                self._evict_locked(now)
            else:
                session.last_active = now
                self._sessions.move_to_end(sid)     # LRU：最近访问的挪到尾部

            return self._snapshot(session)

    def append(self, session_id: str, turn: ConversationTurn) -> int:
        """追加一轮，返回本轮的 1-based 轮次号；失败返回 0。

        ★ 会话已过期 / 已被淘汰时不抛异常：新建会话只保留这一轮。
          接口层把这当成「降级」，不影响本轮回答本身。
        """
        sid = str(session_id or "")
        with self._lock:
            now = self._clock()
            session = self._sessions.get(sid)
            if session is None or self._expired(session, now):
                old_key = session.user_key if session is not None else ""
                session = ConversationSession(
                    session_id=sid, user_key=old_key,
                    created_at=now, last_active=now,
                )
                self._sessions[sid] = session
                self._turn_seq[sid] = 0
                self._evict_locked(now)

            session.turns.append(turn)
            if len(session.turns) > self._max_turns:
                del session.turns[:len(session.turns) - self._max_turns]
            session.last_active = now
            self._sessions.move_to_end(sid)
            self._turn_seq[sid] = self._turn_seq.get(sid, 0) + 1
            return self._turn_seq[sid]

    def next_turn(self, session_id: str) -> int:
        """下一次 `append` 会得到的轮次号（流式响应在 done 前就要填 turn）。"""
        sid = str(session_id or "")
        with self._lock:
            return self._turn_seq.get(sid, 0) + 1

    def count(self) -> int:
        """当前会话数（观测/测试用）。"""
        with self._lock:
            return len(self._sessions)

    def reset_session(self, session_id: str, user_key: str) -> bool:
        """清空某会话的历史，**保留 session_id**（用户点「新对话」）。

        ★ 为什么不换新 id：前端 localStorage 里存着旧值，
          换 id 会让「点了新对话但旧请求还在飞」时串话
          （主干 `new_conversation` 语义，2026-11-08 合并保留）。

        ★ 只清自己的：身份不匹配是别人的记录，不动，返回 False。
        """
        sid = str(session_id or "")
        with self._lock:
            session = self._sessions.get(sid)
            if session is None or session.user_key != user_key:
                return False
            session.turns.clear()
            session.last_active = self._clock()
            self._sessions.move_to_end(sid)
            self._turn_seq[sid] = 0
            return True

    def reset(self) -> None:
        """清空所有会话（供测试隔离用）。"""
        with self._lock:
            self._sessions.clear()
            self._turn_seq.clear()

    # ---------------- 内部 ----------------
    def _expired(self, session: ConversationSession, now: float) -> bool:
        return (now - session.last_active) > self._ttl

    def _evict_locked(self, now: float) -> None:
        """先清过期、再按 LRU 淘汰。调用方必须已持锁。"""
        for sid in [s for s, r in self._sessions.items()
                    if self._expired(r, now)]:
            self._sessions.pop(sid, None)
            self._turn_seq.pop(sid, None)

        while len(self._sessions) > self._max_sessions:
            sid, _ = self._sessions.popitem(last=False)
            self._turn_seq.pop(sid, None)
            logger.warning("多轮会话超出上限，淘汰最久未用: ...%s", sid[-4:])

    @staticmethod
    def _snapshot(session: ConversationSession) -> ConversationSession:
        return ConversationSession(
            session_id=session.session_id,
            user_key=session.user_key,
            turns=list(session.turns),
            created_at=session.created_at,
            last_active=session.last_active,
        )


__all__ = [
    "ConversationSession",
    "ConversationStore",
    "ConversationTurn",
    "DEFAULT_MAX_SESSIONS",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_TTL_SEC",
]
