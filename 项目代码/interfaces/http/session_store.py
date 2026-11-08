"""演练会话存储（进程内）。

────────────────────────────────────────────────────────────────────
为什么先放内存
────────────────────────────────────────────────────────────────────
演练是**短会话**（3–5 分钟），并发低（培训场景几十人）。
内存最省事，且不需要额外服务。

代价：**进程重启后会话全丢**。
对培训场景可接受——学员重开一个演练即可。

★ 什么情况下必须换实现：
    · 多实例部署（负载均衡打到不同进程）
    · 需要「练到一半关浏览器，明天继续」
  那时换 Redis（CachePort 已就绪，只需换个适配器）。

────────────────────────────────────────────────────────────────────
为什么做容量上限
────────────────────────────────────────────────────────────────────
不做上限的话，长期运行会内存泄漏（每个会话含完整对话历史）。
这里用「超过 N 个就淘汰最久未访问的」——
比定时清理简单，且不会出现「正在练的被清掉」。
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from application.use_cases.drill_session import DrillSession
from domain.value_objects import CaseSpec

logger = logging.getLogger(__name__)

#: 最多同时保留多少个会话。
#: 培训场景同时在线不会超过这个数；超了淘汰最久没动的。
DEFAULT_MAX_SESSIONS = 200

#: 会话空闲多久算过期（秒）。30 分钟没动就清掉。
DEFAULT_TTL_SEC = 30 * 60


@dataclass
class SessionRecord:
    """一个演练会话的完整状态。"""

    session_id: str
    spec: CaseSpec
    session: DrillSession
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_access = time.time()


class DrillSessionStore:
    """演练会话存储。线程安全（FastAPI 可能用线程池跑同步接口）。"""

    def __init__(
        self,
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        ttl_sec: int = DEFAULT_TTL_SEC,
    ) -> None:
        self._records: OrderedDict[str, SessionRecord] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_sessions
        self._ttl = ttl_sec

    # ---------------- 增删查 ----------------
    def create(self, spec: CaseSpec, *, trainee_id: str = "") -> SessionRecord:
        """新建会话。

        Args:
            spec: 案例规格。
            trainee_id: 学员编号。

        Returns:
            新建的会话记录。
        """
        session_id = uuid.uuid4().hex[:16]
        record = SessionRecord(
            session_id=session_id,
            spec=spec,
            session=DrillSession(spec, trainee_id=trainee_id),
        )

        with self._lock:
            self._records[session_id] = record
            self._evict_if_needed()

        logger.info(
            "新建演练会话: %s case=%s trainee=%s",
            session_id, spec.case_id, trainee_id or "(未填)")
        return record

    def get(self, session_id: str) -> SessionRecord:
        """取会话。

        Raises:
            KeyError: 会话不存在或已过期。
        """
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                raise KeyError(f"会话不存在或已过期: {session_id}")

            if self._is_expired(record):
                del self._records[session_id]
                raise KeyError(f"会话已过期: {session_id}")

            record.touch()
            self._records.move_to_end(session_id)   # LRU
            return record

    def drop(self, session_id: str) -> None:
        """删除会话（评阅完可以清掉）。"""
        with self._lock:
            self._records.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    # ---------------- 内部 ----------------
    def _is_expired(self, record: SessionRecord) -> bool:
        return (time.time() - record.last_access) > self._ttl

    def _evict_if_needed(self) -> None:
        """超出容量或过期时淘汰。调用方必须已持锁。"""
        # 先清过期
        for sid in [s for s, r in self._records.items() if self._is_expired(r)]:
            del self._records[sid]

        # 再按 LRU 淘汰
        while len(self._records) > self._max:
            sid, _ = self._records.popitem(last=False)
            logger.warning("演练会话超出上限，淘汰最久未用: %s", sid)


#: 进程级单例（与会话一起活）
_store = DrillSessionStore()


def get_session_store() -> DrillSessionStore:
    return _store


__all__ = [
    "DrillSessionStore",
    "SessionRecord",
    "get_session_store",
]
