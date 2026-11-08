"""问答多轮会话存储（进程内）。

────────────────────────────────────────────────────────────────────
它解决的问题
────────────────────────────────────────────────────────────────────
现在每次提问都是**孤立问题**：模型不知道「第 1 点呢」指的是上一轮的第 1 点。
用户想追问只能重新开一个框、把前文重打一遍。

────────────────────────────────────────────────────────────────────
★ 只存「对话摘要」，不存全量回答
────────────────────────────────────────────────────────────────────
一轮里存三样：
    · 用户问题原文
    · 答案的**前 N 字**（`ANSWER_HEAD_CHARS`）
    · 涉及的产品 ID 与意图

不存：完整答案、引用块、annotation。

理由：
    · prompt 有限。三轮全量答案轻松上万字，追问本来只需要「上一轮聊了什么」
    · 大部分答案的细节与追问无关，真正需要的是**话题与结论**
    · 存 annotation 尤其危险：里面有违规原文（供前端标红用），
      若回灌进 prompt，等于**上一轮的违规表述被当成例文教给模型**。
      所以这里只吃 `answer`（合规版），调用方也**不许**把 annotation 传进来。

────────────────────────────────────────────────────────────────────
为什么和演练会话分开
────────────────────────────────────────────────────────────────────
`session_store.py` 存的是 `DrillSession`（案例状态机，有轮次、推进逻辑），
结构完全不同。硬塞在一起会让两类会话互相污染。
但**机制照搬**：内存 + LRU + TTL + owner 校验。

★ 为什么必须做 owner 校验（演练那边踩过）：
    session_id 是 16 位随机十六进制，不好猜，但**不是权限控制**。
    只要知道 id 就能读到别人的对话内容（可能含客户隐私：过敏史、用药）。
    所以存的时候记 owner，读的时候比对，不一致按「不存在」处理
    —— 不暴露「这个 id 存在但不属于你」这个信息。
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: 最多保留多少轮（一问一答算一轮）。
#:
#: ★ 3 轮是权衡后的值：
#:   1 轮 → 追问两三次就接不上了；
#:   5 轮以上 → prompt 显著变长，且越早的内容对当前追问的贡献越小。
#:   实测培训场景的追问链很少超过 3 层。
DEFAULT_MAX_TURNS = 3

#: 每条答案只留前多少字进历史。
#:
#: ★ 为什么是 200：够模型判断「上一轮在讲什么、结论是什么」，
#:   又不至于把整段话术塞进去。追问真正需要的是话题，不是原文细节。
ANSWER_HEAD_CHARS = 200

#: 最多同时保留多少会话。培训场景同时在线远小于这个数。
DEFAULT_MAX_SESSIONS = 500

#: 会话空闲多久过期（秒）。30 分钟没动就清掉。
#:
#: ★ 与前端「手动新对话」配合：用户不点新对话时，
#:   半小时后自动失效，避免一直沿用很久以前的上下文。
DEFAULT_TTL_SEC = 30 * 60


@dataclass(frozen=True, slots=True)
class AskTurn:
    """一轮问答的历史摘要。

    ★ 注意字段都是**摘要**，不是原文全量。
    """

    question: str
    answer_head: str
    product_ids: tuple[str, ...] = ()
    intent: str = ""
    #: 该轮是否被拒答（拒答轮次也要记，否则模型不知道"上次没答上来"）
    refused: bool = False


@dataclass
class AskSession:
    """一个问答会话。"""

    session_id: str
    owner: str
    turns: list[AskTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_access = time.time()

    def add(self, turn: AskTurn, max_turns: int) -> None:
        """追加一轮，超出上限丢最早的。"""
        self.turns.append(turn)
        if len(self.turns) > max_turns:
            # 只保留最近的 N 轮
            del self.turns[:len(self.turns) - max_turns]

    def render_history(self) -> str:
        """把历史渲染成给 prompt 用的文本。

        Returns:
            多行文本；无历史时返回空串。

        ★ 刻意标上「第 N 轮」编号，因为用户追问会说「第 1 点」——
          编号能让模型把指代对上。注意这里说的是「轮」，不是「点」：
          「点」是那一轮答案内部的条目，模型看答案摘要自己数。
        """
        if not self.turns:
            return ""
        lines: list[str] = []
        for i, t in enumerate(self.turns, 1):
            head = (t.answer_head or "").strip().replace("\n", " ")
            if t.refused:
                head = "（上一轮未能回答）" + head
            lines.append(f"第 {i} 轮 · 客户问：{t.question}")
            lines.append(f"第 {i} 轮 · 已答：{head}")
        return "\n".join(lines)

    def last_product_ids(self) -> tuple[str, ...]:
        """最近一轮涉及的产品 ID（倒着找第一个非空的）。

        ★ 为什么取「最近一个非空的」而不是「只看最后一轮」：
          用户可能这样问：
              第 1 轮：鱼油有什么功能        → P001
              第 2 轮：那适合什么人吃        → 产品为空（纯追问，没提产品名）
              第 3 轮：那有什么禁忌          → 仍然指鱼油
          如果只看最后一轮（第 2 轮为空），第 3 轮就接不上了。
        """
        for t in reversed(self.turns):
            if t.product_ids:
                return t.product_ids
        return ()


class AskSessionStore:
    """问答会话存储。线程安全。"""

    def __init__(
        self,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        ttl_sec: int = DEFAULT_TTL_SEC,
    ) -> None:
        self._sessions: OrderedDict[str, AskSession] = OrderedDict()
        self._lock = threading.Lock()
        self._max_turns = max(1, max_turns)
        self._max_sessions = max_sessions
        self._ttl = ttl_sec

    @property
    def max_turns(self) -> int:
        return self._max_turns

    # ---------------- 增删查 ----------------
    def create(self, owner: str) -> AskSession:
        """新建会话。"""
        sid = uuid.uuid4().hex[:16]
        sess = AskSession(session_id=sid, owner=owner)
        with self._lock:
            self._sessions[sid] = sess
            self._evict_if_needed()
        logger.info("新建问答会话: %s owner=%s", sid, owner)
        return sess

    def get_or_create(
        self, session_id: str, owner: str
    ) -> AskSession:
        """取会话；不存在或不属于本人则新建一个。

        ★ 为什么"不属于本人"也新建而不是报错：
          前端可能存了别人的/过期的 id（比如换账号登录后 localStorage 没清）。
          报错会让用户看到一个莫名其妙的失败；新建则是静默开始新对话，
          而且**不会泄露原会话内容**（这是关键：不暴露 id 是否存在）。
        """
        sid = (session_id or "").strip()
        if not sid:
            return self.create(owner)

        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None:
                pass                              # 落到下面新建
            elif sess.owner != owner:
                logger.warning(
                    "会话 %s 的 owner 不匹配（请求=%s 实际=%s），按新会话处理",
                    sid, owner, sess.owner)
                sess = None
            elif (time.time() - sess.last_access) >= self._ttl:
                # ★ 用 >=：ttl=0 表示"立即过期"。
                #   用 > 的话 ttl=0 永远不成立，测试里想验证过期就得等真实时间。
                del self._sessions[sid]
                sess = None
            else:
                sess.touch()
                self._sessions.move_to_end(sid)   # LRU
                return sess

        # 锁外建，避免建的时候还持锁（create 里会再加锁）
        return self.create(owner)

    def append(
        self,
        session_id: str,
        owner: str,
        turn: AskTurn,
    ) -> bool:
        """给会话追加一轮。

        Returns:
            True = 追加成功；False = 会话不存在/不属于本人（**静默忽略**）。

        ★ 为什么要静默：写历史失败不该让整个问答失败。
          学员拿到了答案，只是这一轮没记进上下文 —— 可以接受。
        """
        sid = (session_id or "").strip()
        if not sid:
            return False
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None or sess.owner != owner:
                return False
            sess.add(turn, self._max_turns)
            sess.touch()
            self._sessions.move_to_end(sid)
            return True

    def reset(self, session_id: str, owner: str) -> bool:
        """清空一个会话的历史（用户点「新对话」）。

        Returns:
            True = 已清空；False = 会话不存在或不属于本人。
        """
        sid = (session_id or "").strip()
        if not sid:
            return False
        with self._lock:
            sess = self._sessions.get(sid)
            if sess is None or sess.owner != owner:
                return False
            sess.turns.clear()
            sess.touch()
            return True

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ---------------- 内部 ----------------
    def _evict_if_needed(self) -> None:
        """清过期 + 按 LRU 淘汰。调用方必须已持锁。"""
        now = time.time()
        for sid in [s for s, v in self._sessions.items()
                    if (now - v.last_access) >= self._ttl]:
            del self._sessions[sid]
        while len(self._sessions) > self._max_sessions:
            sid, _ = self._sessions.popitem(last=False)
            logger.warning("问答会话超出上限，淘汰最久未用: %s", sid)


def make_turn(
    question: str,
    answer: str,
    *,
    product_ids: tuple[str, ...] = (),
    intent: str = "",
    refused: bool = False,
) -> AskTurn:
    """从一次问答结果构造历史轮次。

    ★ `answer` 必须是**合规版正文**（`AskResponse.answer`），
      **绝不能**传 annotation 里的 segments ——
      那里含违规原文，回灌进 prompt 等于教模型说错话。
    """
    head = (answer or "").strip()[:ANSWER_HEAD_CHARS]
    return AskTurn(
        question=(question or "").strip(),
        answer_head=head,
        product_ids=tuple(product_ids or ()),
        intent=intent or "",
        refused=refused,
    )


#: 进程级单例
_store = AskSessionStore()


def get_ask_session_store() -> AskSessionStore:
    return _store


__all__ = [
    "ANSWER_HEAD_CHARS",
    "DEFAULT_MAX_TURNS",
    "AskSession",
    "AskSessionStore",
    "AskTurn",
    "get_ask_session_store",
    "make_turn",
]
