"""应用层基础类型。

P0 只定义「用例返回什么」，不实现任何用例。
用例本身从 P1 开始按阶段添加。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.ports import GenerationResult


@dataclass(frozen=True, slots=True)
class UseCaseResult:
    """用例的统一返回结构。

    ★ `refused` 与 `answer` 的关系要分清：
       refused=True  表示「召回到不了，系统主动拒答」，此时 LLM 不应被调用；
       refused=False 表示「正常回答」，即使内容是「不建议使用」也不算拒答。

    两种都可能在 answer 里出现「不可以」，语义完全不同，不能混。
    """

    answer: str
    refused: bool = False
    # 拒答原因（refused=True 时必填，便于排障与日志）
    refusal_reason: str = ""
    # 命中的知识块 id（可追溯性：评委问"凭什么这么说"要能答）
    cited_chunk_ids: tuple[str, ...] = ()
    # 命中的产品 id
    product_ids: tuple[str, ...] = ()
    # 结构化附加信息（意图、各路命中数、耗时等）
    meta: dict[str, Any] = field(default_factory=dict)

    #: 违规内容标注（P5 后新增）。
    #:
    #: ★ 只有当输出被护栏拦过才非 None。
    #:   前端据此把违规句标红并展示原因，其余合规内容正常显示。
    #:
    #: ★ 安全约束：
    #:   `answer` 一定是**剔除违规句后的合规版本**，可直接对外；
    #:   `annotation.segments` 里含违规原文，**只应喂给前端展示**，
    #:   不要写日志、不要当作可复制正文。
    annotation: Any = None

    @property
    def has_violation_annotation(self) -> bool:
        """是否有违规标注可供展示。"""
        return self.annotation is not None and getattr(
            self.annotation, "has_violation", False)

    def with_meta(self, **kwargs: Any) -> UseCaseResult:
        """追加元信息，返回新实例（保持不可变）。"""
        return UseCaseResult(
            answer=self.answer,
            refused=self.refused,
            refusal_reason=self.refusal_reason,
            cited_chunk_ids=self.cited_chunk_ids,
            product_ids=self.product_ids,
            meta={**self.meta, **kwargs},
        )


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """流式输出的单个事件。

    ★ 为什么用事件而不是裸字符串：
      流式除了正文，还要传「结束」「被拦」「溯源」等信息。
      裸字符串无法区分「内容是空的」和「流结束了」。

    两类事件：
        kind="text"  正文增量，`text` 有值
        kind="done"  结束事件，带上 refused / refusal_reason / meta
    """

    kind: str                       # "text" | "done"
    text: str = ""
    refused: bool = False
    refusal_reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("text", "done"):
            raise ValueError(f"非法事件类型: {self.kind}")


@dataclass(frozen=True, slots=True)
class QuestionIntent:
    """查询意图分类结果。

    7 类意图决定了三路召回的权重（见 retrieval-and-agent.md）。
    """

    label: str
    confidence: float = 0.0
    # 是否属于「必须拦或必须提示」的安全类意图
    is_safety_sensitive: bool = False

    #: 安全类意图。这些查询里图谱权重要上调（0.6），宁可不精确召回也不能漏禁忌。
    SAFETY_LABELS = (
        "禁忌咨询", "特殊人群", "要停药", "要诊断", "要处方", "超量服用",
    )

    @classmethod
    def of(cls, label: str, confidence: float = 0.0) -> QuestionIntent:
        return cls(
            label=label,
            confidence=confidence,
            is_safety_sensitive=any(k in label for k in cls.SAFETY_LABELS),
        )


__all__ = ["GenerationResult", "QuestionIntent", "StreamEvent", "UseCaseResult"]
