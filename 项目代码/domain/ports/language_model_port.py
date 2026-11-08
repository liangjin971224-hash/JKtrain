"""语言模型端口。

只声明两项能力：意图分类、文本生成。
Prompt 组装属于上层（application/agents），端口只管「把 prompt 发给模型、拿回文本」。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """一次生成的结果。

    `finish_reason` 用于判断是否被截断（length 时需要告警，不能当完整答案用）。
    """

    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"


class LanguageModelPort(ABC):
    """语言模型端口。"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """生成文本。

        ★ temperature 默认 0.0：本项目的回答必须稳定可复现，
          合规场景下不允许「每次说法略有不同」。

        Args:
            prompt: 用户侧提示（已由上层的 prompt builder 组装好）。
            system_prompt: 系统提示（角色与硬约束）。
            temperature: 采样温度。
            max_tokens: 输出上限。

        Returns:
            生成结果。

        Raises:
            LLMResponseFormatError: 模型返回无法解析。
            LLMCallError: 调用失败（网络/鉴权/限流）。
        """

    @abstractmethod
    def classify(
        self,
        text: str,
        labels: list[str],
        *,
        allow_unknown: bool = True,
    ) -> str:
        """把文本分到给定标签之一（意图分类）。

        Args:
            text: 用户问题。
            labels: 候选标签，如 ["产品功效", "禁忌咨询", "要停药", "要诊断"]。
            allow_unknown: 是否允许返回 "unknown"。

        Returns:
            labels 中的一项；allow_unknown=True 且模型无法判断时返回 "unknown"。
        """

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """流式生成文本，逐个产出文本增量。

        ★ 为什么要有这个方法：
          非流式一次等 3–5s（LLM 生成耗时），P95 无法达标。
          流式下首句 0.5–1s 就能到用户手上。

        ★ 但不是「收到就发」：
          输出护栏是必经节点（约束 #1），未过护栏的内容不得展示。
          调用方应配合 `domain.services.stream_guard.StreamGuard` 使用：
          本方法产出原始增量，StreamGuard 负责分句 + 过护栏后再放行。

        ★ 实现约定：不做流式的适配器（如测试替身）
          只需 `yield self.generate(...).text` 退化为整段产出。
          这样能力完整，只是拿不到延迟收益。

        Args:
            prompt: 用户侧提示。
            system_prompt: 系统提示。
            temperature: 采样温度。
            max_tokens: 输出上限。

        Yields:
            文本增量（可能是几个字，也可能是整段）。

        Raises:
            LLMCallError: 调用失败。
            LLMResponseFormatError: 返回无法解析。
        """

    @abstractmethod
    def is_available(self) -> bool:
        """模型是否可用（Key 是否配置、服务是否连通）。

        启动自检用它决定「告警」还是「报错」：
        LLM 不可用只告警，不阻塞启动（P0 阶段还不需要生成能力）。
        """
