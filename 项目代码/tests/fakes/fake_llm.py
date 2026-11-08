"""语言模型端口的测试替身。

★ 本文件的核心是 `FailOnCallLLM`：
   它一旦被调用就抛异常，用来**证明某条路径完全不依赖 LLM**。
   这是本项目最重要的一条架构约束（召回为空时必须拒答，不能让 LLM 编）。
"""
from __future__ import annotations

from collections.abc import Iterator

from domain.ports import GenerationResult, LanguageModelPort


class FakeLanguageModelPort(LanguageModelPort):
    """返回固定文本的假模型。"""

    def __init__(
        self,
        *,
        reply: str = "（假模型回复）",
        label: str = "unknown",
        available: bool = True,
    ) -> None:
        self._reply = reply
        self._label = label
        self._available = available
        self.calls: list[str] = []
        self.prompts: list[str] = []       # 便于断言 prompt 里是否漏了警示语

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        self.calls.append("generate")
        self.prompts.append(prompt)
        return GenerationResult(
            text=self._reply,
            model="fake-llm",
            prompt_tokens=len(prompt),
            completion_tokens=len(self._reply),
            finish_reason="stop",
        )

    def classify(
        self,
        text: str,
        labels: list[str],
        *,
        allow_unknown: bool = True,
    ) -> str:
        self.calls.append("classify")
        if self._label in labels:
            return self._label
        return labels[0] if labels else "unknown"

    def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """退化为整段产出。

        测试替身不做真流式——真正要验证的是「分句 + 护栏」逻辑，
        那个在 `StreamGuard` 里单独测（见 `test_stream_guardrail.py`）。
        """
        self.calls.append("generate_stream")
        result = self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if result.text:
            yield result.text

    def is_available(self) -> bool:
        return self._available


class FailOnCallLLM(LanguageModelPort):
    """一旦被调用就抛 AssertionError。

    用于证明「无召回 → 拒答」这条路径完全不经过 LLM。
    如果这条测试失败，说明控制流走错了——出现了幻觉风险。
    """

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        raise AssertionError(
            "本路径不应调用 LLM！召回为空时必须直接拒答，不能让模型凭空生成。"
        )

    def classify(
        self,
        text: str,
        labels: list[str],
        *,
        allow_unknown: bool = True,
    ) -> str:
        raise AssertionError("本路径不应调用 LLM！")

    def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        raise AssertionError(
            "本路径不应调用 LLM！召回为空时必须直接拒答，不能让模型凭空生成。"
        )

    def is_available(self) -> bool:
        return True
