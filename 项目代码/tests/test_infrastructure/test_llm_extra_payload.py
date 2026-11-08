"""LLM 适配器的附加请求字段（P1-c 之外的运行时修复）。

守住的语义：
    1. `extra_payload` 会原样合并进 generate / generate_stream 的请求体；
    2. 不传时不产生多余字段（不影响普通模型）；
    3. 关闭思考的参数用 `enable_thinking: false`——推理模型不关时
       qwen3.7-flash 约 14s/1200 tokens，且 max_tokens 小时正文可能为空。
"""
from __future__ import annotations

from infrastructure.llm.openai_compatible_adapter import OpenAICompatibleAdapter


def _adapter(extra=None) -> OpenAICompatibleAdapter:
    return OpenAICompatibleAdapter(
        name="qwen", api_key="k", base_url="https://example.com/v1",
        model="qwen3.7-flash", extra_payload=extra,
    )


def test_generate_merges_extra_payload(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(path, payload):
        captured.update(payload)
        return {
            "model": payload["model"],
            "choices": [{"message": {"content": "收到"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    a = _adapter({"enable_thinking": False})
    monkeypatch.setattr(a, "_post", fake_post)
    r = a.generate("你好")

    assert r.text == "收到"
    assert captured["enable_thinking"] is False
    assert captured["model"] == "qwen3.7-flash"


def test_generate_without_extra_has_no_extra_field(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(path, payload):
        captured.update(payload)
        return {"model": "m", "choices": [{"message": {"content": "ok"}}],
                "usage": {}}

    a = _adapter(None)
    monkeypatch.setattr(a, "_post", fake_post)
    a.generate("你好")

    assert "enable_thinking" not in captured


def test_stream_merges_extra_payload(monkeypatch) -> None:
    captured: dict = {}

    def fake_stream(payload):
        captured.update(payload)
        yield "片段"

    a = _adapter({"enable_thinking": False})
    monkeypatch.setattr(a, "_post_stream", fake_stream)
    out = list(a.generate_stream("你好"))

    assert out == ["片段"]
    assert captured["enable_thinking"] is False
    assert captured["stream"] is True
