"""OpenAI 兼容协议的 LLM 适配器。

一份代码支持两家：
    · 千问（DashScope OpenAI 兼容模式）  base_url = .../compatible-mode/v1
    · DeepSeek                          base_url = https://api.deepseek.com
两者的 `POST {base_url}/chat/completions` 请求/响应格式一致。

────────────────────────────────────────────────────────────────────
为什么不用 openai SDK
────────────────────────────────────────────────────────────────────
1. 少一个依赖：本模块只用标准库 `urllib`，不引入 SDK 版本兼容问题
2. 错误分类更可控：需要精确区分「超时 / 限流 / 5xx / 鉴权 / 格式错」，
   SDK 的异常层级随版本变化，自己判 HTTP 状态码更稳
3. 本项目调用量小（培训场景，非高并发），不需要连接池与异步

────────────────────────────────────────────────────────────────────
温度为什么默认 0.0
────────────────────────────────────────────────────────────────────
合规场景不允许「每次说法略有不同」。同一个问题必须得到稳定的回答，
否则同一个违规表述可能这次拦住了、下次没拦住，无法审计。
"""
from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator

from domain.ports import GenerationResult, LanguageModelPort
from infrastructure.exceptions import LLMCallError, LLMResponseFormatError

logger = logging.getLogger(__name__)


class OpenAICompatibleAdapter(LanguageModelPort):
    """OpenAI 兼容协议适配器（千问 / DeepSeek 通用）。"""

    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str,
        model: str,
        timeout_sec: int = 30,
        extra_payload: dict | None = None,
    ) -> None:
        """
        Args:
            name: 端点标识（qwen / deepseek），用于日志与统计。
            api_key: API Key。
            base_url: 形如 https://api.deepseek.com（不带 /chat/completions）。
            model: 模型名。
            timeout_sec: 单次调用超时。
            extra_payload: 附加到请求体的字段（原样合并）。
                ★ 用于推理模型的 `enable_thinking: false`：不关思考时
                  qwen3.7-flash 单次生成可到 14s / 1200+ tokens，
                  甚至把 max_tokens 全用在思考上、正文为空。
        """
        self._name = name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_sec
        self._extra = dict(extra_payload or {})

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    # ---------------- 端口实现 ----------------
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(self._extra)
        data = self._post("/chat/completions", payload)

        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseFormatError(
                f"返回体缺少 choices[0].message.content | endpoint={self._name}"
            ) from exc

        if not isinstance(text, str):
            raise LLMResponseFormatError(
                f"content 不是字符串: {type(text).__name__} | endpoint={self._name}")

        usage = data.get("usage") or {}
        return GenerationResult(
            text=text,
            model=data.get("model", self._model),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            # ★ length 表示被 max_tokens 截断，上层必须当作不完整答案处理
            finish_reason=choice.get("finish_reason") or "stop",
        )

    def classify(
        self,
        text: str,
        labels: list[str],
        *,
        allow_unknown: bool = True,
    ) -> str:
        """把文本分到给定标签之一。

        用强约束提示词 + 只输出标签名，temperature=0 保证可复现。
        解析失败时按 allow_unknown 返回 unknown，而不是猜一个——
        猜错会导致三路召回权重错配（P3 的权重表按意图查）。
        """
        if not labels:
            return "unknown"

        label_list = "、".join(labels)
        unknown_hint = f"如果无法判断，只输出 unknown。" if allow_unknown else ""
        prompt = (
            f"把下面这句话归类到给定标签之一。\n"
            f"标签：{label_list}\n"
            f"{unknown_hint}\n"
            f"只输出标签本身，不要解释、不要加引号、不要加标点。\n\n"
            f"句子：{text}"
        )
        result = self.generate(prompt, temperature=0.0, max_tokens=16)
        raw = result.text.strip().strip('"\'').strip()

        if raw in labels:
            return raw
        if allow_unknown and raw.lower() == "unknown":
            return "unknown"

        # 模型可能输出了「标签：xxx」或带标点，做一次宽松匹配
        for label in labels:
            if label in raw:
                return label

        logger.warning(
            "意图分类返回无法识别: %r（标签=%s），按 unknown 处理",
            raw, labels,
        )
        return "unknown"

    def is_available(self) -> bool:
        """Key 是否已配置。不做真实探测（避免每次自检都产生一次付费调用）。"""
        return bool(self._api_key.strip())

    # ---------------- 流式 ----------------
    def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """流式生成（SSE）。

        ★ SSE 协议要点：
          响应是逐行文本，每行形如 `data: {...}\\n\\n`
          结束标记是 `data: [DONE]`
          增量在 `choices[0].delta.content`，可能为 None（首帧/尾帧只有 role）

        ★ 为什么用 urllib 手写 SSE 解析：
          与 generate 保持一致（无 SDK 依赖），且能精确控制错误分类。
          解析逻辑很简单：按行读、去 `data: ` 前缀、跳过 `[DONE]`。
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        payload.update(self._extra)
        yield from self._post_stream(payload)

    def _post_stream(self, payload: dict) -> Iterator[str]:
        """发流式请求并逐帧产出文本增量。"""
        url = f"{self._base_url}/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "text/event-stream",
            },
        )

        try:
            resp = urllib.request.urlopen(req, timeout=self._timeout)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "ignore")[:300]
            except Exception:                      # noqa: BLE001
                pass
            raise LLMCallError(
                f"HTTP {exc.code} {exc.reason} | {detail}",
                model=self._model,
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            kind = "超时" if isinstance(reason, (socket.timeout, TimeoutError)) else "连接失败"
            raise LLMCallError(f"{kind}: {reason}", model=self._model) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise LLMCallError(f"超时: {exc}", model=self._model) from exc
        except Exception as exc:                    # noqa: BLE001
            raise LLMCallError(
                f"流式调用失败: {type(exc).__name__}: {exc}",
                model=self._model,
            ) from exc

        got_any = False
        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "ignore").strip()
                if not line or not line.startswith("data:"):
                    continue

                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break

                try:
                    frame = json.loads(data)
                except json.JSONDecodeError:
                    # 单帧解析失败不该中断整个流——跳过，继续读
                    logger.warning(
                        "SSE 帧解析失败，跳过: %r", data[:120])
                    continue

                try:
                    delta = frame["choices"][0].get("delta") or {}
                except (KeyError, IndexError, TypeError):
                    continue

                piece = delta.get("content")
                if piece:
                    got_any = True
                    yield piece

            if not got_any:
                # 流正常结束但一个字都没拿到，说明响应格式异常
                raise LLMResponseFormatError(
                    f"流式响应未产出任何内容 | endpoint={self._name}")
        except LLMCallError:
            raise
        except Exception as exc:                    # noqa: BLE001
            raise LLMCallError(
                f"读取流失败: {type(exc).__name__}: {exc}",
                model=self._model,
            ) from exc
        finally:
            try:
                resp.close()
            except Exception:                       # noqa: BLE001
                pass

    # ---------------- HTTP ----------------
    def _post(self, path: str, payload: dict) -> dict:
        """发 POST 请求，把各类异常分类成 LLMCallError / LLMResponseFormatError。

        ★ 错误分类是主备切换的判据，必须准确：
          切得太少会让用户等超时；切得太多会让「问题本身有错」也白切一次。
        """
        url = f"{self._base_url}{path}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "ignore")[:300]
            except Exception:                      # noqa: BLE001
                pass
            raise LLMCallError(
                f"HTTP {exc.code} {exc.reason} | {detail}",
                model=self._model,
            ) from exc
        except urllib.error.URLError as exc:
            # URLError 包住的是底层 socket 错误，超时也会落这里
            reason = getattr(exc, "reason", exc)
            kind = "超时" if isinstance(reason, (socket.timeout, TimeoutError)) else "连接失败"
            raise LLMCallError(
                f"{kind}: {reason}", model=self._model) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise LLMCallError(f"超时: {exc}", model=self._model) from exc
        except Exception as exc:                    # noqa: BLE001
            raise LLMCallError(
                f"调用失败: {type(exc).__name__}: {exc}",
                model=self._model,
            ) from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMResponseFormatError(
                f"响应不是合法 JSON（前 200 字符）: {raw[:200]}"
            ) from exc
