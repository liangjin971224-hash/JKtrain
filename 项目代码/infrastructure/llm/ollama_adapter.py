"""Ollama（llama.cpp）适配器。

════════════════════════════════════════════════════════════════════
为什么需要这个适配器
════════════════════════════════════════════════════════════════════
`LocalLLMAdapter` 走 transformers，实测 **2.2 tok/s**（每个 token 一次 Python
循环 + 约 800 次 kernel launch），30–38s 才能答完一句，在线问答不可用。

同一个 3B 模型换成 Ollama（llama.cpp 内核，GGUF Q4_K_M）后实测 **101–108 tok/s**，
100 token 仅 0.93s。**快 45–49 倍，瓶颈在框架不在模型。**

────────────────────────────────────────────────────────────────────
★ 必须用 /api/chat，不能用 /api/generate
────────────────────────────────────────────────────────────────────
实测（同一模型、同一问题、同一提示词）：

    /api/generate  system 手写进 prompt  → 「知识库中没有具体信息」+ 抄资料 ❌
    /api/chat      system 作为独立字段   → 「不能。阿司匹林和鱼油都含抗凝成分，
                                          可能增加出血风险」✅

原因：Ollama 的 GGUF 内置 Qwen 对话模板
    <|im_start|>system\n{sys}<|im_end|>
    <|im_start|>user\n{prompt}<|im_end|>
    <|im_start|>assistant\n

`/api/generate` **不套模板**，把 prompt 当裸文本。而微调时每条样本都过了
`tokenizer.apply_chat_template()` —— 模型只见过带模板的格式，喂裸文本等于换任务。

这与「训练/推理 system 必须 import 不手抄」是同一类问题的两个表现：
**格式差异会让微调效果打折，而 loss 曲线完全正常，只能实测发现。**
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
from infrastructure.llm.output_cleaner import clean_model_output

logger = logging.getLogger(__name__)


class OllamaAdapter(LanguageModelPort):
    """Ollama 适配器（走 /api/chat，正确套用 GGUF 内置对话模板）。"""

    def __init__(
        self,
        *,
        name: str = "ollama",
        base_url: str = "http://127.0.0.1:11434",
        model: str = "health-advisor:3b",
        timeout_sec: int = 120,
        max_new_tokens_cap: int = 1024,
        clean_output: bool = True,
    ) -> None:
        """
        Args:
            name: 端点标识，用于日志与统计。
            base_url: Ollama 服务地址（不带 /api/chat）。
            model: 模型名，如 health-advisor:3b。
            timeout_sec: 单次调用超时。本地推理首 token 可能较慢，默认放宽到 120s。
            max_new_tokens_cap: 输出上限（防止误传超大值把耗时拉长）。
            clean_output: 是否做输出清洗（剥 Markdown / 砍回吐）。
                ★ 微调模型仍会输出 `-` 列表和参考链接，必须清洗。
                测试或对比时可关掉，看原始输出。
        """
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_sec
        self._cap = max_new_tokens_cap
        self._clean = clean_output

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def model_id(self) -> str:
        """便于装配层日志统一打印（与 LocalLLMAdapter 对齐）。"""
        return f"{self._base_url} → {self._model}"

    # ---------------- 端口实现 ----------------
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """生成文本（走 /api/chat，system 作为独立 messages 项）。

        ★ 与非流式一次拿完整结果。Ollama 的 /api/chat 在 stream=false 时
          返回单个 JSON 对象，而非 SSE 流。
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": min(max_tokens, self._cap),
            },
        }
        data = self._post("/api/chat", payload)

        # Ollama 的错误也可能以 200 + {"error": "..."} 返回
        if "error" in data and "message" not in data:
            raise LLMCallError(
                f"Ollama 返回错误: {data['error']}", model=self._model)

        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMResponseFormatError(
                f"返回体缺少 message.content | model={self._model}"
            ) from exc

        if not isinstance(text, str):
            raise LLMResponseFormatError(
                f"content 不是字符串: {type(text).__name__} | model={self._model}")

        # ★ 清洗放在适配器出口：微调模型仍会输出 `- ` 列表、参考链接、
        #   甚至回吐资料原文。这些属于「格式」，该由代码管，不靠训练学。
        if self._clean:
            text = clean_model_output(text)

        return GenerationResult(
            text=text,
            model=self._model,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            # ★ done_reason == "length" 表示被 num_predict 截断，
            #   上层必须当作不完整答案处理（与 OpenAI 的 length 同义）
            finish_reason=(
                "length" if data.get("done_reason") == "length" else "stop"
            ),
        )

    def classify(
        self,
        text: str,
        labels: list[str],
        *,
        allow_unknown: bool = True,
    ) -> str:
        """意图分类。与 OpenAICompatibleAdapter 保持同一套解析规则。"""
        if not labels:
            return "unknown"

        label_list = "、".join(labels)
        unknown_hint = "如果无法判断，只输出 unknown。" if allow_unknown else ""
        prompt = (
            f"把下面这句话归类到给定标签之一。\n"
            f"标签：{label_list}\n"
            f"{unknown_hint}\n"
            f"只输出标签本身，不要解释、不要加引号、不要加标点。\n\n"
            f"句子：{text}"
        )
        # 分类不需要清洗（只要标签），传 clean_output=False 避免干扰短输出
        result = self.generate(prompt, temperature=0.0, max_tokens=16)
        raw = result.text.strip().strip('"\'').strip()

        if raw in labels:
            return raw
        if allow_unknown and raw.lower() == "unknown":
            return "unknown"

        for label in labels:
            if label in raw:
                return label

        logger.warning(
            "意图分类返回无法识别: %r（标签=%s），按 unknown 处理", raw, labels)
        return "unknown"

    def is_available(self) -> bool:
        """探测 Ollama 是否在线（本地服务，探测很便宜，不像云端会计费）。"""
        url = f"{self._base_url}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                return resp.status == 200
        except Exception:                          # noqa: BLE001
            return False

    # ---------------- 流式 ----------------
    def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """流式生成。

        ★ Ollama 的流式是 **NDJSON**（每行一个 JSON），不是 OpenAI 的 SSE。
          每行形如 {"message":{"content":"增量"},"done":false}
          末行 {"done":true,...}

        ★ 流式下**不做清洗**：清洗需要看到完整句子才能判断是不是回吐，
          逐字清洗会把正常内容也切碎。调用方应配合 StreamGuard 在句子层面处理。
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": min(max_tokens, self._cap),
            },
        }
        yield from self._post_stream("/api/chat", payload)

    def _post_stream(self, path: str, payload: dict) -> Iterator[str]:
        """发流式请求并逐帧产出文本增量（NDJSON 解析）。"""
        url = f"{self._base_url}{path}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
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
                f"HTTP {exc.code} {exc.reason} | {detail}", model=self._model) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            kind = "超时" if isinstance(reason, (socket.timeout, TimeoutError)) else "连接失败"
            raise LLMCallError(f"{kind}: {reason}", model=self._model) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise LLMCallError(f"超时: {exc}", model=self._model) from exc
        except Exception as exc:                    # noqa: BLE001
            raise LLMCallError(
                f"流式调用失败: {type(exc).__name__}: {exc}", model=self._model) from exc

        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "ignore").strip()
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("NDJSON 帧解析失败，跳过: %r", line[:120])
                    continue

                if frame.get("error"):
                    raise LLMCallError(
                        f"Ollama 流式错误: {frame['error']}", model=self._model)

                piece = frame.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if frame.get("done"):
                    break
        finally:
            try:
                resp.close()
            except Exception:                      # noqa: BLE001
                pass

    # ---------------- HTTP ----------------
    def _post(self, path: str, payload: dict) -> dict:
        """发 POST，把异常分类成 LLMCallError / LLMResponseFormatError。

        错误分类与 OpenAICompatibleAdapter 保持一致（主备切换的判据）。
        """
        url = f"{self._base_url}{path}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
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
                f"HTTP {exc.code} {exc.reason} | {detail}", model=self._model) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            kind = "超时" if isinstance(reason, (socket.timeout, TimeoutError)) else "连接失败"
            raise LLMCallError(f"{kind}: {reason}", model=self._model) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise LLMCallError(f"超时: {exc}", model=self._model) from exc
        except Exception as exc:                    # noqa: BLE001
            raise LLMCallError(
                f"调用失败: {type(exc).__name__}: {exc}", model=self._model) from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMResponseFormatError(
                f"响应不是合法 JSON（前 200 字符）: {raw[:200]}") from exc
