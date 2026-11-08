"""本地大模型适配器（transformers + bitsandbytes 4bit）。

────────────────────────────────────────────────────────────────────
为什么单独写一个适配器，而不是复用 OpenAICompatibleAdapter
────────────────────────────────────────────────────────────────────
OpenAICompatibleAdapter 走 HTTP，本地模型走进程内推理，两者是不同形态：
    · HTTP 端点是「外部服务」，失败要重试 / 切备用
    · 本地模型是「本机资源」，失败多为 OOM，重试没有意义
强行让本地模型起一个 HTTP 服务是可行方案（Ollama / vLLM），
但那样要额外维护一个常驻进程，且 8GB 卡上 embedding 与 LLM 会抢显存。
本适配器走进程内加载，便于与 bge-m3 错峰（见 warmup / unload）。

────────────────────────────────────────────────────────────────────
★ 关键实现约束（都是实测踩出来的，勿随手改）
────────────────────────────────────────────────────────────────────
1. **必须 monkey-patch accelerate.dispatch_model**
   accelerate ≥1.10 的 dispatch_model 在单卡分支无条件调用 `model.to(device)`，
   而 bitsandbytes 4bit 模型禁止 `.to()`（会抛 ValueError）。
   实测：transformers 4.44.2 + accelerate 1.15 + bitsandbytes 0.43.3 必然报错。
   补丁做法是在 dispatch 抛「4-bit」错误时跳过设备转移并补上 hf_device_map。

2. **非参数 buffer 要手动搬到 GPU**
   4bit 只量化参数，rope 的 cos/sin 缓存仍是 CPU 上的 buffer。
   不搬会在 apply_rotary_pos_emb 处报
   `indices should be either on cpu or on the same device as the indexed tensor`。

3. **device_map 不要自己传**
   传 {"": 0} 或 "auto" 都会进 dispatch_model 从而触发第 1 条的报错路径。
   不传 → quantizer 自行处理 → 由我们的补丁兜底。

────────────────────────────────────────────────────────────────────
性能实测（RTX 4060 Laptop 8GB / Qwen2.5-7B-Instruct / NF4）
────────────────────────────────────────────────────────────────────
    · 加载 16.0s，显存 5.65GB（峰值 5.80GB）
    · 生成 2.4 tok/s（batch=1，未装 flash-attn）
    · 与 bge-m3（约 2.5GB）**不能同时驻留**，合计 8.15GB > 8GB

因此：定位是「内网 / 无外部 API / 离线批处理」场景，不适合在线高并发问答。
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

from domain.ports import GenerationResult, LanguageModelPort
from infrastructure.exceptions import LLMCallError
from infrastructure.llm.output_cleaner import clean_model_output

logger = logging.getLogger(__name__)

# accelerate 补丁只需打一次（模块级幂等）
_PATCHED = False
_PATCH_LOCK = threading.Lock()


def _patch_accelerate_dispatch() -> None:
    """让 4bit 模型跳过 accelerate 的强制 .to()。

    幂等：重复调用只生效一次。
    """
    global _PATCHED
    with _PATCH_LOCK:
        if _PATCHED:
            return
        try:
            import accelerate.big_modeling as bm
            import transformers.modeling_utils as mu
        except ImportError:  # pragma: no cover - 依赖缺失时由加载阶段报错
            return

        original = bm.dispatch_model

        def _safe_dispatch(model: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return original(model, *args, **kwargs)
            except ValueError as exc:
                msg = str(exc)
                if "4-bit" not in msg and "8-bit" not in msg:
                    raise
                # 模型已在正确设备上，只需补记录以供后续 API 查询
                device_map = kwargs.get("device_map")
                if device_map is not None:
                    try:
                        model.hf_device_map = dict(device_map)
                    except (AttributeError, TypeError):
                        pass
                logger.debug("已跳过 accelerate 对量化模型的 .to()")
                return model

        bm.dispatch_model = _safe_dispatch
        mu.dispatch_model = _safe_dispatch
        _PATCHED = True


class LocalLLMAdapter(LanguageModelPort):
    """进程内本地大模型（4bit 量化）。

    Args:
        model_path: 本地模型目录（需含 config.json / *.safetensors / tokenizer）。
        name: 端点标识，用于日志与统计。
        load_in_4bit: 是否 4bit 量化。8GB 卡上 7B 必须为 True。
        max_new_tokens_cap: 输出上限，防止误传超大值把显存打爆。
        device: 推理设备。
    """

    def __init__(
        self,
        *,
        model_path: str,
        name: str = "local",
        load_in_4bit: bool = True,
        max_new_tokens_cap: int = 2048,
        device: str = "cuda",
        lora_path: str = "",
    ) -> None:
        self._model_path = model_path
        #: LoRA 适配器目录（可选）。为空则用基座模型。
        self._lora_path = lora_path
        self._name = name
        self._load_in_4bit = load_in_4bit
        self._max_cap = max_new_tokens_cap
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._lock = threading.Lock()  # transformers 非线程安全，串行化

    # ------------------------------------------------------------------ 生命周期

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_id(self) -> str:
        return self._model_path + (f" + {self._lora_path}" if self._lora_path else "")

    def is_available(self) -> bool:
        """只检查目录与关键文件，不做真实加载（加载要 16s，不能每次自检都跑）。"""
        import os

        if not os.path.isdir(self._model_path):
            return False
        has_config = os.path.isfile(os.path.join(self._model_path, "config.json"))
        return has_config

    def load(self) -> None:
        """加载模型（幂等，已加载则直接返回）。"""
        with self._lock:
            if self._model is not None:
                return
            _patch_accelerate_dispatch()
            try:
                import torch
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    BitsAndBytesConfig,
                )
            except ImportError as exc:  # pragma: no cover
                raise LLMCallError(
                    f"本地模型依赖缺失（transformers / bitsandbytes）: {exc}"
                ) from exc

            self._torch = torch
            t0 = time.time()

            quant_config = None
            if self._load_in_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_path, trust_remote_code=True
                )
                # ★ 不传 device_map：见模块 docstring 第 3 条
                self._model = AutoModelForCausalLM.from_pretrained(
                    self._model_path,
                    quantization_config=quant_config,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if quant_config is None else None,
                    trust_remote_code=True,
                )
            except Exception as exc:
                raise LLMCallError(f"本地模型加载失败: {exc}") from exc

            # ★ 4bit 参数已量化，但 buffer（rope cos/sin）仍在 CPU，必须搬
            try:
                if next(self._model.parameters()).device.type != "cuda" and (
                    self._device.startswith("cuda")
                ):
                    self._model = self._model.to(self._device)
            except (StopIteration, AttributeError):
                pass
            for _, buf in list(self._model.named_buffers()):
                if buf.device.type != "cuda" and self._device.startswith("cuda"):
                    buf.data = buf.data.to(self._device)

            # ★ 可选挂载 LoRA 适配器（微调后的权重）。
            #   放在 buffer 迁移之后、eval() 之前，避免适配器层未被搬到 GPU。
            if self._lora_path:
                try:
                    from peft import PeftModel
                    self._model = PeftModel.from_pretrained(
                        self._model, self._lora_path
                    )
                    logger.info("已挂载 LoRA 适配器: %s", self._lora_path)
                except Exception as exc:
                    raise LLMCallError(
                        f"LoRA 适配器加载失败: {exc}"
                    ) from exc

            self._model.eval()
            logger.info(
                "本地模型已加载: %s（%.1fs），显存 %.2fGB",
                self._model_path,
                time.time() - t0,
                torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            )

    def unload(self) -> None:
        """释放模型与显存（用于与 embedding 错峰）。"""
        with self._lock:
            if self._model is None:
                return
            self._model = None
            self._tokenizer = None
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
            logger.info("本地模型已卸载，显存已释放")

    # ------------------------------------------------------------------ 内部工具

    def _build_inputs(self, prompt: str, system_prompt: str) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt")
        device = next(self._model.parameters()).device
        return {k: v.to(device) for k, v in inputs.items()}

    def _finish_reason(self, output_ids: Any, input_len: int, max_tokens: int) -> str:
        produced = output_ids.shape[-1] - input_len
        return "length" if produced >= max_tokens else "stop"

    # ------------------------------------------------------------------ 端口实现

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """同步生成。temperature=0 时用 greedy，保证可复现。"""
        self.load()
        torch = self._torch
        limit = min(max_tokens, self._max_cap)

        with self._lock:
            try:
                inputs = self._build_inputs(prompt, system_prompt)
                gen_kwargs: dict[str, Any] = {
                    "max_new_tokens": limit,
                    "pad_token_id": self._tokenizer.eos_token_id,
                }
                if temperature and temperature > 0:
                    gen_kwargs.update(
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                    )
                else:
                    gen_kwargs["do_sample"] = False

                prompt_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    out = self._model.generate(**inputs, **gen_kwargs)
            except Exception as exc:
                raise LLMCallError(f"本地模型生成失败: {exc}") from exc

            raw = self._tokenizer.decode(
                out[0][prompt_len:], skip_special_tokens=True
            ).strip()
            # ★ 清洗放在适配器出口，而不是靠训练学会「不输出 Markdown」。
            #   实测 v5 用训练解决格式问题 → 格式干净了但答案正确表述丢了。
            #   模型管内容，格式管代码：正则 100% 可靠，且不牺牲内容。
            answer = clean_model_output(raw)

            return GenerationResult(
                text=answer,
                model=self._name,
                prompt_tokens=int(prompt_len),
                completion_tokens=int(out.shape[-1] - prompt_len),
                finish_reason=self._finish_reason(out[0], prompt_len, limit),
            )

    def classify(
        self,
        text: str,
        labels: list[str],
        *,
        allow_unknown: bool = True,
    ) -> str:
        """意图分类。提示词与解析逻辑对齐 OpenAICompatibleAdapter，保证行为一致。"""
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
        result = self.generate(prompt, temperature=0.0, max_tokens=16)
        raw = result.text.strip().strip("\"'").strip()

        if raw in labels:
            return raw
        if allow_unknown and raw.lower() == "unknown":
            return "unknown"
        for label in labels:
            if label in raw:
                return label

        logger.warning("意图分类返回无法识别: %r（标签=%s），按 unknown 处理", raw, labels)
        return "unknown"

    def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """流式生成。

        ★ 用 TextIteratorStreamer 在后台线程解码，产出「新增文本增量」，
          而不是把累计文本重复 yield 出去——上层是按句缓冲再送护栏的，
          重复内容会让护栏重复校验同一句。
        """
        self.load()
        torch = self._torch
        limit = min(max_tokens, self._max_cap)

        from transformers import TextIteratorStreamer

        inputs = self._build_inputs(prompt, system_prompt)
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": limit,
            "pad_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen_kwargs["do_sample"] = False

        error_box: list[BaseException] = []

        def _run() -> None:
            try:
                with torch.no_grad():
                    self._model.generate(**inputs, **gen_kwargs)
            except BaseException as exc:  # noqa: BLE001 - 需传回主线程
                error_box.append(exc)
            finally:
                streamer.end()

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            worker.join(timeout=5)

        if error_box:
            raise LLMCallError(f"本地模型流式生成失败: {error_box[0]}") from error_box[0]
