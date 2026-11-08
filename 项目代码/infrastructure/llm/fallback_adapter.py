"""LLM 主备双通道（Failover）适配器。

★ 本项目的「双重保险」实现：千问为主，DeepSeek 为备。
  千问任何异常 → 自动切 DeepSeek，对上层完全透明。

────────────────────────────────────────────────────────────────────
为什么需要它
────────────────────────────────────────────────────────────────────
本项目的问答是核心功能（P4 演示点）。单 Key 意味着：
    · DeepSeek 一次限流 → 整个问答不可用
    · Key 欠费 → 演示当天崩在半路
    · 网络抖动 → 用户等 30 秒后看到报错

两个不同厂商的 Key 同时挂的概率极低（不同云、不同链路线路）。

────────────────────────────────────────────────────────────────────
什么情况切、什么情况不切（最关键的设计）
────────────────────────────────────────────────────────────────────
切（都是**端点级故障**，换个端点大概率能好）：
    · 超时
    · 限流（429）
    · 服务端错误（5xx）
    · 鉴权失败（401/403）——Key 失效了，必须换
    · 返回体格式无法解析 —— 该端点协议异常

不额外切（已经在切了）：
    重试成功后**不切**。避免不必要的模型切换导致同一次会话里回答风格突变。

★ 全部端点都失败时**抛异常**，不返回空字符串。
  返回空串会被上层当成「模型说没内容」，掩盖真实故障；
  而本项目最怕的就是「看起来正常但实际没依据」的回答。

────────────────────────────────────────────────────────────────────
可观测性
────────────────────────────────────────────────────────────────────
切换是「静默降级」，如果没留痕，出故障时根本不知道切过。
所以记录：
    · `last_used`  最后一次成功的端点
    · `last_error` 最后一次切换的原因
    · `stats()`    每个端点成功/失败次数（用于监控主备比例）
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from domain.ports import GenerationResult, LanguageModelPort
from infrastructure.exceptions import LLMCallError

logger = logging.getLogger(__name__)


@dataclass
class _EndpointStat:
    """单个端点的调用统计。"""

    success: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"success": self.success, "failed": self.failed}


@dataclass
class FallbackLanguageModel(LanguageModelPort):
    """主备双通道 LLM。

    按 `primary → backups...` 顺序尝试，第一个成功即返回。

    Attributes:
        last_used: 最后一次成功的端点名（None 表示还没成功过）。
        last_error: 最后一次切换的错误描述（None 表示没切过）。
    """

    primary: LanguageModelPort
    backups: list[LanguageModelPort] = field(default_factory=list)
    primary_retries: int = 0

    last_used: str | None = None
    last_error: str | None = None

    _stats: dict[str, _EndpointStat] = field(default_factory=dict, init=False)

    # ---------------- 端口实现 ----------------
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        return self._try_all(
            "generate",
            lambda ep: ep.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )

    def classify(
        self,
        text: str,
        labels: list[str],
        *,
        allow_unknown: bool = True,
    ) -> str:
        return self._try_all(
            "classify",
            lambda ep: ep.classify(text, labels, allow_unknown=allow_unknown),
        )

    def is_available(self) -> bool:
        """任一端点可用即视为可用。"""
        return any(ep.is_available() for ep in self._endpoints())

    # ---------------- 流式 ----------------
    def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """流式生成，带主备切换。

        ★ 切换规则与 `generate` 不同（流式的固有限制）：
          只有在**还没产出任何内容之前**失败，才能切到备用。
          一旦已经向调用方 yield 过内容，再切换会导致：
            · 前半段来自千问、后半段来自 DeepSeek，风格断裂
            · 内容重复或逻辑不连贯
          所以此时**直接抛异常**，由上层转拒答。

        ★ 这也意味着流式下「兜底」能力弱于非流式：
          千问在生成中途挂掉，流式无法救回来。
          取舍：拿延迟收益，换「中途故障不可切」。
          对演示场景值得（中途故障概率远低于首帧失败）。
        """
        endpoints = self._endpoints()
        errors: dict[str, str] = {}

        for idx, ep in enumerate(endpoints):
            name = self._name_of(ep)
            stat = self._stat_of(name)

            # 流式不做重试：重试已产出的部分会重复
            emitted_any = False
            try:
                for piece in ep.generate_stream(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    if not emitted_any:
                        emitted_any = True
                        # 首帧成功即认为该端点可用（后续失败无法切换）
                        self.last_used = name
                        stat.success += 1
                    yield piece
            except Exception as exc:                # noqa: BLE001
                if not emitted_any:
                    stat.success -= 1               # 撤销刚才记的成功
                    stat.failed += 1
                msg = f"{type(exc).__name__}: {exc}"
                errors[name] = msg
                logger.warning(
                    "LLM 流式端点失败: endpoint=%s, 已产出=%s, error=%s",
                    name, emitted_any, msg,
                )

                if emitted_any:
                    # ★ 已经吐过内容，不能再切换
                    self.last_error = msg
                    raise LLMCallError(
                        f"流式中途失败，无法切换端点（已产出内容）。"
                        f"{name}: {msg}"
                    ) from exc

                if errors:
                    self.last_error = "；".join(
                        f"{k}: {v}" for k, v in errors.items())
                continue                            # 未产出，可切下一个

            else:
                # 正常结束
                if emitted_any:
                    if errors:
                        logger.warning(
                            "LLM 已降级到 %s（流式），前序失败: %s",
                            name, self.last_error)
                    return
                # 一帧都没出，视为该端点失败，试下一个
                errors[name] = "流式响应为空"
                stat.success -= 1
                stat.failed += 1
                continue

        tried = "；".join(f"{k}: {v}" for k, v in errors.items()) or "无可用端点"
        self.last_error = tried
        raise LLMCallError(
            f"所有 LLM 端点的流式调用均失败（{len(endpoints)} 个）。{tried}"
        )

    # ---------------- 内部 ----------------
    def _endpoints(self) -> list[LanguageModelPort]:
        return [self.primary, *self.backups]

    def _name_of(self, ep: LanguageModelPort) -> str:
        return getattr(ep, "name", type(ep).__name__)

    def _stat_of(self, name: str) -> _EndpointStat:
        if name not in self._stats:
            self._stats[name] = _EndpointStat()
        return self._stats[name]

    def _try_all(self, action: str, call):
        """按顺序尝试各端点。

        Args:
            action: 方法名，仅用于日志。
            call: 接受端点、返回结果的调用函数。

        Returns:
            第一个成功的结果。

        Raises:
            LLMCallError: 全部端点失败。异常信息列出每个端点的错误。
        """
        endpoints = self._endpoints()
        errors: dict[str, str] = {}

        for idx, ep in enumerate(endpoints):
            name = self._name_of(ep)
            stat = self._stat_of(name)

            # 主模型可配置重试；备用不做重试（已经慢了一步，再重试更慢）
            attempts = 1
            if idx == 0:
                attempts += max(self.primary_retries, 0)

            for attempt in range(attempts):
                try:
                    result = call(ep)
                except LLMCallError as exc:
                    stat.failed += 1
                    errors[name] = str(exc)
                    logger.warning(
                        "LLM 端点失败: endpoint=%s, action=%s, "
                        "attempt=%d/%d, error=%s",
                        name, action, attempt + 1, attempts, exc,
                    )
                    if attempt + 1 < attempts:
                        continue        # 还有重试次数，重试同一端点
                    break               # 该端点放弃，换下一个
                except Exception as exc:            # noqa: BLE001
                    # 非 LLMCallError 也要计入失败并切换——
                    # 但记录成「未预期异常」，提示这里可能该补分类
                    stat.failed += 1
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "LLM 端点未预期异常: endpoint=%s, action=%s, error=%s",
                        name, action, exc, exc_info=True,
                    )
                    break
                else:
                    stat.success += 1
                    # ★ 切换成功时也要记录 last_error。
                    #   只在「全部失败」时记录是不够的——最常见的场景恰恰是
                    #   「切了并且成功了」，这时如果不记，运维看到的就是
                    #   「一切正常」，完全不知道千问挂过。
                    if errors:
                        self.last_error = "；".join(
                            f"{k}: {v}" for k, v in errors.items())
                        logger.warning(
                            "LLM 已降级到 %s，前序端点失败原因: %s",
                            name, self.last_error)
                    self.last_used = name
                    return result

        # 全部失败
        tried = "；".join(f"{k}: {v}" for k, v in errors.items()) or "无可用端点"
        self.last_error = tried
        logger.error("LLM 全部端点失败: action=%s, %s", action, tried)
        raise LLMCallError(
            f"所有 LLM 端点均失败（{len(endpoints)} 个）。{tried}"
        )

    # ---------------- 观测 ----------------
    def stats(self) -> dict[str, dict[str, int]]:
        """各端点的成功/失败次数。

        用途：监控主备比例。若 deepseek 的 success 持续上升，
        说明千问在持续故障，该检查配额或换主模型。

        ★ 未被调用过的端点也返回全 0，而不是缺 key。
          否则监控做 `stats["deepseek"]["success"]` 会 KeyError，
          而「备用从未被用过」本身就是一个重要信号（说明主模型一直健康）。
        """
        out: dict[str, dict[str, int]] = {}
        for ep in self._endpoints():
            name = self._name_of(ep)
            out[name] = self._stat_of(name).to_dict()
        return out

    def reset_stats(self) -> None:
        """清零统计（测试与运维排查用）。"""
        self._stats.clear()
        self.last_used = None
        self.last_error = None
