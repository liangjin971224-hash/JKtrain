"""参数透传的结构性守护（2026-10-24）。

────────────────────────────────────────────────────────────────────
为什么要专门写这个文件
────────────────────────────────────────────────────────────────────
同一个坑踩了**三次**：

    2026-10-23  路由 `answer_one` 漏透传 `fact_hints`
                → 多跳查到的事实传不到生成层
                → 接口里又编出「目录中没有改善体脂等表述」
                → 而直连用例的测试全绿（单测绿 ≠ 接口可用）

    2026-10-23  同一处漏透传 `product_ids`
                → 护栏拿不到产品类别，普通食品声称判不出来

    2026-10-24  同一处漏透传 `history`
                → 追问「那它有什么禁忌」接不上，模型反问「您问的是哪一款」

三次都是**可选项没从路由传到用例**。每次都靠人肉盯，盯不住。

所以这个文件不测行为，测**结构**：
    `AnswerQuestionUseCase.execute` 接受哪些可选参数，
    `_build_multi_hop` 里那层包装就必须**全部**接受并透传。

新加一个可选参数忘了透传 → 这里立刻 FAIL，不用等到端到端才发现。
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from application.use_cases.answer_question import AnswerQuestionUseCase

ASK_PY = (Path(__file__).resolve().parents[2]
          / "interfaces" / "http" / "routers" / "ask.py")

#: 这些参数**必须**被多跳包装层透传（有实际语义的可选项）。
#:
#: 不在此列的：`vector_collection`（包装层自己从 settings 取）、
#: `top_k`（多跳用默认值）、`intent`（包装层不传，让用例自己分类）。
#:
#: ★ 2026-11-08 合并：`history` 已从列表移除——多轮改为「改写问题」方案
#:   （`conversation` + `query_rewriter`），不再把历史注进 prompt（D4）。
MUST_FORWARD = ("fact_hints", "product_ids")


@pytest.fixture(scope="module")
def ask_source() -> str:
    return ASK_PY.read_text(encoding="utf-8")


class TestExecuteSignature:
    def test_optional_params_present(self) -> None:
        """确认用例确实有这些可选项（防止改名后测试形同虚设）。"""
        sig = inspect.signature(AnswerQuestionUseCase.execute)
        for name in MUST_FORWARD:
            assert name in sig.parameters, (
                f"`execute` 没有参数 {name} —— 测试里 MUST_FORWARD 该更新了"
            )

    def test_stream_has_same_params(self) -> None:
        """★ 流式与非流式必须接受同一组可选参数。

        只给一边加参数是常见疏漏：非流式能追问、流式不能，
        而前端默认勾的就是流式。
        """
        sync_sig = inspect.signature(AnswerQuestionUseCase.execute)
        stream_sig = inspect.signature(AnswerQuestionUseCase.execute_stream)
        for name in MUST_FORWARD:
            assert name in stream_sig.parameters, (
                f"`execute_stream` 缺少 {name}，流式追问会失效"
            )
            assert name in sync_sig.parameters


class TestRouterForwards:
    def test_every_param_is_forwarded(self, ask_source: str) -> None:
        """★ 本文件核心：`_build_multi_hop` 里的 `answer_one` 必须透传全部可选项。

        做法：截出 `def answer_one(` 到下一次 `return QueryPlanUseCase` 之间的代码，
        逐个检查必须出现的透传关键字。
        """
        start = ask_source.find("def answer_one(")
        assert start != -1, "找不到 _build_multi_hop 里的 answer_one"
        end = ask_source.find("return QueryPlanUseCase", start)
        assert end != -1
        body = ask_source[start:end]

        missing = [n for n in MUST_FORWARD
                   if not re.search(rf"\b{n}\s*=\s*[^,\n]*{n}", body)]
        assert not missing, (
            f"answer_one 没有透传: {missing}\n"
            f"  这个坑踩过四次（fact_hints / product_ids / history），\n"
            f"  每次都表现为「直连用例对、走接口错」。\n"
            f"  判断方式：要求出现 `名字=...名字`（允许 cast，如\n"
            f"  `product_ids=tuple(product_ids or ())`）。\n"
            f"  透传代码段:\n{body}"
        )

    def test_answer_one_accepts_them(self, ask_source: str) -> None:
        """包装层函数签名本身也要接受这些参数，否则调用方传进来会 TypeError。

        ★ 不能用 `def answer_one\\(([^)]*)\\)` 这种正则：
          参数默认值里有 `product_ids=()`，那个 `)` 会在第一个类型注解处
          把匹配截断，导致「history 不在签名里」的假失败（红灯抓出）。
          所以按括号配对找函数头。
        """
        start = ask_source.find("def answer_one(")
        assert start != -1
        i = ask_source.find("(", start)
        depth = 0
        end = -1
        for j in range(i, len(ask_source)):
            if ask_source[j] == "(":
                depth += 1
            elif ask_source[j] == ")":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        assert end != -1, "answer_one 签名括号不配对"
        params = ask_source[i + 1:end]

        missing = [n for n in MUST_FORWARD if n not in params]
        assert not missing, (
            f"answer_one 签名里没有: {missing}\n  实际签名: {params}"
        )


class TestQueryPlanForwards:
    """多跳层内部的透传（`_run_step` → `_answer` → `answer_one`）。

    ★ 2026-11-08 合并：`history` 相关的两条断言已随 D4 退役
      （多轮改用问题改写，不再透传历史字符串）。`seed_product_ids`
      仍在用——多轮产品切换靠它把新主语带进每一跳。
    """

    QP = (Path(__file__).resolve().parents[2]
          / "application" / "use_cases" / "query_plan.py")

    def test_execute_accepts_seed_products(self) -> None:
        """★ 纯指代追问（「那它有什么禁忌」）全靠这个锚点。

        没有 seed_product_ids 时，这类问题里没有任何产品名，
        召回拿不到东西，模型只能反问。
        """
        from application.use_cases.query_plan import QueryPlanUseCase
        sig = inspect.signature(QueryPlanUseCase.execute)
        assert "seed_product_ids" in sig.parameters
