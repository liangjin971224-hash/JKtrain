"""测试替身统一出口。每个 Port 一个 Fake，另有 FailOnCallLLM。

纪律：Fake 的关键语义（权限过滤、唯一约束、文件白名单）必须与真实现一致，
否则用它测出来的结论是假绿。
"""
from __future__ import annotations

from tests.fakes.fake_cache import FakeCachePort
from tests.fakes.fake_clock import now_iso
from tests.fakes.fake_filestorage import InMemoryFileStorage
from tests.fakes.fake_graph import FakeGraphQueryPort
from tests.fakes.fake_keyword import FakeKeywordSearchPort
from tests.fakes.fake_llm import FailOnCallLLM, FakeLanguageModelPort
from tests.fakes.fake_rerank import ExplodingRerankPort, FakeRerankPort
from tests.fakes.fake_repository import FakeRepositoryPort
from tests.fakes.fake_sparse import FakeSparseSearchPort
from tests.fakes.fake_vector import FakeVectorSearchPort

__all__ = [
    "FailOnCallLLM",
    "FakeCachePort",
    "FakeGraphQueryPort",
    "FakeKeywordSearchPort",
    "FakeLanguageModelPort",
    "FakeRerankPort",
    "ExplodingRerankPort",
    "FakeRepositoryPort",
    "FakeSparseSearchPort",
    "FakeVectorSearchPort",
    "InMemoryFileStorage",
    "now_iso",
]
