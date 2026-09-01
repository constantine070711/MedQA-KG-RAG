# -*- coding: utf-8 -*-
"""Phase 10-B B2 — 旁路 Reranker 单元测试（mock 模型，零加载 / 零网络 / 零 LLM / 零 DB）。

覆盖要求：
    1. 正常 rerank 排序（score 降序 + 截取 top_m）
    2. empty candidates 安全返回
    3. score exception → fallback 原始顺序
    4. input immutability（不修改输入对象）
    5. mock model（不可用 / 打分异常均可注入）

运行:
    python -m unittest tests.test_reranker -v
"""

import copy
import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.retrieval.rerank import Reranker, rerank_chunks  # noqa: E402


class MockReranker:
    """可注入的 mock reranker：控制可用性 / 分数 / 是否抛异常。"""

    def __init__(self, scores=None, available=True, raise_on_score=False,
                 short_scores=False):
        self._scores = scores if scores is not None else [0.0]
        self.available = available
        self.raise_on_score = raise_on_score
        self.short_scores = short_scores
        self.calls = 0

    @property
    def is_available(self):
        return self.available

    def score_pairs(self, pairs):
        self.calls += 1
        if self.raise_on_score:
            raise RuntimeError("mock score failure")
        n = len(pairs)
        if self.short_scores:
            return self._scores[: max(n - 1, 0)]  # 数量不匹配
        out = []
        for i in range(n):
            out.append(self._scores[i] if i < len(self._scores) else 0.0)
        return out


def _chunks(*contents):
    return [{"chunk_id": f"c{i}", "content": c, "file_path": "x.md"}
            for i, c in enumerate(contents)]


# ============================================================
# 1. 正常 rerank 排序
# ============================================================

class TestRerankOrdering(unittest.TestCase):
    def test_sorts_by_score_desc(self):
        chunks = _chunks("A 内容", "B 内容", "C 内容")
        # 分数反序：第 2 条最相关 → 应排第一
        rk = MockReranker(scores=[0.1, 0.9, 0.5])
        out = rerank_chunks("问题", chunks, top_m=3, reranker=rk)
        self.assertEqual([c["chunk_id"] for c in out], ["c1", "c2", "c0"])

    def test_top_m_truncation(self):
        chunks = _chunks("A", "B", "C", "D")
        rk = MockReranker(scores=[0.9, 0.8, 0.7, 0.6])
        out = rerank_chunks("q", chunks, top_m=2, reranker=rk)
        self.assertEqual(len(out), 2)

    def test_rerank_score_attached(self):
        chunks = _chunks("A", "B")
        rk = MockReranker(scores=[0.9, 0.2])
        out = rerank_chunks("q", chunks, top_m=2, reranker=rk)
        self.assertIn("rerank_score", out[0])
        self.assertGreater(out[0]["rerank_score"], out[1]["rerank_score"])

    def test_runs_reranker_after_retrieval(self):
        chunks = _chunks("A", "B")
        rk = MockReranker(scores=[0.5, 0.5])
        rerank_chunks("q", chunks, top_m=2, reranker=rk)
        self.assertEqual(rk.calls, 1)


# ============================================================
# 2. empty candidates
# ============================================================

class TestEmptyCandidates(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(rerank_chunks("q", [], top_m=10, reranker=MockReranker()), [])

    def test_none_chunks(self):
        self.assertEqual(rerank_chunks("q", None, top_m=10, reranker=MockReranker()), [])

    def test_top_m_zero(self):
        chunks = _chunks("A", "B")
        self.assertEqual(rerank_chunks("q", chunks, top_m=0, reranker=MockReranker()), [])

    def test_top_m_negative(self):
        chunks = _chunks("A", "B")
        self.assertEqual(rerank_chunks("q", chunks, top_m=-1, reranker=MockReranker()), [])

    def test_all_chunks_without_content(self):
        chunks = [{"chunk_id": "c0"}, {"chunk_id": "c1"}]  # 无 content → 无候选
        rk = MockReranker()
        out = rerank_chunks("q", chunks, top_m=2, reranker=rk)
        # 候选为空且可用 → 内部返回空（无 pair 可打分）；不得崩溃
        self.assertIsInstance(out, list)


# ============================================================
# 3. score exception → fallback 原始顺序
# ============================================================

class TestScoreExceptionFallback(unittest.TestCase):
    def test_raise_falls_back_to_original_order(self):
        chunks = _chunks("A", "B", "C")
        rk = MockReranker(raise_on_score=True)
        out = rerank_chunks("q", chunks, top_m=3, reranker=rk)
        self.assertEqual([c["chunk_id"] for c in out], ["c0", "c1", "c2"])  # 原始顺序
        self.assertNotIn("rerank_score", out[0])  # 未附加分数

    def test_short_scores_falls_back(self):
        chunks = _chunks("A", "B", "C")
        rk = MockReranker(short_scores=True)
        out = rerank_chunks("q", chunks, top_m=3, reranker=rk)
        self.assertEqual([c["chunk_id"] for c in out], ["c0", "c1", "c2"])

    def test_fallback_still_truncates_to_top_m(self):
        chunks = _chunks("A", "B", "C", "D")
        rk = MockReranker(raise_on_score=True)
        out = rerank_chunks("q", chunks, top_m=2, reranker=rk)
        self.assertEqual([c["chunk_id"] for c in out], ["c0", "c1"])


# ============================================================
# 4. input immutability
# ============================================================

class TestInputImmutability(unittest.TestCase):
    def test_input_chunks_not_mutated(self):
        chunks = _chunks("A 内容", "B 内容", "C 内容")
        before = [copy.deepcopy(c) for c in chunks]
        rk = MockReranker(scores=[0.1, 0.9, 0.5])
        rerank_chunks("q", chunks, top_m=3, reranker=rk)
        self.assertEqual(chunks, before)

    def test_fallback_does_not_mutate_input(self):
        chunks = _chunks("A", "B")
        before = [copy.deepcopy(c) for c in chunks]
        rk = MockReranker(raise_on_score=True)
        rerank_chunks("q", chunks, top_m=2, reranker=rk)
        self.assertEqual(chunks, before)

    def test_returned_objects_are_new(self):
        chunks = _chunks("A", "B")
        rk = MockReranker(scores=[0.9, 0.1])
        out = rerank_chunks("q", chunks, top_m=2, reranker=rk)
        for o, c in zip(out, chunks):
            self.assertIsNot(o, c)          # 新对象
            self.assertEqual(o["content"], c["content"])  # 内容一致


# ============================================================
# 5. mock model（可用性控制）
# ============================================================

class TestModelAvailability(unittest.TestCase):
    def test_unavailable_falls_back(self):
        chunks = _chunks("A", "B", "C")
        rk = MockReranker(available=False)
        out = rerank_chunks("q", chunks, top_m=3, reranker=rk)
        self.assertEqual([c["chunk_id"] for c in out], ["c0", "c1", "c2"])
        self.assertEqual(rk.calls, 0)  # 不可用则不打分

if __name__ == "__main__":
    unittest.main()
