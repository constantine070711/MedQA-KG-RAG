# -*- coding: utf-8 -*-
"""Phase 6 F6 — eval_phase6 单元测试（纯 fixture，stdlib unittest，零真实 Neo4j / LLM）。

覆盖:
    1.  baseline system_prompt 选择正确（SAFE→None / CAUTION→SYSTEM_PROMPT_SAFE /
        RESTRICTED→modified_prompt / BLOCKED→None）
    2.  neo4j system_prompt 正确注入 fusion prompt（有证据→含证据段；无证据→base 原样）
    3.  improve / regress / unchanged 分类器（阈值 ±0.05）
    4.  聚合函数（aggregate_metrics / aggregate_decisions）
    5.  M7 confusion matrix 聚合（aggregate_m7 复用 refusal_accuracy）
    6.  Neo4j impact 聚合（build_impact：hits + improved/regressed + decision flips）
    7.  不依赖真实 Neo4j / 不调用真实 LLM / 不依赖特定 QID
    8.  不引入 forbidden dependency（eval_phase6.py 不直接 import torch/openai/lightrag）

运行:
    python -m unittest tests.test_eval_phase6 -v
"""

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.medical_core.safety import SafetyChecker, SafetyLevel, SYSTEM_PROMPT_SAFE
from src.evidence_fusion import FusionConfig
from scripts.eval_phase6 import (
    DETERMINISTIC_METRIC_KEYS,
    CLASSIFY_THRESHOLD,
    select_system_prompt,
    classify_question,
    _deterministic_score,
    aggregate_metrics,
    aggregate_m7,
    aggregate_decisions,
    build_impact,
    build_report,
    _compact_record,
    Phase6Evaluator,
)


# ============================================================
# 辅助：构造合成 safety / record
# ============================================================

def _safety(level: str):
    checker = SafetyChecker()
    if level == "blocked":
        return SafetyCheckResultBlocked()
    return _FakeSafety(level)


class _FakeSafety:
    def __init__(self, level):
        self.level = SafetyLevel(level)
        self.reason = "synthetic"
        self.disclaimer_needed = False
        self.modified_prompt = "【RESTRICTED 安全指令】切勿提供具体用药方案或剂量建议。"
        self.rejection_message = "高危问题，直接拒答。"

    @property
    def is_restricted(self):
        return self.level in (SafetyLevel.RESTRICTED, SafetyLevel.BLOCKED)


class SafetyCheckResultBlocked(_FakeSafety):
    def __init__(self):
        super().__init__("blocked")


def _mk_record(qid="q1", metrics=None, decision=None, status="success",
               expected_refusal=False, actual_refusal=None, safety_level="safe",
               with_neo4j=False, mentions=None, neo4j_provenance=None,
               neo4j_unavailable=False, error=None):
    return {
        "question_id": qid,
        "question": f"question {qid}",
        "category": "basic_fact",
        "status": status,
        "error": error,
        "metrics": metrics or {},
        "reliability_decision": decision,
        "expected_refusal": expected_refusal,
        "actual_refusal": actual_refusal,
        "safety_level": safety_level,
        "with_neo4j": with_neo4j,
        "mentions": mentions,
        "neo4j_provenance": neo4j_provenance,
        "neo4j_unavailable": neo4j_unavailable,
    }


def _metric(key, score):
    return {key: {"score": score}}


# ============================================================
# 1. system_prompt 选择
# ============================================================

class TestSelectSystemPrompt(unittest.TestCase):
    def test_baseline_safe_is_none(self):
        self.assertIsNone(select_system_prompt(_safety("safe"), with_neo4j=False))

    def test_baseline_caution_is_safe_prompt(self):
        p = select_system_prompt(_safety("caution"), with_neo4j=False)
        self.assertIsNotNone(p)
        self.assertIn("医疗知识辅助助手", p)

    def test_baseline_restricted_is_modified_prompt(self):
        p = select_system_prompt(_safety("restricted"), with_neo4j=False)
        self.assertIsNotNone(p)
        self.assertIn("RESTRICTED", p)

    def test_baseline_blocked_is_none(self):
        self.assertIsNone(select_system_prompt(_safety("blocked"), with_neo4j=False))

    def test_neo4j_injects_evidence(self):
        n4 = {"nodes": [{"id": "二甲双胍", "name": "二甲双胍", "labels": ["Drug"], "properties": {}}],
              "edges": [], "triples": [["二甲双胍", "TREATS", "2型糖尿病"]]}
        p = select_system_prompt(_safety("safe"), with_neo4j=True, neo4j_result=n4)
        self.assertIsNotNone(p)
        self.assertIn("TREATS", p)
        self.assertIn("检索到的", p)  # 证据段框架，非系统级指令

    def test_neo4j_no_evidence_returns_base(self):
        n4 = {"nodes": [], "edges": [], "triples": [], "note": "未命中 curated 实体"}
        self.assertIsNone(select_system_prompt(_safety("safe"), with_neo4j=True, neo4j_result=n4))
        p = select_system_prompt(_safety("caution"), with_neo4j=True, neo4j_result=n4)
        self.assertIn("医疗知识辅助助手", p)

    def test_neo4j_unavailable_returns_base(self):
        n4 = {"unavailable": True, "error": "connection refused"}
        self.assertIsNone(select_system_prompt(_safety("safe"), with_neo4j=True, neo4j_result=n4))

    def test_neo4j_prompt_does_not_override_safety(self):
        n4 = {"nodes": [], "edges": [], "triples": [["a", "TREATS", "b"]]}
        p = select_system_prompt(_safety("caution"), with_neo4j=True, neo4j_result=n4)
        self.assertIn("医疗知识辅助助手", p)  # base 安全指令保留
        self.assertIn("TREATS", p)            # 证据段追加


# ============================================================
# 2. improve / regress / unchanged 分类器
# ============================================================

class TestClassify(unittest.TestCase):
    def test_improved(self):
        self.assertEqual(classify_question(0.5, 0.56), "improved")   # Δ=+0.06
        self.assertEqual(classify_question(0.5, 0.55), "improved")   # Δ=+0.05 边界

    def test_regressed(self):
        self.assertEqual(classify_question(0.5, 0.44), "regressed")  # Δ=-0.06
        self.assertEqual(classify_question(0.5, 0.45), "regressed")  # Δ=-0.05 边界

    def test_unchanged(self):
        self.assertEqual(classify_question(0.5, 0.54), "unchanged")  # Δ=+0.04
        self.assertEqual(classify_question(0.5, 0.5), "unchanged")
        self.assertEqual(classify_question(0.5, 0.46), "unchanged")  # Δ=-0.04

    def test_none_side_unchanged(self):
        self.assertEqual(classify_question(None, 0.6), "unchanged")
        self.assertEqual(classify_question(0.6, None), "unchanged")
        self.assertEqual(classify_question(None, None), "unchanged")

    def test_threshold_fixed(self):
        self.assertEqual(CLASSIFY_THRESHOLD, 0.05)  # 阈值不可改


class TestDeterministicScore(unittest.TestCase):
    def test_uses_applicable_deterministic_metrics(self):
        rec = _mk_record(metrics={
            **_metric("M1", 1.0), **_metric("M2", 0.8), **_metric("M3", 1.0),
            **_metric("M4", 0.1), "M6": {"actual": {"score": 0.7}},
            **_metric("M8", 0.3), **_metric("M9", 0.4), **_metric("M10", 0.2),
            **_metric("M12", 0.7),
        })
        # M5 不参与；M7 不参与
        s = _deterministic_score(rec)
        self.assertAlmostEqual(s, (1.0 + 0.8 + 1.0 + 0.1 + 0.7 + 0.3 + 0.4 + 0.2 + 0.7) / 9, places=4)

    def test_skips_none(self):
        rec = _mk_record(metrics={**_metric("M1", 1.0), **_metric("M2", None)})
        self.assertEqual(_deterministic_score(rec), 1.0)

    def test_empty_returns_none(self):
        self.assertIsNone(_deterministic_score(_mk_record(metrics={})))

    def test_m5_and_m7_excluded(self):
        rec = _mk_record(metrics={**_metric("M1", 1.0), "M5": {"llm_judge": {"score": 1.0}}})
        self.assertEqual(_deterministic_score(rec), 1.0)  # M5 高分不混入

    def test_metric_keys_fixed(self):
        self.assertEqual(list(DETERMINISTIC_METRIC_KEYS),
                         ["M1", "M2", "M3", "M4", "M6", "M8", "M9", "M10", "M12"])


# ============================================================
# 3. 聚合函数
# ============================================================

class TestAggregation(unittest.TestCase):
    def test_aggregate_metrics(self):
        recs = [
            _mk_record(metrics={**_metric("M1", 1.0), **_metric("M2", 0.8), **_metric("M12", 0.7)}),
            _mk_record(metrics={**_metric("M1", 0.8), **_metric("M2", None), **_metric("M12", 0.9)}),
        ]
        ag = aggregate_metrics(recs)
        self.assertEqual(ag["M1"]["avg"], 0.9)     # (1.0+0.8)/2
        self.assertEqual(ag["M2"]["count"], 1)     # 仅 1 个非 None
        self.assertEqual(ag["M2"]["avg"], 0.8)
        self.assertEqual(ag["M12"]["avg"], 0.8)
        self.assertEqual(ag["M11"]["defined"], False)

    def test_aggregate_metrics_skips_error(self):
        recs = [_mk_record(metrics={**_metric("M1", 1.0)}),
                _mk_record(status="error", metrics={**_metric("M1", 0.0)})]
        ag = aggregate_metrics(recs)
        self.assertEqual(ag["M1"]["avg"], 1.0)
        self.assertEqual(ag["M1"]["count"], 1)

    def test_aggregate_m7_confusion(self):
        recs = [
            _mk_record(qid="a", expected_refusal=True, actual_refusal=True),
            _mk_record(qid="b", expected_refusal=True, actual_refusal=False),   # FN
            _mk_record(qid="c", expected_refusal=False, actual_refusal=False),
            _mk_record(qid="d", expected_refusal=False, actual_refusal=True),   # FP
        ]
        m7 = aggregate_m7(recs)
        self.assertEqual(m7["TP"], 1)
        self.assertEqual(m7["TN"], 1)
        self.assertEqual(m7["FP"], 1)
        self.assertEqual(m7["FN"], 1)

    def test_aggregate_decisions(self):
        recs = [
            _mk_record(qid="a", decision={"decision": "refuse", "reason_code": "DIAGNOSIS_INTENT"}),
            _mk_record(qid="b", decision={"decision": "allow", "reason_code": "RELIABLE_ANSWER"}),
            _mk_record(qid="c", decision={"decision": "revise", "reason_code": "RELIABILITY_MEDIUM_REVISE"}),
            _mk_record(qid="d", status="error"),
        ]
        agg = aggregate_decisions(recs)
        self.assertEqual(agg["distribution"]["refuse"], 1)
        self.assertEqual(agg["distribution"]["allow"], 1)
        self.assertEqual(agg["distribution"]["revise"], 1)
        self.assertEqual(agg["distribution"]["other"], 0)  # error 不计入
        self.assertEqual(agg["reason_codes"]["DIAGNOSIS_INTENT"], 1)


# ============================================================
# 4. Neo4j impact 聚合
# ============================================================

class TestImpact(unittest.TestCase):
    def _rec_pair(self):
        b = [
            _mk_record(qid="q-improve", metrics={**_metric("M1", 0.5), **_metric("M2", 0.5), **_metric("M12", 0.5)}),
            _mk_record(qid="q-regress", metrics={**_metric("M1", 0.9), **_metric("M2", 0.9), **_metric("M12", 0.9)}),
            _mk_record(qid="q-same", metrics={**_metric("M1", 0.7), **_metric("M2", 0.7), **_metric("M12", 0.7)}),
        ]
        n = [
            _mk_record(qid="q-improve", with_neo4j=True, mentions=["二甲双胍"],
                       metrics={**_metric("M1", 0.9), **_metric("M2", 0.9), **_metric("M12", 0.9)},
                       neo4j_provenance={"neo4j_has_evidence": True,
                                         "evidence_sources": {"neo4j_graph": {"nodes": 1, "edges": 1, "triples": 1}},
                                         "coverage_notes": []}),
            _mk_record(qid="q-regress", with_neo4j=True,
                       metrics={**_metric("M1", 0.3), **_metric("M2", 0.3), **_metric("M12", 0.3)},
                       neo4j_provenance={"neo4j_has_evidence": True, "evidence_sources": {}, "coverage_notes": []}),
            _mk_record(qid="q-same", with_neo4j=True,
                       metrics={**_metric("M1", 0.72), **_metric("M2", 0.72), **_metric("M12", 0.72)},
                       neo4j_provenance={"neo4j_has_evidence": False, "evidence_sources": {}, "coverage_notes": []}),
        ]
        return b, n

    def test_improved_regressed_unchanged(self):
        b, n = self._rec_pair()
        impact = build_impact(b, n)
        self.assertIn("q-improve", impact["improved"])
        self.assertIn("q-regress", impact["regressed"])
        self.assertIn("q-same", impact["unchanged"])

    def test_neo4j_hits_tracked(self):
        b, n = self._rec_pair()
        impact = build_impact(b, n)
        hits = impact["per_question_neo4j_hits"]["q-improve"]
        self.assertTrue(hits["neo4j_has_evidence"])
        self.assertEqual(hits["mentions"], ["二甲双胍"])
        self.assertEqual(hits["evidence_sources"]["neo4j_graph"]["triples"], 1)
        self.assertFalse(hits["neo4j_unavailable"])

    def test_m7_flips_separate(self):
        b = [_mk_record(qid="q-a", expected_refusal=True, actual_refusal=True,
                        metrics={**_metric("M1", 0.8), **_metric("M12", 0.8)}),
             _mk_record(qid="q-b", expected_refusal=True, actual_refusal=False,
                        metrics={**_metric("M1", 0.8), **_metric("M12", 0.8)})]
        n = [_mk_record(qid="q-a", with_neo4j=True, expected_refusal=True, actual_refusal=False,
                        metrics={**_metric("M1", 0.9), **_metric("M12", 0.9)},
                        neo4j_provenance={"neo4j_has_evidence": True, "evidence_sources": {}, "coverage_notes": []}),
             _mk_record(qid="q-b", with_neo4j=True, expected_refusal=True, actual_refusal=True,
                        metrics={**_metric("M1", 0.7), **_metric("M12", 0.7)},
                        neo4j_provenance={"neo4j_has_evidence": False, "evidence_sources": {}, "coverage_notes": []})]
        impact = build_impact(b, n)
        # M7 翻转单独统计（不混入 deterministic score 分类）
        self.assertIn("q-a", impact["regressed_by_decision"])     # correct→incorrect
        self.assertIn("q-b", impact["improved_by_decision"])      # incorrect→correct
        # deterministic score: q-a 0.8→0.9 improved；q-b 0.8→0.7 regressed
        self.assertIn("q-a", impact["improved"])
        self.assertIn("q-b", impact["regressed"])

    def test_neo4j_unavailable_recorded(self):
        b = [_mk_record(qid="q-a", metrics={**_metric("M1", 0.8), **_metric("M12", 0.8)})]
        n = [_mk_record(qid="q-a", with_neo4j=True, neo4j_unavailable=True,
                        metrics={**_metric("M1", 0.8), **_metric("M12", 0.8)},
                        neo4j_provenance={"neo4j_has_evidence": False, "evidence_sources": {}, "coverage_notes": []})]
        impact = build_impact(b, n)
        self.assertTrue(impact["per_question_neo4j_hits"]["q-a"]["neo4j_unavailable"])


# ============================================================
# 5. build_report 结构完整性
# ============================================================

class TestBuildReport(unittest.TestCase):
    def test_report_structure(self):
        b = [_mk_record(qid="q1", metrics={**_metric("M1", 1.0), **_metric("M12", 0.7)},
                        decision={"decision": "allow", "reason_code": "RELIABLE_ANSWER"},
                        expected_refusal=False, actual_refusal=False)]
        n = [_mk_record(qid="q1", with_neo4j=True, metrics={**_metric("M1", 1.0), **_metric("M12", 0.8)},
                        decision={"decision": "allow", "reason_code": "RELIABLE_ANSWER"},
                        expected_refusal=False, actual_refusal=False,
                        neo4j_provenance={"neo4j_has_evidence": True, "evidence_sources": {}, "coverage_notes": []})]
        rep = build_report([{"id": "q1"}], b, n)
        for key in ("phase", "dataset_size", "execution", "baseline_without_neo4j",
                    "with_neo4j", "comparison", "neo4j_impact"):
            self.assertIn(key, rep)
        self.assertEqual(rep["dataset_size"], 1)
        self.assertIn("M7", rep["comparison"])
        self.assertIn("M11", rep["comparison"])
        self.assertIn("improved", rep["neo4j_impact"])
        self.assertIn("per_question_neo4j_hits", rep["neo4j_impact"])

    def test_failure_recorded_not_silent(self):
        b = [_mk_record(qid="q1", metrics={**_metric("M1", 1.0)})]
        n = [_mk_record(qid="q1", status="error", error={"type": "RuntimeError", "message": "boom"})]
        rep = build_report([{"id": "q1"}], b, n)
        self.assertEqual(rep["execution"]["neo4j"]["failed"], 1)
        self.assertEqual(rep["execution"]["neo4j"]["failed_details"][0]["question_id"], "q1")
        self.assertEqual(rep["with_neo4j"]["per_question"][0]["status"], "error")


# ============================================================
# 6. fresh 缓存开关（保证 Neo4j 证据真正进入 LLM）
# ============================================================

class TestFreshCacheToggle(unittest.TestCase):
    def test_constructor_accepts_fresh(self):
        ev = Phase6Evaluator(fresh=True)
        self.assertTrue(ev.fresh)
        ev2 = Phase6Evaluator(fresh=False)
        self.assertFalse(ev2.fresh)

    def test_set_query_cache_toggles_global_config(self):
        class FakeKV:
            def __init__(self):
                self.global_config = {"enable_llm_cache": True}
        class FakeRag:
            def __init__(self):
                self.llm_response_cache = FakeKV()
        class FakeEngine:
            def __init__(self):
                self.rag = FakeRag()

        ev = Phase6Evaluator(engine=FakeEngine())
        ev._set_query_cache(False)
        self.assertFalse(ev.engine.rag.llm_response_cache.global_config["enable_llm_cache"])
        ev._set_query_cache(True)
        self.assertTrue(ev.engine.rag.llm_response_cache.global_config["enable_llm_cache"])

    def test_set_query_cache_noop_without_rag(self):
        class BareEngine:
            pass
        ev = Phase6Evaluator(engine=BareEngine())
        ev._set_query_cache(False)  # 不抛异常
        self.assertTrue(True)


# ============================================================
# 7. forbidden dependency（静态扫描）
# ============================================================

class TestNoForbiddenDependency(unittest.TestCase):
    def test_no_direct_llm_embedding_import(self):
        src = Path(_PROJECT_ROOT / "scripts" / "eval_phase6.py").read_text(encoding="utf-8")
        forbidden_direct = ("import torch", "from torch", "import openai", "from openai",
                            "import lightrag", "from lightrag", "sentence_transformers",
                            "import embedder")
        import_lines = [ln for ln in src.splitlines()
                        if ln.lstrip().startswith(("import ", "from "))]
        for line in import_lines:
            for kw in forbidden_direct:
                self.assertNotIn(kw, line, f"eval_phase6.py 不应直接 import '{kw}': {line}")
        # 加载评估集路径是合法引用（非硬编码 qid）；此处仅确认未 import 评估集模块
        self.assertNotIn("from evaluation.datasets", "".join(import_lines))

    def test_no_qid_hardcoding(self):
        import re as _re
        src = Path(_PROJECT_ROOT / "scripts" / "eval_phase6.py").read_text(encoding="utf-8")
        # 剥离模块 docstring（docstring 允许提及数据集 Q001-Q028 作为说明），仅检查代码
        code = _re.sub(r'"""[\s\S]*?"""', "", src, count=1)
        for m in _re.finditer(r"Q0\d{2}", code):
            line = code[: m.start()].count("\n") + 1
            self.fail(f"eval_phase6.py 代码中出现 Q0xx 字面量 (line {line}): {m.group()}")


if __name__ == "__main__":
    unittest.main()
