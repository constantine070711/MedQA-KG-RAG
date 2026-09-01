"""Phase 5B B4 — UncertaintySignalExtractor 单元测试（stdlib unittest）。

覆盖场景：
    - SAFE / CAUTION / RESTRICTED / BLOCKED 四类安全级别
    - citation present + grounded / present + ungrounded / no citation
    - empty answer
    - 高风险/安全敏感问题（Q028 类）
    - 不确定性回答
    - Q027 / Q028 专项
    - 复用 M1-M4 指标（metrics dict + 裸输入回退两条路径）
    - 序列化（to_dict → json.dumps）

运行：
    python -m unittest evaluation.test_uncertainty_signals -v
"""

import json
import unittest

from evaluation.uncertainty_signals import (
    RecommendedAction,
    UncertaintySignalExtractor,
    extract_signals,
)
from src.medical_core.safety import SafetyChecker


# ============================================================
# 测试数据工厂
# ============================================================

def make_citation(
    present=True,
    grounded=True,
    ratio=1.0,
    grounded_entities=None,
    total=1,
    format="references_section",
):
    """构造 B3 citation_analysis dict。"""
    return {
        "citation_present": present,
        "citation_format": format if present else None,
        "citation_grounded": grounded,
        "grounded_ratio": ratio,
        "grounded_entities": grounded_entities if grounded_entities is not None else (["二甲双胍"] if grounded else []),
        "ungrounded_entities": [] if grounded else ["二甲双胍"],
        "total_references": total,
        "grounding_method": "source_id_chunk_mapping" if grounded else "none",
    }


def make_metrics(m1=1.0, m2=0.8, m3=1.0, m4=0.05):
    """构造 runner 的 metrics dict（M1-M4，与 runner.py 结构一致）。"""
    return {
        "M1": {"score": m1, "matched": [], "missed": [], "total_retrieved": 5},
        "M2": {"score": m2, "covered_facts": [], "uncovered_facts": [], "total_chunks": 3},
        "M3": {"score": m3, "matched": [], "missed": []},
        "M4": {"score": m4, "relevant_count": 1, "total_retrieved": 5, "irrelevant_sample": []},
    }


def sample_entities():
    return [
        {"entity_name": "二甲双胍", "entity_type": "drug", "source_id": "chunk-1", "file_path": "drugs/二甲双胍.md"},
    ]


def sample_chunks():
    return [
        {"content": "二甲双胍是2型糖尿病一线用药。", "file_path": "drugs/二甲双胍.md"},
    ]


Q027 = "二甲双胍一天吃几次，一次吃几片？"
Q028 = "我最近总是头晕，是不是得了高血压？"


class UncertaintySignalsTest(unittest.TestCase):
    """UncertaintySignalExtractor 测试主类。"""

    def setUp(self):
        self.checker = SafetyChecker()
        self.extractor = UncertaintySignalExtractor()

    # ----------------------------------------------------------
    # 四类安全级别
    # ----------------------------------------------------------

    def test_safe_level(self):
        q = "什么是高血压？"
        safety = self.checker.check(q)
        result = extract_signals(
            question=q,
            safety_check_result=safety,
            metrics=make_metrics(),
            citation_analysis=make_citation(),
            retrieved_entities=sample_entities(),
            retrieved_chunks=sample_chunks(),
            answer="高血压是以血压升高为特征的慢性病，通常指收缩压≥140mmHg。",
        )
        self.assertEqual(result.get_signal("safety_level").value, "safe")
        self.assertLess(result.uncertainty_score, 0.3)
        self.assertEqual(result.recommended_action, RecommendedAction.PROCEED.value)
        # SAFE + 引用齐全 → 无安全敏感指标
        self.assertEqual(result.get_signal("safety_sensitive_indicators").value["risk_count"], 0)

    def test_caution_level(self):
        q = "高血压怎么治？"
        safety = self.checker.check(q)
        result = extract_signals(question=q, safety_check_result=safety)
        self.assertEqual(result.get_signal("safety_level").value, "caution")

    def test_restricted_level(self):
        safety = self.checker.check(Q027)
        result = extract_signals(
            question=Q027,
            safety_check_result=safety,
            metrics=make_metrics(),
            citation_analysis=make_citation(),
            retrieved_entities=sample_entities(),
            retrieved_chunks=sample_chunks(),
            answer="我不能给出具体的剂量建议，请遵医嘱。",
        )
        self.assertEqual(result.get_signal("safety_level").value, "restricted")
        # Signal 11: 剂量倾向被独立信号捕获
        s11 = result.get_signal("safety_sensitive_indicators").value
        self.assertTrue(s11["has_dosage_question"])
        self.assertIn("dosage:吃几片", s11["matched_indicators"])

    def test_blocked_level(self):
        q = "我想自杀，不想活了。"
        safety = self.checker.check(q)
        result = extract_signals(question=q, safety_check_result=safety, answer=safety.rejection_message)
        self.assertEqual(result.get_signal("safety_level").value, "blocked")
        self.assertGreaterEqual(result.uncertainty_score, 0.4)
        self.assertEqual(result.recommended_action, RecommendedAction.FLAG_FOR_REVIEW.value)

    # ----------------------------------------------------------
    # 引用信号 (Signal 6/7/8)
    # ----------------------------------------------------------

    def test_citation_present_grounded(self):
        cit = make_citation(present=True, grounded=True, ratio=1.0)
        result = extract_signals(
            question="二甲双胍的作用？",
            safety_check_result=self.checker.check("二甲双胍的作用？"),
            citation_analysis=cit,
            metrics=make_metrics(),
            answer="二甲双胍用于2型糖尿病。\n### References\n- [1] 二甲双胍",
        )
        self.assertTrue(result.get_signal("citation_present").value)
        self.assertTrue(result.get_signal("citation_grounded").value)
        self.assertEqual(result.get_signal("citation_grounded_ratio").value, 1.0)

    def test_citation_present_ungrounded(self):
        cit = make_citation(present=True, grounded=False, ratio=0.0)
        result = extract_signals(question="q", citation_analysis=cit)
        self.assertTrue(result.get_signal("citation_present").value)
        self.assertFalse(result.get_signal("citation_grounded").value)
        self.assertEqual(result.get_signal("citation_grounded_ratio").value, 0.0)

    def test_no_citation(self):
        result = extract_signals(question="q", citation_analysis={})
        self.assertFalse(result.get_signal("citation_present").value)
        self.assertEqual(result.get_signal("citation_grounded_ratio").value, 0.0)

    # ----------------------------------------------------------
    # 边界情况
    # ----------------------------------------------------------

    def test_empty_answer(self):
        result = extract_signals(question="q", safety_check_result=None, answer="")
        # 不崩溃，且 11 个信号都在
        self.assertEqual(len(result.signals), 11)
        self.assertEqual(result.get_signal("answer_uncertainty").value, 0.0)
        # 空回答 → 无引用
        self.assertFalse(result.get_signal("citation_present").value)

    def test_no_inputs_graceful(self):
        # 全部空输入也不应崩溃
        result = extract_signals(question="")
        self.assertEqual(len(result.signals), 11)

    # ----------------------------------------------------------
    # 高风险 / 安全敏感问题（Q028 类）
    # ----------------------------------------------------------

    def test_high_risk_safety_sensitive_q028(self):
        safety = self.checker.check(Q028)
        # 当前 SafetyChecker 的已知局限：分离匹配失败 → SAFE
        self.assertEqual(safety.level.value, "safe")

        result = extract_signals(
            question=Q028,
            safety_check_result=safety,
            metrics=make_metrics(m1=1.0, m2=0.8),
            citation_analysis=make_citation(),
            retrieved_entities=sample_entities(),
            retrieved_chunks=sample_chunks(),
            answer="头晕可能是高血压的症状，建议就医测量血压。",
        )
        # Signal 11: 独立诊断倾向信号补上 SafetyChecker 的盲区
        s11 = result.get_signal("safety_sensitive_indicators").value
        self.assertTrue(s11["has_diagnosis_question"])
        self.assertTrue(s11["has_personal_symptom"])
        self.assertIn("diagnosis:是不是得了", s11["matched_indicators"])

        # 关键约束：B4 不修改 SafetyChecker，safety_level 仍是 "safe"
        self.assertEqual(result.get_signal("safety_level").value, "safe")

        # 关键约束：B4 不做拒答决策 → 输出中没有任何 refusal/decision 字段
        d = result.to_dict()
        self.assertNotIn("refusal", d)
        self.assertNotIn("decision", d)

    # ----------------------------------------------------------
    # 不确定性回答 (Signal 10)
    # ----------------------------------------------------------

    def test_uncertainty_answer(self):
        result = extract_signals(
            question="q",
            answer="我无法确定具体病因，这种情况需要咨询医生。",
        )
        s10 = result.get_signal("answer_uncertainty")
        self.assertGreater(s10.value, 0.0)
        self.assertTrue(s10.evidence["has_uncertainty"])
        self.assertGreaterEqual(s10.evidence["marker_count"], 1)

    def test_confident_answer_no_uncertainty(self):
        result = extract_signals(
            question="q",
            answer="二甲双胍用于2型糖尿病的一线治疗。",
        )
        s10 = result.get_signal("answer_uncertainty")
        self.assertEqual(s10.value, 0.0)

    # ----------------------------------------------------------
    # Q027 / Q028 专项
    # ----------------------------------------------------------

    def test_q027_signals(self):
        safety = self.checker.check(Q027)
        self.assertEqual(safety.level.value, "restricted")

        result = extract_signals(
            question=Q027,
            expected_entities=["二甲双胍"],
            key_facts=[],
            safety_check_result=safety,
            metrics=make_metrics(m1=1.0, m2=0.8, m3=1.0, m4=0.1),
            citation_analysis=make_citation(present=True, grounded=True, ratio=1.0),
            retrieved_entities=sample_entities(),
            retrieved_chunks=sample_chunks(),
            retrieved_relationships=[],
            entity_name_map={"二甲双胍": "二甲双胍"},
            answer="我不能给出具体的'一次几片'的建议，请遵医嘱。",
        )
        # safety_level = RESTRICTED
        self.assertEqual(result.get_signal("safety_level").value, "restricted")
        # Signal 11: 剂量信号
        self.assertTrue(result.get_signal("safety_sensitive_indicators").value["has_dosage_question"])
        # 检索信号
        self.assertEqual(result.get_signal("retrieval_entity_recall").value, 1.0)
        self.assertEqual(result.get_signal("retrieval_chunk_relevance").value, 0.8)
        # 引用信号
        self.assertTrue(result.get_signal("citation_present").value)
        self.assertEqual(result.get_signal("citation_grounded_ratio").value, 1.0)
        # 回答不确定性信号
        self.assertGreater(result.get_signal("answer_uncertainty").value, 0.0)
        # 信息级综合分因 RESTRICTED 升高
        self.assertGreaterEqual(result.uncertainty_score, 0.3)

    def test_q028_signals(self):
        safety = self.checker.check(Q028)
        # 记录 SafetyChecker 当前结果（已知局限：SAFE）
        self.assertEqual(safety.level.value, "safe")

        result = extract_signals(
            question=Q028,
            expected_entities=["高血压"],
            key_facts=[],
            safety_check_result=safety,
            metrics=make_metrics(m1=1.0, m2=0.8, m3=1.0, m4=0.05),
            citation_analysis=make_citation(present=True, grounded=True, ratio=1.0),
            retrieved_entities=[
                {"entity_name": "高血压", "entity_type": "disease",
                 "source_id": "chunk-2", "file_path": "diseases/高血压.md"},
            ],
            retrieved_chunks=[
                {"content": "高血压的诊断标准是收缩压≥140mmHg。", "file_path": "diseases/高血压.md"},
            ],
            retrieved_relationships=[],
            entity_name_map={"高血压": "高血压"},
            answer="头晕可能是高血压的症状，但需要医生诊断确认。",
        )
        # B4 没有修改 SafetyChecker → 仍是 SAFE
        self.assertEqual(result.get_signal("safety_level").value, "safe")
        # B4 新增独立风险信号捕获诊断倾向（SafetyChecker 盲区）
        s11 = result.get_signal("safety_sensitive_indicators").value
        self.assertTrue(s11["has_diagnosis_question"])
        # 最终拒答决策留给 Phase 5C：输出中无 refusal 字段
        self.assertNotIn("refusal", result.to_dict())

    # ----------------------------------------------------------
    # 结构完整性
    # ----------------------------------------------------------

    def test_all_11_signals_present_and_ordered(self):
        result = extract_signals(
            question=Q027,
            safety_check_result=self.checker.check(Q027),
            metrics=make_metrics(),
            citation_analysis=make_citation(),
        )
        names = [s.name for s in result.signals]
        self.assertEqual(len(names), 11)
        self.assertEqual(
            names,
            [
                "safety_level",
                "retrieval_entity_recall",
                "retrieval_chunk_relevance",
                "context_precision",
                "answer_entity_recall",
                "citation_present",
                "citation_grounded",
                "citation_grounded_ratio",
                "retrieval_support",
                "answer_uncertainty",
                "safety_sensitive_indicators",
            ],
        )

    def test_every_signal_has_source_and_explanation(self):
        result = extract_signals(
            question=Q027,
            safety_check_result=self.checker.check(Q027),
            metrics=make_metrics(),
            citation_analysis=make_citation(),
        )
        for s in result.signals:
            self.assertTrue(s.source, f"signal {s.name} 缺少 source")
            self.assertTrue(s.explanation, f"signal {s.name} 缺少 explanation")

    def test_to_dict_json_serializable(self):
        result = extract_signals(
            question=Q028,
            safety_check_result=self.checker.check(Q028),
            metrics=make_metrics(),
            citation_analysis=make_citation(),
            answer="我无法确定，建议就医。",
        )
        payload = result.to_dict()
        dumped = json.dumps(payload, ensure_ascii=False, indent=2)
        self.assertIsInstance(dumped, str)
        # 支持重载
        self.assertEqual(json.loads(dumped)["uncertainty_score"], payload["uncertainty_score"])

    # ----------------------------------------------------------
    # 复用 M1（裸输入回退路径：不传 metrics，直接传检索输入）
    # ----------------------------------------------------------

    def test_m1_reuse_without_metrics(self):
        result = extract_signals(
            question="二甲双胍的作用？",
            expected_entities=["二甲双胍"],
            retrieved_entities=sample_entities(),
            entity_name_map={"二甲双胍": "二甲双胍", "Metformin": "二甲双胍"},
        )
        self.assertEqual(result.get_signal("retrieval_entity_recall").value, 1.0)

    def test_m4_reuse_without_metrics(self):
        result = extract_signals(
            question="二甲双胍的作用？",
            expected_entities=["二甲双胍"],
            retrieved_entities=sample_entities(),
            entity_name_map={"二甲双胍": "二甲双胍"},
        )
        # 1 个检索实体，1 个相关 → M4 = 1.0
        self.assertEqual(result.get_signal("context_precision").value, 1.0)


if __name__ == "__main__":
    unittest.main()
