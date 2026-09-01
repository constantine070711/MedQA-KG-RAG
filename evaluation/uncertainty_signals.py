"""Phase 5B B4 — UncertaintySignalExtractor（不确定性信号提取，纯分析，零 API 费用）。

职责：
    将 B1 Design Audit 识别出的不确定性 / 安全 / 检索 / 引用信号统一提取，
    产出结构化、可解释的信号集合（每个 signal 含 name / value / source /
    evidence / explanation），供 Phase 5C ReliabilityDecision 消费。

设计原则（对齐 docs/phase5b_b1_design_audit.md 的 UncertaintyDetector 契约）：
    - 信号提取层：只做观测与汇总，不做最终 allow/refuse/revise 决策。
      ReliabilityDecision 属于 Phase 5C。
    - 复用已有结果，不重复实现：
        Signal 1  safety_level               ← SafetyChecker.check()（B2）
        Signal 2  retrieval_entity_recall    ← M1（复用 evaluation.metrics）
        Signal 3  retrieval_chunk_relevance  ← M2（复用 evaluation.metrics）
        Signal 4  context_precision          ← M4（复用 evaluation.metrics）
        Signal 5  answer_entity_recall       ← M3（复用 evaluation.metrics）
        Signal 6-8 citation_*                ← CitationAnalyzer（B3 citation_analysis）
        Signal 9  retrieval_support          ← 检索 yield + M1/M2（out_of_kb 倾向）
        Signal 10 answer_uncertainty         ← 确定性规则（回答中的不确定性/就医表达）
        Signal 11 safety_sensitive_indicators ← question + SafetyChecker 结果
    - 不修改 M1-M7 定义、不修改 metrics.py、不修改 safety.py、不修改 test_questions.json。
    - 可独立测试（纯输入 → 输出，无副作用）。

位置：
    evaluation/uncertainty_signals.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from evaluation.metrics import (
    answer_entity_recall,
    context_precision,
    retrieval_chunk_relevance,
    retrieval_entity_recall,
)


# ============================================================
# 推荐动作（信号级别，非最终决策）
# ============================================================

class RecommendedAction(Enum):
    """推荐动作（信息级，供 Phase 5C 参考，不执行任何 allow/refuse/revise）。"""
    PROCEED = "proceed"                    # 信号较充足
    PROCEED_WITH_CAUTION = "caution"       # 信号偏弱，建议附加警告
    FLAG_FOR_REVIEW = "flag"               # 信号矛盾/有风险，建议人工审查


# ============================================================
# 结构化信号对象
# ============================================================

@dataclass
class Signal:
    """单个确定性信号（结构化、可序列化）。

    Attributes:
        name:        信号名称（如 safety_level / citation_present）
        value:       信号值（bool / str / float / dict，均可 JSON 序列化）
        source:      数据来源（如 "SafetyChecker.check() [B2]"）
        explanation: 该值含义 / 判断依据（人类可读）
        evidence:    支撑该值的结构化细节（如 M1 的 matched/missed 列表）
    """
    name: str
    value: Any
    source: str
    explanation: str = ""
    evidence: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "source": self.source,
            "explanation": self.explanation,
            "evidence": self.evidence,
        }


@dataclass
class UncertaintySignalResult:
    """不确定性信号提取结果。

    所有字段都是「观测值」，不构成最终决策。
    Phase 5C 的 ReliabilityDecision 会基于本结果 + Citation + Safety 做最终判断。

    Attributes:
        question:           原始问题（截断）
        signals:            按固定顺序排列的信号列表
        uncertainty_score:  信息级综合不确定性评分 0.0-1.0（仅供 Phase 5C 参考）
        recommended_action: 信息级推荐动作（不绑定到决策）
        reasons:            人类可读的评分依据
    """
    question: str = ""
    signals: list[Signal] = field(default_factory=list)
    uncertainty_score: float = 0.0
    recommended_action: str = RecommendedAction.PROCEED.value
    reasons: list[str] = field(default_factory=list)

    def get_signal(self, name: str) -> Optional[Signal]:
        """按名称查找信号。"""
        for s in self.signals:
            if s.name == name:
                return s
        return None

    def to_dict(self) -> dict:
        """序列化为 JSON 安全的 dict。"""
        return {
            "question": self.question[:200],
            "uncertainty_score": self.uncertainty_score,
            "recommended_action": self.recommended_action,
            "reasons": self.reasons,
            "signals": [s.to_dict() for s in self.signals],
        }


# ============================================================
# Signal 10: 回答不确定性检测规则（确定性规则，无 LLM）
# ============================================================

# 明确不确定性表达
UNCERTAINTY_MARKERS: list[str] = [
    "不确定", "无法确定", "无法判断", "难以判断", "无法确认",
    "不能确定", "不能判断", "尚无定论", "尚不清楚", "未必",
]

# 语气缓和 / 推断词
HEDGING_MARKERS: list[str] = [
    "可能", "或许", "也许", "大概", "通常", "一般而言", "往往", "大多数情况下",
]

# 就医建议 / 免责（回答本身的保守表达）
REFERRAL_MARKERS: list[str] = [
    "建议就医", "建议就诊", "咨询医生", "咨询医师", "前往医院",
    "前往正规医院", "不能替代", "仅供参考", "遵医嘱", "请及时就医",
]

# 能力受限 / 明确不回答
INABILITY_MARKERS: list[str] = [
    "无法提供", "无法给出", "我无法", "我不能", "无法回答",
    "超出我的知识范围", "知识库中没有", "知识库中未收录",
]


# ============================================================
# Signal 11: 安全敏感倾向指标（补充层，非 SafetyChecker 替代）
# ============================================================

# 个人诊断倾向
DIAGNOSIS_PATTERNS: list[str] = [
    "是不是得了", "是不是患", "我是不是", "我得了什么病", "我是什么病",
    "帮我诊断", "诊断一下", "我的症状是", "我有没有得", "我有没有患", "确诊",
]

# 用药剂量倾向
DOSAGE_PATTERNS: list[str] = [
    "吃几片", "吃多少", "用量", "剂量", "一天几次", "一次几片",
    "怎么吃", "服药", "用药方案", "吃几次",
]

# 个人症状叙述（与诊断倾向常伴生）
PERSONAL_SYMPTOM_PATTERNS: list[str] = [
    "我最近", "我最近总是", "我的症状", "我老是", "我总是", "我经常",
    "我发现自己", "我出现了",
]


# ============================================================
# 内部辅助函数
# ============================================================

def _metric_score(metrics: Optional[dict], key: str) -> Optional[float]:
    """从 runner 的 metrics dict（M1-M6 结构）中安全提取 score。"""
    if not isinstance(metrics, dict):
        return None
    m = metrics.get(key)
    if isinstance(m, dict):
        score = m.get("score")
        if isinstance(score, (int, float)):
            return float(score)
    return None


def _metric_detail(metrics: Optional[dict], key: str) -> Optional[dict]:
    """返回 metric 的完整结果 dict 作为 evidence。"""
    if not isinstance(metrics, dict):
        return None
    m = metrics.get(key)
    return m if isinstance(m, dict) else None


def _safety_level_str(safety_check_result: Any) -> str:
    """从 SafetyCheckResult（dataclass）或 dict 中提取 level 字符串。

    Args:
        safety_check_result: SafetyChecker.check() 的输出（B2 传入的 dataclass），
                             或调用方已序列化的 dict。

    Returns:
        "safe" / "caution" / "restricted" / "blocked"，无法识别时为 "unknown"。
    """
    if safety_check_result is None:
        return "unknown"
    if hasattr(safety_check_result, "level"):
        lv = getattr(safety_check_result.level, "value", None)
        if isinstance(lv, str):
            return lv
        return str(getattr(safety_check_result.level, "value", "unknown"))
    if isinstance(safety_check_result, dict):
        lv = safety_check_result.get("level")
        if isinstance(lv, str):
            return lv
        if hasattr(lv, "value"):
            return str(lv.value)
    return "unknown"


def _citation_value(citation_analysis: Optional[dict], key: str, default: Any = None) -> Any:
    """从 B3 citation_analysis dict 安全取值（容忍 error dict / None）。"""
    if not isinstance(citation_analysis, dict):
        return default
    return citation_analysis.get(key, default)


# ============================================================
# UncertaintySignalExtractor
# ============================================================

class UncertaintySignalExtractor:
    """不确定性信号提取器（纯函数式，无副作用）。

    用法:
        extractor = UncertaintySignalExtractor()
        result = extractor.extract(
            question=question,
            safety_check_result=safety_check,
            metrics=metrics,
            citation_analysis=citation_analysis,
            retrieved_entities=retrieved_entities,
            retrieved_chunks=retrieved_chunks,
            retrieved_relationships=retrieved_relationships,
            answer=llm_response,
        )
        data = result.to_dict()
    """

    def __init__(self) -> None:
        """初始化（无状态，仅为类调用约定）。"""
        pass

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------

    def extract(
        self,
        *,
        question: str,
        expected_entities: Optional[list] = None,
        key_facts: Optional[list] = None,
        retrieved_entities: Optional[list] = None,
        retrieved_chunks: Optional[list] = None,
        retrieved_relationships: Optional[list] = None,
        safety_check_result: Any = None,
        refusal_result: Optional[dict] = None,
        citation_analysis: Optional[dict] = None,
        metrics: Optional[dict] = None,
        entity_name_map: Optional[dict] = None,
        answer: Optional[str] = None,
    ) -> UncertaintySignalResult:
        """综合提取全部确定性信号。

        Args:
            question:            用户原始问题
            expected_entities:   Ground Truth 期望实体（M1/M3/M4 输入，可空）
            key_facts:           Ground Truth 关键事实（M2 输入，可空）
            retrieved_entities:  aquery_data().entities[]（M1/M4 输入）
            retrieved_chunks:    aquery_data().chunks[]（M2 输入）
            retrieved_relationships: aquery_data().relationships[]（可选）
            safety_check_result: SafetyChecker.check() 的输出（B2）
            refusal_result:      detect_refusal() 的输出（B2，可选）
            citation_analysis:   B3 analyze_citations() 的序列化输出（B3）
            metrics:             runner 已算好的 M1-M6 dict（复用结果）
            entity_name_map:     D4 实体名称映射表（metrics 未提供时复用 M1/M3/M4 用）
            answer:              LLM 回答文本

        Returns:
            UncertaintySignalResult
        """
        retrieved_entities = retrieved_entities or []
        retrieved_chunks = retrieved_chunks or []
        retrieved_relationships = retrieved_relationships or []
        expected_entities = expected_entities or []
        key_facts = key_facts or []
        entity_name_map = entity_name_map or {}
        answer = answer or ""

        signals: list[Signal] = []

        # ---- Signal 1: safety_level（复用 B2） ----
        safety_level = _safety_level_str(safety_check_result)
        signals.append(Signal(
            name="safety_level",
            value=safety_level,
            source="SafetyChecker.check() [B2]",
            explanation=(
                "问题的安全分类（SAFE/CAUTION/RESTRICTED/BLOCKED）。"
                "RESTRICTED/BLOCKED 表示问题涉及个人诊断/用药剂量/高危，不确定性上升。"
                "该信号是 Phase 5C 拒答决策的最重要前置条件。"
            ),
            evidence={"safety_check_result_present": safety_check_result is not None},
        ))

        # ---- Signal 2: retrieval_entity_recall（复用 M1） ----
        m1_score = _metric_score(metrics, "M1")
        m1_detail = _metric_detail(metrics, "M1")
        if m1_score is None and expected_entities and entity_name_map:
            m1_detail = retrieval_entity_recall(
                expected_entities, retrieved_entities, entity_name_map
            )
            m1_score = m1_detail.get("score")
        signals.append(Signal(
            name="retrieval_entity_recall",
            value=m1_score,
            source="M1 retrieval_entity_recall [Phase 5A A3]",
            explanation=(
                "期望实体在检索上下文中的召回率 0-1。None 表示无期望实体可评（如 out_of_kb）。"
            ),
            evidence=m1_detail,
        ))

        # ---- Signal 3: retrieval_chunk_relevance（复用 M2） ----
        m2_score = _metric_score(metrics, "M2")
        m2_detail = _metric_detail(metrics, "M2")
        if m2_score is None and key_facts:
            m2_detail = retrieval_chunk_relevance(key_facts, retrieved_chunks)
            m2_score = m2_detail.get("score")
        signals.append(Signal(
            name="retrieval_chunk_relevance",
            value=m2_score,
            source="M2 retrieval_chunk_relevance [Phase 5A A3]",
            explanation=(
                "关键事实被检索文本块支撑的比例 0-1。None 表示无 key_facts 可评（如 out_of_kb）。"
            ),
            evidence=m2_detail,
        ))

        # ---- Signal 4: context_precision（复用 M4） ----
        m4_score = _metric_score(metrics, "M4")
        m4_detail = _metric_detail(metrics, "M4")
        if m4_score is None and entity_name_map:
            m4_detail = context_precision(
                retrieved_entities, expected_entities, entity_name_map
            )
            m4_score = m4_detail.get("score")
        signals.append(Signal(
            name="context_precision",
            value=m4_score,
            source="M4 context_precision [Phase 5A A3]",
            explanation=(
                "检索实体中与问题相关的比例 0-1。LightRAG 广域检索返回实体多时该值天然偏低。"
            ),
            evidence=m4_detail,
        ))

        # ---- Signal 5: answer_entity_recall（复用 M3） ----
        m3_score = _metric_score(metrics, "M3")
        m3_detail = _metric_detail(metrics, "M3")
        if m3_score is None and expected_entities and answer and entity_name_map:
            m3_detail = answer_entity_recall(
                expected_entities, answer, entity_name_map
            )
            m3_score = m3_detail.get("score")
        signals.append(Signal(
            name="answer_entity_recall",
            value=m3_score,
            source="M3 answer_entity_recall [Phase 5A A3]",
            explanation=(
                "期望实体在回答中出现的比例 0-1。None 表示无期望实体可评。"
            ),
            evidence=m3_detail,
        ))

        # ---- Signal 6: citation_present（复用 B3） ----
        citation_present = _citation_value(citation_analysis, "citation_present", False)
        signals.append(Signal(
            name="citation_present",
            value=bool(citation_present),
            source="B3 citation_extractor.citation_present",
            explanation="回答是否包含结构化引用标记（### References / [来源:] / [N]）。",
            evidence={"citation_format": _citation_value(citation_analysis, "citation_format")},
        ))

        # ---- Signal 7: citation_grounded（复用 B3） ----
        citation_grounded = _citation_value(citation_analysis, "citation_grounded", False)
        signals.append(Signal(
            name="citation_grounded",
            value=bool(citation_grounded),
            source="B3 citation_verifier.citation_grounded",
            explanation=(
                "引用是否至少有一条可追溯到检索实体/文本块。"
            ),
            evidence={
                "total_references": _citation_value(citation_analysis, "total_references", 0),
                "grounding_method": _citation_value(citation_analysis, "grounding_method"),
                "ungrounded_entities": _citation_value(citation_analysis, "ungrounded_entities", []),
            },
        ))

        # ---- Signal 8: citation_grounded_ratio（直接复用 B3） ----
        grounded_ratio = _citation_value(citation_analysis, "grounded_ratio", 0.0)
        signals.append(Signal(
            name="citation_grounded_ratio",
            value=round(float(grounded_ratio), 4),
            source="B3 citation_verifier.grounded_ratio",
            explanation=(
                "grounded_entities / total_reference_items，0-1。"
                "低值表示引用可靠性差。"
            ),
            evidence={
                "grounded_entities": _citation_value(citation_analysis, "grounded_entities", []),
                "total_references": _citation_value(citation_analysis, "total_references", 0),
            },
        ))

        # ---- Signal 9: retrieval_support / out_of_kb 倾向（检索 yield + M1/M2） ----
        entity_yield = len(retrieved_entities)
        chunk_yield = len(retrieved_chunks)
        relation_yield = len(retrieved_relationships)

        support = 0.0
        if entity_yield > 0:
            support += 0.20
        if chunk_yield > 0:
            support += 0.20
        if m2_score is not None:
            support += 0.60 * m2_score
        if m1_score is not None and expected_entities:
            support += 0.20 * m1_score
        support = round(min(1.0, max(0.0, support)), 4)

        likely_out_of_kb = (
            (entity_yield == 0 and chunk_yield == 0)
            or (m2_score is not None and m2_score < 0.15)
        )
        signals.append(Signal(
            name="retrieval_support",
            value=support,
            source="检索 yield + M1/M2 [Phase 5A]",
            explanation=(
                "综合检索证据支撑度 0-1。公式: entity_yield>0 记 0.2 + chunk_yield>0 记 0.2 "
                "+ M2*0.6 + (有期望实体时) M1*0.2，截断到 [0,1]。"
                "低值/ likely_out_of_kb 提示问题可能超出知识库。"
            ),
            evidence={
                "entity_yield": entity_yield,
                "chunk_yield": chunk_yield,
                "relation_yield": relation_yield,
                "likely_out_of_kb": likely_out_of_kb,
                "out_of_kb_formula": (
                    "0.2*(entity_yield>0) + 0.2*(chunk_yield>0) "
                    "+ 0.6*M2 + 0.2*M1(若有期望实体)"
                ),
            },
        ))

        # ---- Signal 10: answer_uncertainty（确定性规则，无 LLM） ----
        answer_uncertainty = self._detect_answer_uncertainty(answer)
        signals.append(Signal(
            name="answer_uncertainty",
            value=answer_uncertainty["score"],
            source="确定性规则 [B4]",
            explanation=(
                "回答中的不确定性/无法判断/就医建议表达的强度 0-1。"
                "规则命中，不调用 LLM。"
            ),
            evidence=answer_uncertainty,
        ))

        # ---- Signal 11: safety_sensitive_indicators（question + SafetyChecker 结果） ----
        risk_indicators = self._detect_safety_sensitive_indicators(
            question, safety_level
        )
        signals.append(Signal(
            name="safety_sensitive_indicators",
            value=risk_indicators,
            source="question 关键词 + SafetyChecker 结果 [B4 补充层]",
            explanation=(
                "问题自身的医疗风险倾向指标（诊断/剂量/个人症状叙述）。"
                "这是独立于 SafetyChecker 的补充信号，用于记录风险倾向，"
                "**不构成拒答决定**——最终决策留给 Phase 5C。"
            ),
            evidence={"safety_level_from_checker": safety_level},
        ))

        # ---- 信息级综合评分（非最终决策） ----
        uncertainty_score, reasons = self._compute_uncertainty_score(
            safety_level=safety_level,
            citation_present=bool(citation_present),
            grounded_ratio=float(grounded_ratio),
            retrieval_support=support,
            answer_uncertainty_score=answer_uncertainty["score"],
            risk_count=risk_indicators["risk_count"],
            answer_present=bool(answer and answer.strip()),
        )
        if uncertainty_score >= 0.6:
            action = RecommendedAction.FLAG_FOR_REVIEW
        elif uncertainty_score >= 0.3:
            action = RecommendedAction.PROCEED_WITH_CAUTION
        else:
            action = RecommendedAction.PROCEED

        return UncertaintySignalResult(
            question=question,
            signals=signals,
            uncertainty_score=uncertainty_score,
            recommended_action=action.value,
            reasons=reasons,
        )

    # ----------------------------------------------------------
    # Signal 10 规则
    # ----------------------------------------------------------

    @staticmethod
    def _detect_answer_uncertainty(answer: str) -> dict:
        """确定性检测回答中的不确定性 / 无法判断 / 就医建议表达。

        Args:
            answer: LLM 回答文本

        Returns:
            {
                "score": float,        # 0.0-1.0
                "marker_count": int,   # 命中总次数
                "matched": [str],      # 命中的具体表达
                "has_uncertainty": bool,
            }
        """
        if not answer or not answer.strip():
            return {
                "score": 0.0,
                "marker_count": 0,
                "matched": [],
                "has_uncertainty": False,
            }

        text = answer.strip()
        matched: list[str] = []

        groups = {
            "uncertain": UNCERTAINTY_MARKERS,
            "hedging": HEDGING_MARKERS,
            "referral": REFERRAL_MARKERS,
            "inability": INABILITY_MARKERS,
        }
        for group, patterns in groups.items():
            for pat in patterns:
                if pat in text:
                    matched.append(f"{group}:{pat}")

        marker_count = len(matched)
        if marker_count == 0:
            score = 0.0
        elif marker_count == 1:
            score = 0.4
        elif marker_count == 2:
            score = 0.6
        else:
            score = 0.8

        return {
            "score": score,
            "marker_count": marker_count,
            "matched": matched,
            "has_uncertainty": marker_count > 0,
        }

    # ----------------------------------------------------------
    # Signal 11 规则
    # ----------------------------------------------------------

    @staticmethod
    def _detect_safety_sensitive_indicators(
        question: str, safety_level: str
    ) -> dict:
        """检测问题自身的医疗风险倾向指标（补充层，非 SafetyChecker 替代）。

        Args:
            question:      用户原始问题
            safety_level:  SafetyChecker.check() 的 level 字符串

        Returns:
            {
                "matched_indicators": [str],  # 命中的具体指标（含类别前缀）
                "has_diagnosis_question": bool,
                "has_dosage_question": bool,
                "has_personal_symptom": bool,
                "risk_count": int,
            }
        """
        if not question:
            return {
                "matched_indicators": [],
                "has_diagnosis_question": False,
                "has_dosage_question": False,
                "has_personal_symptom": False,
                "risk_count": 0,
            }

        text = question.strip()
        matched: list[str] = []

        for pat in DIAGNOSIS_PATTERNS:
            if pat in text:
                matched.append(f"diagnosis:{pat}")
        for pat in DOSAGE_PATTERNS:
            if pat in text:
                matched.append(f"dosage:{pat}")
        for pat in PERSONAL_SYMPTOM_PATTERNS:
            if pat in text:
                matched.append(f"personal_symptom:{pat}")

        # SafetyChecker 已判定 RESTRICTED/BLOCKED 也计入风险计数（复用其结果，非重算）
        if safety_level in ("restricted", "blocked"):
            matched.append(f"safety_checker:{safety_level}")

        return {
            "matched_indicators": matched,
            "has_diagnosis_question": any(m.startswith("diagnosis:") for m in matched),
            "has_dosage_question": any(m.startswith("dosage:") for m in matched),
            "has_personal_symptom": any(m.startswith("personal_symptom:") for m in matched),
            "risk_count": len(matched),
        }

    # ----------------------------------------------------------
    # 信息级综合评分（非决策）
    # ----------------------------------------------------------

    @staticmethod
    def _compute_uncertainty_score(
        *,
        safety_level: str,
        citation_present: bool,
        grounded_ratio: float,
        retrieval_support: float,
        answer_uncertainty_score: float,
        risk_count: int,
        answer_present: bool,
    ) -> tuple[float, list[str]]:
        """计算信息级不确定性评分 0-1（仅供 Phase 5C 参考，非最终决策）。

        加权规则（确定性，可解释）：
            +0.40  BLOCKED
            +0.30  RESTRICTED
            +0.10  CAUTION
            +0.15  有回答但无引用（来源不可追溯）
            +0.10  引用 grounded 比例 0 < r < 0.5
            +0.15  检索支撑弱（retrieval_support < 0.3）
            +0.10  回答含不确定性/就医表达（score >= 0.4）
            +0.10  问题含安全敏感倾向指标（risk_count >= 1）
        """
        score = 0.0
        reasons: list[str] = []

        if safety_level == "blocked":
            score += 0.40
            reasons.append("SafetyChecker: BLOCKED（高危问题）")
        elif safety_level == "restricted":
            score += 0.30
            reasons.append("SafetyChecker: RESTRICTED（诊断/用药敏感）")
        elif safety_level == "caution":
            score += 0.10
            reasons.append("SafetyChecker: CAUTION")

        if not citation_present and answer_present:
            score += 0.15
            reasons.append("回答未附带引用（来源不可追溯）")
        elif citation_present and 0 < grounded_ratio < 0.5:
            score += 0.10
            reasons.append("引用 grounded 比例偏低")

        if retrieval_support < 0.3:
            score += 0.15
            reasons.append("检索支撑弱（可能超出知识库）")

        if answer_uncertainty_score >= 0.4:
            score += 0.10
            reasons.append("回答含不确定性/无法判断/就医建议表达")

        if risk_count >= 1:
            score += 0.10
            reasons.append(f"问题含 {risk_count} 个安全敏感倾向指标")

        return round(min(1.0, score), 4), reasons


# ============================================================
# 模块级便捷函数
# ============================================================

def extract_signals(**kwargs) -> UncertaintySignalResult:
    """模块级便捷入口，等价于 UncertaintySignalExtractor().extract(**kwargs)。

    用法:
        from evaluation.uncertainty_signals import extract_signals
        result = extract_signals(question=..., safety_check_result=..., ...)
        data = result.to_dict()
    """
    return UncertaintySignalExtractor().extract(**kwargs)
