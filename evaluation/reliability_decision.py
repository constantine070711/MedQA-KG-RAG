"""Phase 5C C1 — ReliabilityDecision（可靠性决策层，纯函数，零 LLM / 零 DB / 零副作用）。

职责：
    消费 B2/B3/B4/B5 已产出的观测信号，按 docs/phase5c_design_audit.md 审计通过的
    R1-R9 通用规则（P1→P6 优先级）做出最终可靠性决策：
        ALLOW  /  REVISE  /  REFUSE

设计原则（严格对齐 docs/phase5c_design_audit.md）：
    - 决策层独立于测量层（B4 uncertainty_signals / B5 metrics）：只读，不修改任何输入。
    - 只消费已存在数据：不重新调用 SafetyChecker、不重算 M8-M12、不重新解析 citation、
      不访问数据库、不调用 LLM。
    - 严禁 qid / question-id 特判（本模块不含任何题号）。
    - P1/P2（安全/个人风险）的 REFUSE 永不被高 M12 覆盖；
      高可靠性只能软化 CAUTION（R7 的 REVISE→ALLOW 升级），不能把 REFUSE 降为 ALLOW。
    - REVISE 仅是决策标签：不执行二次 LLM generation，不修改 answer。
    - final_refusal = decision==REFUSE OR (refusal_result.is_refusal AND NOT veto)；
      veto 仅覆盖 detect_refusal 的 emergency_response / short_referral 分支，
      且仅当决策层判定该题可答（ALLOW/REVISE）时——matched / mental_health 拒答不会被误伤。

规则摘要（R1-R9，优先级从上到下）：
    P1  HARD SAFETY    R1 BLOCKED -> REFUSE / SAFETY_BLOCKED
    P2  PERSONAL RISK  R2 restricted + (dosage|diagnosis) -> REFUSE / SAFETY_RESTRICTED_PERSONAL
                       R3 has_diagnosis_question -> REFUSE / DIAGNOSIS_INTENT   （SafetyChecker 盲区兜底）
                       R4 has_dosage_question -> REFUSE / DOSAGE_REQUEST
    P3  OUT-OF-KB      R5 likely_out_of_kb + 无检索支撑 -> REFUSE / OUT_OF_KB
    P4  RELIABILITY    R6 M12 < 0.40 -> REFUSE / RELIABILITY_LOW
    P5  REVISE         R7 M12 medium + 引用缺失/高不确定 -> REVISE / RELIABILITY_MEDIUM_REVISE
                          （引用可靠 + 低不确定 -> 升级 ALLOW / RELIABILITY_MEDIUM_ALLOW）
                       R8 personal_symptom（无诊断意图）-> REVISE / PERSONAL_SYMPTOM_REVISE
    P6  ALLOW          R9 其余 -> ALLOW / RELIABLE_ANSWER

位置：
    evaluation/reliability_decision.py
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


# ============================================================
# 决策状态
# ============================================================

class Decision(Enum):
    """最终决策状态。"""
    ALLOW = "allow"             # 正常回答
    REVISE = "revise"           # 回答但需修订（免责/就医建议），非拒答
    REFUSE = "refuse"           # 不提供实质性医学内容


# ============================================================
# 规则置信度（元数据，非决策阈值；审计未约束，按优先级语义确定）
# 按 reason_code 索引（R7 有 REVISE / ALLOW 两个 variant，故不以 rule_id 为键）。
# ============================================================

_CONFIDENCE_BY_REASON: dict[str, float] = {
    "SAFETY_BLOCKED": 1.0,                 # 硬性安全拦截
    "SAFETY_RESTRICTED_PERSONAL": 0.95,    # restricted + 个人风险
    "DIAGNOSIS_INTENT": 0.95,              # 诊断意图
    "DOSAGE_REQUEST": 0.95,                # 剂量请求
    "OUT_OF_KB": 0.80,                     # out-of-KB
    "RELIABILITY_LOW": 0.80,               # 可靠性过低
    "RELIABILITY_MEDIUM_REVISE": 0.65,     # 中等可靠性修订
    "RELIABILITY_MEDIUM_ALLOW": 0.85,      # 中等可靠性升级允许
    "PERSONAL_SYMPTOM_REVISE": 0.70,       # 个人症状（无诊断）
    "RELIABLE_ANSWER": 0.90,               # 默认允许
}


# ============================================================
# 输出对象
# ============================================================

@dataclass
class ReliabilityDecision:
    """可靠性决策输出（JSON 可序列化）。

    Attributes:
        decision:               "allow" / "revise" / "refuse"
        reason_code:            主触发规则的原因码（如 SAFETY_BLOCKED / DIAGNOSIS_INTENT）
        reason:                 人类可读的主决策原因
        triggered_rules:        本次输入下条件成立的规则 id（按优先级排序，含 R9）
        confidence:             决策置信度 0-1（规则级，非阈值）
        safety_level:           透传 SafetyChecker level（safe/caution/restricted/blocked/unknown）
        is_final_refusal:       最终是否构成拒答（decision REFUSE，或 detect_refusal 且未被 veto）
        reliability_score:      透传 M12.score（缺失 None）
        uncertainty_level:      M8.level（high/medium/low，缺失 None）
        citation_quality:       M9.score（缺失 None）
        evidence:               触发依据关键值快照
        explanation:            决策说明（组合 reason + triggered_rules）
        disclaimer_needed:      REFUSE/REVISE 时为 True
        cancel_posthoc_refusal: veto 标记（是否取消 detect_refusal 的 emergency/short_referral 拒答）
        note:                   降级/缺失说明
    """
    decision: str
    reason_code: str
    reason: str
    triggered_rules: list[str]
    confidence: float
    safety_level: str
    is_final_refusal: bool
    reliability_score: Optional[float]
    uncertainty_level: Optional[str]
    citation_quality: Optional[float]
    evidence: dict
    explanation: str
    disclaimer_needed: bool
    cancel_posthoc_refusal: bool
    note: Optional[str]

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "triggered_rules": self.triggered_rules,
            "confidence": self.confidence,
            "safety_level": self.safety_level,
            "is_final_refusal": self.is_final_refusal,
            "reliability_score": self.reliability_score,
            "uncertainty_level": self.uncertainty_level,
            "citation_quality": self.citation_quality,
            "evidence": self.evidence,
            "explanation": self.explanation,
            "disclaimer_needed": self.disclaimer_needed,
            "cancel_posthoc_refusal": self.cancel_posthoc_refusal,
            "note": self.note,
        }


# ============================================================
# 只读信号提取辅助函数（不重新计算任何测量）
# ============================================================

def _extract_safety_level(safety_check: Any) -> str:
    """从 SafetyCheckResult（dataclass）或 dict 中提取 level 字符串（只读）。

    与 metrics.py / uncertainty_signals.py 的同类辅助一致：仅读取既有结果，
    不重新调用 SafetyChecker。
    """
    if safety_check is None:
        return "unknown"
    if hasattr(safety_check, "level"):
        lv = getattr(safety_check.level, "value", None)
        if isinstance(lv, str):
            return lv
        return str(lv) if lv is not None else "unknown"
    if isinstance(safety_check, dict):
        lv = safety_check.get("level")
        if isinstance(lv, str):
            return lv
        if hasattr(lv, "value"):
            return str(lv.value)
    return "unknown"


def _metric_score(metric: Optional[dict]) -> Optional[float]:
    """从 metric dict（M8/M9/M10/M12 产物）中提取 score；缺失/error 返回 None。"""
    if isinstance(metric, dict):
        s = metric.get("score")
        return float(s) if isinstance(s, (int, float)) else None
    return None


def _metric_level(metric: Optional[dict]) -> Optional[str]:
    """从 metric dict 中提取 level；缺失返回 None。"""
    if isinstance(metric, dict):
        lv = metric.get("level")
        return lv if isinstance(lv, str) else None
    return None


def _find_signal(uncertainty_analysis: Optional[dict], name: str) -> Optional[dict]:
    """从 B4 uncertainty_analysis.signals[] 中按 name 取完整信号 dict。"""
    if not isinstance(uncertainty_analysis, dict):
        return None
    for s in uncertainty_analysis.get("signals", []):
        if isinstance(s, dict) and s.get("name") == name:
            return s
    return None


def _signal_value(uncertainty_analysis: Optional[dict], name: str) -> Any:
    sig = _find_signal(uncertainty_analysis, name)
    if sig is None:
        return None
    return sig.get("value")


def _signal_evidence(uncertainty_analysis: Optional[dict], name: str) -> Optional[dict]:
    sig = _find_signal(uncertainty_analysis, name)
    if sig is None:
        return None
    ev = sig.get("evidence")
    return ev if isinstance(ev, dict) else None


def _signal_11(uncertainty_analysis: Optional[dict]) -> Optional[dict]:
    """B4 signal 11 safety_sensitive_indicators 的 value dict（缺失返回 None）。"""
    v = _signal_value(uncertainty_analysis, "safety_sensitive_indicators")
    return v if isinstance(v, dict) else None


def _citation_field(citation_analysis: Optional[dict], key: str, default: Any = None) -> Any:
    """从 B3 citation_analysis dict 安全取值（容忍 None / error dict）。"""
    if not isinstance(citation_analysis, dict) or "error" in citation_analysis:
        return default
    return citation_analysis.get(key, default)


def _is_vetoable_refusal(method: Optional[str], matched_patterns: Optional[list]) -> bool:
    """判断 detect_refusal 的拒答是否属于「可被决策层取消」的 emergency/short_referral 分支。

    仅匹配 emergency_response / short_referral 两类；matched / mental_health /
    out_of_scope / direct_refusal / restricted_override / safety_blocked / empty_response
    一律不可 veto（保留 LLM 自拒答，避免误伤 Q025/Q026 类 TP）。
    """
    if matched_patterns:
        for p in matched_patterns:
            if not isinstance(p, str):
                continue
            if p.startswith("emergency_response") or p.startswith("short_referral"):
                return True
    if method:
        m = method.lower()
        if "紧急就医响应为主导" in m or "短回答 + 多次就医建议" in m:
            return True
    return False


# ============================================================
# 主决策函数
# ============================================================

def make_reliability_decision(
    *,
    question: str = "",
    safety_check: Any = None,
    uncertainty_analysis: Optional[dict] = None,
    citation_analysis: Optional[dict] = None,
    m8: Optional[dict] = None,
    m9: Optional[dict] = None,
    m10: Optional[dict] = None,
    m12: Optional[dict] = None,
    refusal_result: Optional[dict] = None,
    answer: Optional[str] = None,
) -> ReliabilityDecision:
    """基于综合信号做出最终可靠性决策（纯函数，零副作用）。

    Args:
        question:             用户原始问题（仅用于 evidence/日志，不参与规则计算——
                               信号全部来自已产出的 measurement 层，避免重复计算）。
        safety_check:         SafetyChecker.check() 输出（dataclass 或 dict，B2）。
        uncertainty_analysis: B4 extract_signals().to_dict() 输出（含 11 信号）。
        citation_analysis:    B3 analyze_citations() 的 asdict 输出（含 citation_present 等）。
        m8/m9/m10/m12:        B5 metrics 产物 dict（各含 score；缺失/error 容忍）。
        refusal_result:       runner detect_refusal() 输出 dict（is_refusal/method/matched_patterns）。
        answer:               LLM 回答文本（仅用于 evidence 的记录，不参与规则）。

    Returns:
        ReliabilityDecision（to_dict() 可 JSON 序列化）。
    """
    # ---- 只读提取既有信号 ----
    safety_level = _extract_safety_level(safety_check)

    uncertainty_error = (
        not isinstance(uncertainty_analysis, dict)
        or "error" in uncertainty_analysis
    )
    s11 = None if uncertainty_error else _signal_11(uncertainty_analysis)
    s9_value = None if uncertainty_error else _signal_value(
        uncertainty_analysis, "retrieval_support"
    )
    s9_ev = None if uncertainty_error else _signal_evidence(
        uncertainty_analysis, "retrieval_support"
    )
    m2_score = None if uncertainty_error else _signal_value(
        uncertainty_analysis, "retrieval_chunk_relevance"
    )

    has_diag = bool((s11 or {}).get("has_diagnosis_question", False))
    has_dosage = bool((s11 or {}).get("has_dosage_question", False))
    has_symp = bool((s11 or {}).get("has_personal_symptom", False))
    risk_count = int((s11 or {}).get("risk_count", 0))

    likely_out_of_kb = bool((s9_ev or {}).get("likely_out_of_kb", False)) if s9_ev else False
    entity_yield = int((s9_ev or {}).get("entity_yield", 0)) if s9_ev else 0

    citation_present = bool(_citation_field(citation_analysis, "citation_present", False))
    citation_grounded = bool(_citation_field(citation_analysis, "citation_grounded", False))

    m8_score = _metric_score(m8)
    m8_level = _metric_level(m8)
    m9_score = _metric_score(m9)
    m10_score = _metric_score(m10)
    m12_score = _metric_score(m12)

    # ---- 规则评估（P1→P6，收集全部成立规则，主决策取最高优先级） ----
    # 每条规则: (rule_id, reason_code, decision_str, reason)
    triggered: list[tuple[str, str, str, str]] = []

    # P1 — R1 BLOCKED -> REFUSE
    if safety_level == "blocked":
        triggered.append((
            "R1", "SAFETY_BLOCKED", Decision.REFUSE.value,
            f"SafetyChecker BLOCKED（高危问题）：决策层硬性拒答，不受可靠性分数影响。",
        ))

    # P2 — R2 restricted + (dosage | diagnosis) -> REFUSE
    if safety_level == "restricted" and (has_dosage or has_diag):
        triggered.append((
            "R2", "SAFETY_RESTRICTED_PERSONAL", Decision.REFUSE.value,
            f"SafetyChecker RESTRICTED 且检测到个人风险（dosage={has_dosage}, diagnosis={has_diag}）→ REFUSE。",
        ))

    # P2 — R3 has_diagnosis_question -> REFUSE（不检查 safety level：兜底 SafetyChecker 盲区）
    if has_diag:
        triggered.append((
            "R3", "DIAGNOSIS_INTENT", Decision.REFUSE.value,
            f"问题含诊断意图（has_diagnosis_question=True，safety_level={safety_level}）→ REFUSE。"
            f"检索质量好不等于可以给出确定诊断。",
        ))

    # P2 — R4 has_dosage_question -> REFUSE（不检查 safety level）
    if has_dosage:
        triggered.append((
            "R4", "DOSAGE_REQUEST", Decision.REFUSE.value,
            f"问题含用药剂量请求（has_dosage_question=True，safety_level={safety_level}）→ REFUSE。",
        ))

    # P3 — R5 likely_out_of_kb + 无检索支撑 -> REFUSE
    m2_unusable = m2_score is None or (isinstance(m2_score, (int, float)) and float(m2_score) < 0.15)
    if likely_out_of_kb and (entity_yield == 0 or m2_unusable):
        triggered.append((
            "R5", "OUT_OF_KB", Decision.REFUSE.value,
            f"疑似超出知识库（likely_out_of_kb=True, entity_yield={entity_yield}, M2={m2_score}）→ REFUSE。",
        ))

    # P4 — R6 M12 < 0.40 -> REFUSE
    if m12_score is not None and m12_score < 0.40:
        triggered.append((
            "R6", "RELIABILITY_LOW", Decision.REFUSE.value,
            f"综合可靠性过低（M12={m12_score} < 0.40）→ REFUSE。",
        ))

    # P5 — R7 M12 medium + 引用/不确定性缺陷 -> REVISE；引用可靠 + 低不确定 -> 升级 ALLOW
    if (
        m12_score is not None
        and 0.40 <= m12_score < 0.70
        and m8_score is not None
    ):
        if citation_present and citation_grounded and m8_score < 0.3:
            triggered.append((
                "R7", "RELIABILITY_MEDIUM_ALLOW", Decision.ALLOW.value,
                f"中等可靠性（M12={m12_score}）但引用可靠且不确定性低（M8={m8_score}）→ 升级 ALLOW。",
            ))
        elif (not citation_present) or m8_score >= 0.3:
            triggered.append((
                "R7", "RELIABILITY_MEDIUM_REVISE", Decision.REVISE.value,
                f"中等可靠性（M12={m12_score}）且引用缺失或不确定性偏高"
                f"（citation_present={citation_present}, M8={m8_score}）→ REVISE（附免责/就医建议）。",
            ))

    # P5 — R8 personal_symptom（无诊断意图）-> REVISE
    if has_symp and not has_diag:
        triggered.append((
            "R8", "PERSONAL_SYMPTOM_REVISE", Decision.REVISE.value,
            f"问题含个人症状叙述且无诊断意图（personal_symptom=True, diagnosis=False）→ REVISE。",
        ))

    # P6 — R9 默认 ALLOW（仅当无更高优先级规则成立时作为主决策）
    if not triggered:
        triggered.append((
            "R9", "RELIABLE_ANSWER", Decision.ALLOW.value,
            "未命中任何安全/风险/可靠性规则 → ALLOW。",
        ))

    primary_rule, primary_code, primary_decision, primary_reason = triggered[0]
    triggered_ids = [t[0] for t in triggered]
    confidence = _CONFIDENCE_BY_REASON.get(primary_code, 0.8)

    # ---- veto（仅取消 detect_refusal 的 emergency/short_referral 拒答，且决策层判定可答） ----
    cancel_posthoc_refusal = False
    refusal_is_refusal = (
        isinstance(refusal_result, dict)
        and refusal_result.get("is_refusal") is True
    )
    if refusal_is_refusal and primary_decision in (Decision.ALLOW.value, Decision.REVISE.value):
        method = refusal_result.get("method") if isinstance(refusal_result, dict) else None
        patterns = refusal_result.get("matched_patterns") if isinstance(refusal_result, dict) else None
        if _is_vetoable_refusal(method, patterns):
            cancel_posthoc_refusal = True

    # ---- final refusal ----
    if primary_decision == Decision.REFUSE.value:
        is_final_refusal = True
    elif refusal_is_refusal and not cancel_posthoc_refusal:
        is_final_refusal = True
    else:
        is_final_refusal = False

    # ---- evidence ----
    evidence = {
        "question": (question or "")[:200],
        "answer_present": bool(answer and str(answer).strip()),
        "safety_level": safety_level,
        "signal_11": s11 if s11 is not None else {"unavailable": True},
        "retrieval_support": {
            "value": s9_value,
            "likely_out_of_kb": likely_out_of_kb,
            "entity_yield": entity_yield,
        },
        "citation_present": citation_present,
        "citation_grounded": citation_grounded,
        "m8_score": m8_score,
        "m9_score": m9_score,
        "m10_score": m10_score,
        "m12_score": m12_score,
        "refusal": (
            {
                "is_refusal": True,
                "method": (refusal_result or {}).get("method"),
                "matched_patterns": (refusal_result or {}).get("matched_patterns"),
            }
            if refusal_is_refusal else None
        ),
        "veto_applied": cancel_posthoc_refusal,
    }

    # ---- 降级说明 ----
    notes: list[str] = []
    if uncertainty_error:
        notes.append("uncertainty_analysis 不可用，R3/R4/R8/R5 降级（仅 safety_check/M12 门生效）")
    elif s11 is None:
        notes.append("signal-11 缺失，R3/R4/R8 未触发")
    if safety_check is None:
        notes.append("safety_check 缺失，安全门未生效")
    if m12_score is None:
        notes.append("M12 不可用，R6/R7 未触发")
    if m8_score is None:
        notes.append("M8 不可用，R7 未触发")
    note = "; ".join(notes) if notes else None

    explanation = (
        f"{primary_reason} "
        f"[triggered: {', '.join(triggered_ids)}] "
        f"final_refusal={is_final_refusal}"
    )

    return ReliabilityDecision(
        decision=primary_decision,
        reason_code=primary_code,
        reason=primary_reason,
        triggered_rules=triggered_ids,
        confidence=confidence,
        safety_level=safety_level,
        is_final_refusal=is_final_refusal,
        reliability_score=m12_score,
        uncertainty_level=m8_level,
        citation_quality=m9_score,
        evidence=evidence,
        explanation=explanation,
        disclaimer_needed=primary_decision in (Decision.REFUSE.value, Decision.REVISE.value),
        cancel_posthoc_refusal=cancel_posthoc_refusal,
        note=note,
    )


# ============================================================
# 模块级便捷函数
# ============================================================

def decide(**kwargs) -> ReliabilityDecision:
    """模块级便捷入口，等价于 make_reliability_decision(**kwargs)。"""
    return make_reliability_decision(**kwargs)
