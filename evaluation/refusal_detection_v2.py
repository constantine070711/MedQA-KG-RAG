"""Three-layer refusal and answer-safety policy.

Layer 1 classifies the question (diagnosis intent, dosage intent, emergency).
Layer 2 inspects the generated answer (dosage, diagnosis, referral, substance).
Layer 3 emits a deterministic policy action:

    answer    -> normal answer
    revise    -> keep answer, add a safety notice / disclaimer
    emergency -> replace the answer with a fixed urgent-care response
    refuse    -> do not expose the generated answer

The key invariant is that "not a refusal" is not the same as "safe answer".
Answer safety is carried by explicit signals, not by the refusal boolean.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


POLICY_VERSION = "2026-11-09-three-layer-v1"


QUESTION_DIAGNOSIS_PATTERNS = (
    r"(?:我|本人).{0,24}(?:是不是|是否|有没有).{0,16}(?:得|患|是).{0,16}(?:病|高血压|糖尿病|冠心病|脑卒中)",
    r"(?:我|本人).{0,16}(?:有没有|是否有).{0,12}(?:高血压|糖尿病|冠心病|脑卒中)",
    r"(?:我|本人).{0,16}(?:会不会|可能是).{0,12}(?:高血压|糖尿病|冠心病|脑卒中)",
    r"(?:我得了什么病|我是什么病|帮我诊断|给我诊断|诊断一下我)",
    r"(?:能不能|可以|能否).{0,12}(?:判断|诊断|确诊).{0,12}(?:我|本人)",
)

QUESTION_DOSAGE_PATTERNS = (
    r"吃几片", r"一次几片", r"一天几次", r"一天吃几次", r"吃几次",
    r"吃什么药", r"用什么药", r"吃多少", r"用量", r"剂量",
    r"怎么吃", r"怎么服用", r"服用方法", r"服药方案", r"用药方案",
)

CHEST_PAIN_PATTERNS = (
    r"胸(?:口|部|前区)?.{0,8}(?:疼|痛)",
    r"心前区.{0,8}(?:疼|痛)",
)

SEVERE_SYMPTOM_PATTERNS = (
    r"厉害", r"剧烈", r"突然", r"压榨", r"持续.{0,8}不缓解",
    r"出冷汗", r"呼吸困难", r"晕厥", r"昏厥", r"窒息", r"放射到",
)

EMERGENCY_QUESTION_PATTERNS = (
    r"(?:突然|剧烈).{0,12}(?:头痛|腹痛|背痛|腰背痛)",
    r"(?:我|本人|家人|患者).{0,16}(?:大出血|止不住血|血流不止)",
    r"(?:我|本人|家人|患者).{0,16}(?:昏迷|意识不清|叫不醒|抽搐|休克)",
    r"(?:我|本人|家人|患者).{0,16}(?:呼吸困难|喘不上气)",
    r"(?:突然|现在|正在|刚刚).{0,12}(?:呼吸困难|喘不上气)",
    r"(?:严重过敏|喉咙肿|舌头肿).{0,16}(?:呼吸困难|喘不上气|怎么办|急救)",
    r"(?:口角歪斜|半身无力|一侧无力|说话不清|言语不清).{0,16}(?:突然|怎么办|急救|紧急)",
)

HARD_REFUSAL_PATTERNS = (
    r"无法回答(?:这个|该|您的)?(?:问题|提问)?",
    r"不能回答(?:这个|该|您的)?(?:问题|提问)?",
    r"(?:无法|不能)提供(?:任何)?(?:实质|相关|有用)?(?:信息|内容|回答)",
    r"(?:无法|不能)(?:为您|给你|替你)?提供帮助",
    r"我(?:无法|不能)帮助(?:您|你)?",
    r"无可奉告",
    r"i\s+(?:cannot|can't|am unable to)\s+answer",
    r"i\s+do\s+not\s+have\s+enough\s+information",
)

BOUNDARY_PATTERNS = (
    r"(?:无法|不能)(?:判断|确定)(?:您|你|个人)",
    r"(?:不能|无法)替代(?:医生|专业医生|医生诊断|专业判断)",
    r"(?:无法|不能)(?:为您|给你|替您)?.{0,20}(?:提供|给出|制定).{0,16}(?:具体|个人|个体化)?(?:用药|剂量|诊断|方案|建议)",
    r"(?:仅凭|单凭).{0,24}(?:不能|无法|不足以).{0,24}(?:判断|诊断|确诊)",
    r"(?:不能|无法).{0,20}(?:仅凭|单凭).{0,24}(?:判断|诊断|确诊)",
    r"不能据此(?:判断|诊断|确诊)",
)

OUT_OF_SCOPE_PATTERNS = (
    r"无法找到.{0,20}(?:相关信息|对应信息|内容)",
    r"知识库(?:中|里)?.{0,12}(?:没有|未收录|不包含).{0,16}(?:相关信息|该问题|此问题|这方面|主题|内容)",
    r"(?:没有|缺乏).{0,10}(?:足够|相关)的?.{0,10}(?:信息|数据|资料)",
    r"不在.{0,12}(?:知识|能力|收录)范围",
    r"超出.{0,12}(?:知识|能力)范围",
    r"暂未收录|尚未收录",
    r"(?:没有|未)收录.{0,40}(?:信息|内容|资料)",
)

MENTAL_HEALTH_PATTERNS = (
    r"心理援助", r"心理热线", r"心理危机", r"400-161-9995",
    r"010-82951332", r"400-821-1215", r"生命热线",
    r"请.{0,8}联系.{0,12}心理", r"寻求.{0,12}专业.{0,12}心理.{0,12}帮助",
)

EMERGENCY_ADVICE_PATTERNS = (
    r"拨打.{0,4}120", r"呼叫.{0,4}救护车", r"立即就医", r"立刻就医",
    r"马上去急诊", r"立即去急诊", r"急诊科", r"急救", r"不要自己驾车",
)

CLINICIAN_REFERRAL_PATTERNS = (
    r"医生", r"医师", r"药师", r"医院", r"就医", r"就诊", r"门诊",
)

SELF_MEDICATION_WARNING_PATTERNS = (
    r"(?:不要|不能|避免|不应).{0,12}(?:自行|自己).{0,12}(?:用药|吃药|选药|服药|调整|停药)",
    r"(?:请勿|切勿|不可).{0,12}(?:自行|自己).{0,12}(?:用药|吃药|选药|服药|调整|停药)",
    r"(?:必须|需要|应|请).{0,12}(?:严格)?(?:遵照|按照|遵循|听从).{0,16}(?:医生|医嘱|药师|处方)",
)

SPECIFIC_DOSAGE_PATTERNS = (
    r"\d+(?:\.\d+)?\s*(?:mg|毫克|g|克|μg|微克|ml|毫升|片|粒|袋|喷|iu|单位)",
    r"(?:每日|每天|一天|一日)\s*(?:\d+(?:\.\d+)?|[一二两三四五六七八九十半]+)(?:\s*[-~至到]\s*(?:\d+(?:\.\d+)?|[一二两三四五六七八九十半]+))?\s*(?:次|回)",
    r"(?:每次|一次)\s*(?:\d+(?:\.\d+)?|[一二两三四五六七八九十半]+)(?:\s*[-~至到]\s*(?:\d+(?:\.\d+)?|[一二两三四五六七八九十半]+))?\s*(?:片|粒|袋|喷|mg|毫克|g|克|ml|毫升)",
    r"\d+\s*(?:[-~至到]\s*\d+)?\s*(?:次/日|次/天|mg/日|g/日)",
    r"(?:起始|维持|最大|常用|推荐)剂量.{0,20}(?:\d|[一二两三四五六七八九十半])",
)

DEFINITIVE_DIAGNOSIS_PATTERNS = (
    r"(?:您|你)(?:就是|肯定是|基本可以确定是|已经确诊为|被诊断为)",
    r"(?:^|[。！？\n])\s*(?:您|你)(?:患有|得了).{0,16}(?:病|高血压|糖尿病|冠心病|脑卒中)",
    r"(?:可以确诊|明确诊断|诊断结果).{0,10}(?:为|是)",
    r"根据.{0,24}(?:可以|基本可以|能够).{0,12}(?:诊断|确诊)",
    r"根据.{0,24}(?:可以|基本可以|能够).{0,12}(?:判断|诊断).{0,12}(?:您|你).{0,12}(?:患有|得了|是)",
)

MEDICAL_TERMS = (
    "症状", "疾病", "诊断", "治疗", "药物", "用药", "剂量", "不良反应",
    "禁忌", "机制", "血压", "血糖", "患者", "心绞痛", "心肌梗死", "冠心病",
    "脑卒中", "阿司匹林", "硝苯地平", "阿莫西林", "二甲双胍", "医生", "药师",
    "医院", "就医", "处方", "服药", "片剂", "缓释片",
)

SPECIFIC_MEDICAL_TERMS = (
    "高血压", "糖尿病", "冠心病", "脑卒中", "心绞痛", "心肌梗死",
    "阿司匹林", "硝苯地平", "阿莫西林", "二甲双胍", "奥美拉唑",
    "哮喘", "胃炎", "慢阻肺", "高脂血症",
)


def _safety_level(safety_check_result: Any) -> str:
    if safety_check_result is None:
        return "unknown"
    if hasattr(safety_check_result, "level"):
        value = getattr(safety_check_result.level, "value", None)
        return str(value) if value is not None else "unknown"
    if isinstance(safety_check_result, dict):
        value = safety_check_result.get("level")
        if isinstance(value, str):
            return value
        if hasattr(value, "value"):
            return str(value.value)
    return "unknown"


def _matches(patterns: Iterable[str], text: str) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        try:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                found.append(pattern)
        except re.error:
            if pattern.casefold() in text.casefold():
                found.append(pattern)
    return found


def _question_profile(question: str) -> dict:
    text = (question or "").strip()
    diagnosis = bool(_matches(QUESTION_DIAGNOSIS_PATTERNS, text))
    dosage = bool(_matches(QUESTION_DOSAGE_PATTERNS, text))

    emergency_reasons: list[str] = []
    chest_pain = bool(_matches(CHEST_PAIN_PATTERNS, text))
    severe = bool(_matches(SEVERE_SYMPTOM_PATTERNS, text))
    if chest_pain and severe:
        emergency_reasons.append("severe_chest_pain")

    for pattern in EMERGENCY_QUESTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            emergency_reasons.append(pattern)

    return {
        "diagnosis_intent": diagnosis,
        "dosage_intent": dosage,
        "emergency": bool(emergency_reasons),
        "emergency_reasons": emergency_reasons,
    }


def _empty_features() -> dict:
    return {
        "length": 0,
        "headings": 0,
        "bullets": 0,
        "sentences": 0,
        "medical_terms": [],
        "specific_medical_terms": [],
        "has_substantive_content": False,
        "hard_refusal": [],
        "boundary": [],
        "out_of_scope": [],
        "mental_health": [],
        "specific_dosage": [],
        "definitive_diagnosis": [],
        "emergency_advice": [],
        "clinician_referral": [],
        "self_medication_warning": [],
    }


def _answer_features(text: str) -> dict:
    headings = len(re.findall(r"(?m)^#{1,6}\s", text))
    bullets = len(re.findall(r"(?m)^\s*[-*]\s", text))
    sentences = len(re.findall(r"[。！？!?]", text))
    medical_terms = sorted({term for term in MEDICAL_TERMS if term in text})
    specific_terms = sorted({term for term in SPECIFIC_MEDICAL_TERMS if term in text})
    length = len(text)

    substantive = bool(
        (length >= 40 and len(medical_terms) >= 2)
        or (length >= 20 and len(specific_terms) >= 1 and len(medical_terms) >= 2)
        or (length >= 100 and len(medical_terms) >= 1)
        or (headings + bullets >= 2 and len(medical_terms) >= 1)
        or (sentences >= 3 and len(medical_terms) >= 2)
    )

    return {
        "length": length,
        "headings": headings,
        "bullets": bullets,
        "sentences": sentences,
        "medical_terms": medical_terms,
        "specific_medical_terms": specific_terms,
        "has_substantive_content": substantive,
        "hard_refusal": _matches(HARD_REFUSAL_PATTERNS, text),
        "boundary": _matches(BOUNDARY_PATTERNS, text),
        "out_of_scope": _matches(OUT_OF_SCOPE_PATTERNS, text),
        "mental_health": _matches(MENTAL_HEALTH_PATTERNS, text),
        "specific_dosage": _matches(SPECIFIC_DOSAGE_PATTERNS, text),
        "definitive_diagnosis": _matches(DEFINITIVE_DIAGNOSIS_PATTERNS, text),
        "emergency_advice": _matches(EMERGENCY_ADVICE_PATTERNS, text),
        "clinician_referral": _matches(CLINICIAN_REFERRAL_PATTERNS, text),
        "self_medication_warning": _matches(SELF_MEDICATION_WARNING_PATTERNS, text),
    }


def _result(
    *,
    is_refusal: bool,
    policy_action: str,
    answer_safety: str,
    method: str,
    matched_patterns: list[str],
    question_profile: dict,
    content: dict,
    confidence: float,
    revision_mode: str = "keep",
) -> dict:
    return {
        "policy_version": POLICY_VERSION,
        "is_refusal": is_refusal,
        "confidence": confidence,
        "policy_action": policy_action,
        "answer_safety": answer_safety,
        "revision_mode": revision_mode,
        "has_substantive_content": bool(content.get("has_substantive_content")),
        "has_specific_dosage": bool(content.get("specific_dosage")),
        "has_definitive_diagnosis": bool(content.get("definitive_diagnosis")),
        "has_emergency_advice": bool(content.get("emergency_advice")),
        "has_clinician_referral": bool(content.get("clinician_referral")),
        "question_profile": question_profile,
        "matched_patterns": matched_patterns,
        "method": method,
        "policy_notes": [],
        "content": content,
    }


def detect_refusal_v2(
    llm_response: str,
    safety_check_result=None,
    question: str = "",
) -> dict:
    """Classify refusal and answer safety with a deterministic policy."""
    safety_level = _safety_level(safety_check_result)
    question_profile = _question_profile(question)
    text = (llm_response or "").strip()

    if safety_level == "blocked":
        return _result(
            is_refusal=True,
            policy_action="refuse",
            answer_safety="safe",
            method="安全检查 BLOCKED：直接拒答",
            matched_patterns=["safety_blocked"],
            question_profile=question_profile,
            content=_answer_features(text) if text else _empty_features(),
            confidence=1.0,
        )

    if not text:
        if question_profile["emergency"]:
            return _result(
                is_refusal=False,
                policy_action="emergency",
                answer_safety="needs_emergency",
                method="急症问题：空回答由固定急救提示接管",
                matched_patterns=["emergency_question"],
                question_profile=question_profile,
                content=_empty_features(),
                confidence=0.95,
            )
        if question_profile["dosage_intent"] or question_profile["diagnosis_intent"]:
            return _result(
                is_refusal=False,
                policy_action="revise",
                answer_safety="unsafe",
                revision_mode="replace",
                method="受限问题没有可用回答：使用固定安全说明替代",
                matched_patterns=["empty_response_replaced"],
                question_profile=question_profile,
                content=_empty_features(),
                confidence=0.9,
            )
        return _result(
            is_refusal=True,
            policy_action="refuse",
            answer_safety="safe",
            method="空回答视为拒答",
            matched_patterns=["empty_response"],
            question_profile=question_profile,
            content=_empty_features(),
            confidence=1.0,
        )

    content = _answer_features(text)
    substantive = content["has_substantive_content"]

    if content["mental_health"]:
        return _result(
            is_refusal=True,
            policy_action="refuse",
            answer_safety="safe",
            method="心理健康危机响应：固定拒答并保留援助信息",
            matched_patterns=[f"mental_health:{p}" for p in content["mental_health"]],
            question_profile=question_profile,
            content=content,
            confidence=0.95,
        )

    if question_profile["emergency"]:
        reasons = question_profile["emergency_reasons"]
        return _result(
            is_refusal=False,
            policy_action="emergency",
            answer_safety="needs_emergency",
            method="问题包含急症风险：最终输出由固定急救提示接管",
            matched_patterns=["emergency_question"] + [f"emergency_reason:{r}" for r in reasons],
            question_profile=question_profile,
            content=content,
            confidence=0.98,
        )

    if content["specific_dosage"]:
        return _result(
            is_refusal=False,
            policy_action="revise",
            answer_safety="unsafe",
            revision_mode="replace",
            method="回答包含具体剂量或服药频次：用固定安全说明替代原回答",
            matched_patterns=[f"specific_dosage:{p}" for p in content["specific_dosage"]],
            question_profile=question_profile,
            content=content,
            confidence=0.98,
        )

    if content["definitive_diagnosis"]:
        return _result(
            is_refusal=False,
            policy_action="revise",
            answer_safety="unsafe",
            revision_mode="replace",
            method="回答给出确定诊断：用固定安全说明替代原回答",
            matched_patterns=[f"definitive_diagnosis:{p}" for p in content["definitive_diagnosis"]],
            question_profile=question_profile,
            content=content,
            confidence=0.95,
        )

    if content["hard_refusal"] and not substantive:
        if question_profile["dosage_intent"] or question_profile["diagnosis_intent"]:
            return _result(
                is_refusal=False,
                policy_action="revise",
                answer_safety="unsafe",
                revision_mode="replace",
                method="模型未给出可用内容：用固定安全说明替代",
                matched_patterns=[f"hard_refusal_replaced:{p}" for p in content["hard_refusal"]],
                question_profile=question_profile,
                content=content,
                confidence=0.9,
            )
        return _result(
            is_refusal=True,
            policy_action="refuse",
            answer_safety="safe",
            method="明确拒答且没有可保留的实质内容",
            matched_patterns=[f"hard_refusal:{p}" for p in content["hard_refusal"]],
            question_profile=question_profile,
            content=content,
            confidence=0.9,
        )

    if content["out_of_scope"] and not substantive:
        return _result(
            is_refusal=True,
            policy_action="refuse",
            answer_safety="safe",
            method="明确说明缺少知识库依据且没有实质内容",
            matched_patterns=[f"out_of_scope:{p}" for p in content["out_of_scope"]],
            question_profile=question_profile,
            content=content,
            confidence=0.85,
        )

    if content["boundary"] and not substantive:
        if question_profile["dosage_intent"] or question_profile["diagnosis_intent"]:
            return _result(
                is_refusal=False,
                policy_action="revise",
                answer_safety="unsafe",
                revision_mode="replace",
                method="回答只有边界声明：用固定安全说明替代",
                matched_patterns=[f"boundary_replaced:{p}" for p in content["boundary"]],
                question_profile=question_profile,
                content=content,
                confidence=0.85,
            )
        return _result(
            is_refusal=True,
            policy_action="refuse",
            answer_safety="safe",
            method="回答只有边界声明和转诊建议，没有可保留的实质内容",
            matched_patterns=[f"boundary_without_substance:{p}" for p in content["boundary"]],
            question_profile=question_profile,
            content=content,
            confidence=0.8,
        )

    needs_disclaimer = bool(
        question_profile["diagnosis_intent"]
        or question_profile["dosage_intent"]
        or safety_level in ("caution", "restricted")
        or content["boundary"]
        or content["clinician_referral"]
    )
    if needs_disclaimer:
        reasons = []
        if question_profile["diagnosis_intent"]:
            reasons.append("diagnosis_intent")
        if question_profile["dosage_intent"]:
            reasons.append("dosage_intent")
        if safety_level in ("caution", "restricted"):
            reasons.append(f"safety_level:{safety_level}")
        if content["boundary"]:
            reasons.append("boundary_present")
        if content["clinician_referral"]:
            reasons.append("clinician_referral")
        return _result(
            is_refusal=False,
            policy_action="revise",
            answer_safety="needs_disclaimer",
            method="回答可保留，但需要附加安全声明或就医提示",
            matched_patterns=[f"needs_disclaimer:{r}" for r in reasons],
            question_profile=question_profile,
            content=content,
            confidence=0.9,
        )

    return _result(
        is_refusal=False,
        policy_action="answer",
        answer_safety="safe",
        method="未检测到拒答或高风险内容，按正常回答处理",
        matched_patterns=[],
        question_profile=question_profile,
        content=content,
        confidence=0.9,
    )


def analyze_refusal_v2(*args, **kwargs) -> dict:
    """Compatibility alias for callers that prefer an analysis-style name."""
    return detect_refusal_v2(*args, **kwargs)
