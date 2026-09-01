#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 7 G1 — MedQAAgent：LangGraph Agent 工作流。

将 SafetyChecker / LightRAG 检索 / Neo4jGraphRetriever / F5 evidence_fusion /
5C ReliabilityDecision / LLM generation 编排为明确的 StateGraph
（设计见 docs/phase7_design_audit.md）。

图结构（7 节点）:
    START → safety_check ──conditional── BLOCKED → decide
                                      └ 正常 → neo4j_retrieve → build_prompt
                                        → generate → analyze → decide → finalize → END

核心约束:
    - 非 BLOCKED 恰 1 次 LLM generation（generate 节点 query_with_context）
    - BLOCKED 0 次 LLM generation / 0 次 Neo4j 检索
    - 不新增 intent / query-rewrite / reflection / retry / judge LLM；不自动重检索
    - ReliabilityDecision 只复用 evaluation/reliability_decision.py，不重写 R1-R9
    - F6 cache 风险：generate 节点在 system_prompt 非 None 时临时禁用 generation 缓存
    - 不修改 rag_engine.py / 任何既有文件

依赖注入：
    MedQAAgent(engine=..., retriever=..., safety_checker=..., name_map=..., config=...)
    便于单测注入 mock（零真实 LLM / 零真实 DB）。
"""

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langgraph.graph import StateGraph, START, END
from langgraph.types import RunnableConfig

from src.medical_core.safety import SafetyChecker, MEDICAL_DISCLAIMER
from src.evidence_fusion import (
    FusionConfig,
    NEO4J_UNAVAILABLE_KEY,
    build_fusion_prompt,
    extract_curated_mentions,
    fuse_evidence,
)
from scripts.run_graph_query import (
    _base_safety_prompt,
    _retrieve_neo4j,
    _safety_dict,
)
from src.rag_engine import RAGEngine
from src.neo4j_graph import Neo4jGraphRetriever
from evaluation.metrics import (
    load_entity_name_map,
    uncertainty_score_metric,
    citation_quality,
    safety_signal_presence,
    combined_reliability,
)
from evaluation.runner import detect_refusal
from evaluation.citation_verifier import analyze_citations
from evaluation.uncertainty_signals import extract_signals
from evaluation.reliability_decision import make_reliability_decision


# ============================================================
# State
# ============================================================

class AgentState(TypedDict, total=False):
    """Agent 运行时状态（LangGraph StateGraph schema）。

    全部字段序列化友好（safety_result 例外：保存 SafetyCheckResult 对象供
    detect_refusal / safety_signal_presence / finalize 消费，与 F5 一致）。
    """
    # 输入
    question: str
    mode: str

    # safety_check 节点
    safety_result: Any            # SafetyCheckResult（对象，功能消费）
    safety_check: dict            # 序列化 audit（level/reason/disclaimer_needed/is_restricted）
    safety_level: str             # "safe"/"caution"/"restricted"/"blocked"
    base_safety_prompt: Optional[str]

    # neo4j_retrieve 节点
    mentions: list[str]
    neo4j_evidence: dict
    neo4j_unavailable: bool

    # build_prompt 节点
    system_prompt: Optional[str]

    # generate 节点（唯一一次 LLM generation）
    lightrag_evidence: dict
    llm_response: str
    llm_generation_count: int

    # analyze 节点（确定性信号链，零 LLM）
    refusal: dict
    citation_analysis: dict
    uncertainty_analysis: dict
    metrics: dict
    intent: Optional[str]

    # decide 节点（终端门）
    reliability_decision: dict

    # finalize 节点
    fused_context: dict
    coverage: dict
    final_response: dict

    # 错误 / 覆盖
    errors: list[dict]


# ============================================================
# 确定性辅助（纯函数，可独立测试）
# ============================================================

def route_after_safety(state: AgentState) -> str:
    """条件边路由：BLOCKED → decide；其余 → neo4j_retrieve。"""
    return "blocked" if state.get("safety_level") == "blocked" else "continue"


# ============================================================
# conversation_context（Phase 9 P2 多轮 Memory，仅 LLM prompt 背景）
# ============================================================
# Memory = 对话背景，不是 evidence/citation/retrieval/state 事实。
# 唯一进入位置：build_prompt 节点在 generation 前追加的 system_prompt 段。

_MEMORY_SAFE_BASE = (
    "你是 MedQA 医疗知识辅助助手。请基于以下检索到的医学知识回答用户问题，"
    "不要依据对话背景中的历史信息作出诊断或用药建议。\n\n"
    "检索到的医学知识：\n{context_data}"
)
_CONVERSATION_CONTEXT_PREFIX = (
    "\n\n<conversation_context>\n"
    "以下内容仅用于理解用户当前问题与前文指代。它不是当前轮医疗证据，"
    "不得替代当前轮检索结果，不得据此作出诊断或用药建议。\n"
)
_CONVERSATION_CONTEXT_SUFFIX = "\n</conversation_context>"


def _append_conversation_context(
    base_prompt: Optional[str], conversation_context: Optional[str]
) -> Optional[str]:
    """把对话背景（Memory）追加到 system_prompt 末尾（纯函数，可单测）。

    - conversation_context 为空 → 原样返回 base_prompt（行为不变）。
    - base_prompt 非空（CAUTION/RESTRICTED 安全指令或 Neo4j 证据段）→ 追加记忆段。
    - base_prompt 为空（SAFE 且无证据，原本为 None=LightRAG 默认）→ 用
      _MEMORY_SAFE_BASE 打底并保留 {context_data} 占位符，保证当前轮检索
      （LightRAG 向量块）仍进入 generation prompt；记忆段仅作背景。
    - 只用字符串拼接，不调用 .format 处理用户内容（避免花括号注入）。
    """
    ctx = str(conversation_context or "").strip()
    if not ctx:
        return base_prompt
    section = _CONVERSATION_CONTEXT_PREFIX + ctx + _CONVERSATION_CONTEXT_SUFFIX
    if base_prompt:
        return base_prompt + section
    return _MEMORY_SAFE_BASE + section


_REFUSAL_TEXT: dict[str, str] = {
    "SAFETY_BLOCKED": "检测到该问题涉及安全边界，本系统无法提供进一步回答。"
                      "如有紧急情况，请立即拨打急救电话（中国：120）。",
    "SAFETY_RESTRICTED_PERSONAL": "该问题涉及个人诊疗建议，本系统无法提供具体的诊断或用药方案。"
                                  "建议您前往正规医院就诊，由专业医生评估。",
    "DIAGNOSIS_INTENT": "本系统无法根据您的描述进行疾病诊断。"
                        "建议您尽快前往医院就诊，由专业医生结合检查结果判断。",
    "DOSAGE_REQUEST": "本系统无法提供具体用药剂量建议。"
                      "请务必遵医嘱或按药品说明书使用；如有疑问请咨询医生或药师。",
    "OUT_OF_KB": "该问题超出了本系统的医学知识范围，无法给出可靠回答。"
                 "建议您咨询专业医生或查阅权威医学资料。",
    "RELIABILITY_LOW": "由于缺少可靠的医学依据，本系统无法就该问题给出确定回答。"
                       "建议您前往正规医疗机构咨询。",
}
_DEFAULT_REFUSAL = "本系统无法就该问题提供实质性医学内容。建议您咨询专业医生。"


def _refusal_answer(decision: Optional[dict]) -> str:
    """展示层：按 reason_code 生成拒答文案（+ 免责声明）。仅格式化，不改决策。"""
    rc = (decision or {}).get("reason_code")
    msg = _REFUSAL_TEXT.get(rc, _DEFAULT_REFUSAL)
    return f"{msg}\n\n---\n{MEDICAL_DISCLAIMER}"


def _derive_intent(state: AgentState, uncertainty: Optional[dict]) -> str:
    """确定性意图标签（零 LLM，仅审计，不参与路由）。

    依据：safety_level + B4 signal-11（diagnosis/dosage/symptom）+ curated mentions。
    """
    level = state.get("safety_level")
    s11: Optional[dict] = None
    if isinstance(uncertainty, dict) and "error" not in uncertainty:
        for s in uncertainty.get("signals", []):
            if isinstance(s, dict) and s.get("name") == "safety_sensitive_indicators":
                if isinstance(s.get("value"), dict):
                    s11 = s["value"]
                break
    has_diag = bool((s11 or {}).get("has_diagnosis_question", False))
    has_dosage = bool((s11 or {}).get("has_dosage_question", False))
    has_symp = bool((s11 or {}).get("has_personal_symptom", False))
    mentions = state.get("mentions") or []

    if level in ("restricted", "blocked") or has_diag or has_dosage:
        return "personal_health"
    if has_symp:
        return "personal_symptom"
    if mentions:
        return "medical_knowledge"
    return "general"


# ============================================================
# MedQAAgent
# ============================================================

class MedQAAgent:
    """LangGraph Agent（7 节点 StateGraph）。

    Attributes:
        engine:       RAGEngine 或 mock（query_with_context）
        retriever:    Neo4jGraphRetriever 或 mock
        safety_checker: SafetyChecker 或 mock
        name_map:     D4 load_entity_name_map() 输出（默认自动加载）
        config:       FusionConfig
        mode:         LightRAG 检索模式（默认 "mix"）
        disable_generation_cache: 生成期缓存开关（默认 True，见 cache 风险设计）
    """

    def __init__(
        self,
        engine: Optional[RAGEngine] = None,
        retriever: Optional[Neo4jGraphRetriever] = None,
        safety_checker: Optional[SafetyChecker] = None,
        name_map: Optional[dict] = None,
        config: Optional[FusionConfig] = None,
        mode: str = "mix",
        disable_generation_cache: bool = True,
    ):
        self._engine = engine
        self._retriever = retriever
        self._safety_checker = safety_checker or SafetyChecker()
        self._name_map = name_map
        self._config = config or FusionConfig()
        self.mode = mode
        self.disable_generation_cache = disable_generation_cache
        self._compiled_graph = None

    # ----------------------------------------------------------
    # 基础设施
    # ----------------------------------------------------------

    def _get_name_map(self) -> dict:
        if self._name_map is None:
            self._name_map = load_entity_name_map()
        return self._name_map

    async def _get_engine(self):
        if self._engine is None:
            self._engine = RAGEngine()
            await self._engine.initialize()
        return self._engine

    def _get_retriever(self):
        if self._retriever is None:
            self._retriever = Neo4jGraphRetriever()
        return self._retriever

    def _set_llm_cache(self, enabled: bool) -> None:
        """运行时切换 LightRAG generation 缓存（不改源码，F6 已验证的机制）。

        背景：LightRAG 查询缓存键不含 system_prompt，若沿用缓存，
        动态 system_prompt（安全指令 / Neo4j 证据）的 generation 可能命中旧回答。
        """
        rag = getattr(self._engine, "rag", None)
        kv = getattr(rag, "llm_response_cache", None) if rag is not None else None
        gc = getattr(kv, "global_config", None) if kv is not None else None
        if gc is not None:
            gc["enable_llm_cache"] = bool(enabled)

    # ----------------------------------------------------------
    # 节点 1: safety_check
    # ----------------------------------------------------------

    async def _safety_check(self, state: AgentState) -> dict:
        safety = self._safety_checker.check(state["question"])
        return {
            "safety_result": safety,
            "safety_check": _safety_dict(safety),
            "safety_level": getattr(safety.level, "value", str(safety.level)),
            "base_safety_prompt": _base_safety_prompt(safety),
        }

    # ----------------------------------------------------------
    # 节点 2: neo4j_retrieve（复用 F5 逻辑，零 LLM）
    # ----------------------------------------------------------

    async def _neo4j_retrieve(self, state: AgentState) -> dict:
        mentions = extract_curated_mentions(state["question"], self._get_name_map())
        neo4j_result = _retrieve_neo4j(mentions, retriever=self._get_retriever())
        return {
            "mentions": mentions,
            "neo4j_evidence": neo4j_result,
            "neo4j_unavailable": bool(neo4j_result.get(NEO4J_UNAVAILABLE_KEY)),
        }

    # ----------------------------------------------------------
    # 节点 3: build_prompt（复用 F5 build_fusion_prompt，纯函数）
    # ----------------------------------------------------------

    async def _build_prompt(
        self,
        state: AgentState,
        config: Optional[RunnableConfig] = None,
    ) -> dict:
        system_prompt = build_fusion_prompt(
            state.get("neo4j_evidence"),
            base_prompt=state.get("base_safety_prompt"),
            config=self._config,
        )
        # Phase 9 P2: conversation_context（Memory）仅作为 LLM prompt 背景追加。
        # 经 invoke config 传入，不进 AgentState、不进检索、不进证据/决策。
        ctx = None
        if config:
            ctx = (config.get("configurable") or {}).get("conversation_context")
        if ctx:
            system_prompt = _append_conversation_context(system_prompt, ctx)
        return {"system_prompt": system_prompt}

    # ----------------------------------------------------------
    # 节点 4: generate（唯一一次 LLM generation + 缓存开关）
    # ----------------------------------------------------------

    async def _generate(self, state: AgentState) -> dict:
        # 防御：BLOCKED 不应进入本节点（图已路由到 decide）；直接短路保证零 LLM
        if state.get("safety_level") == "blocked":
            return {"lightrag_evidence": {}, "llm_response": "", "llm_generation_count": 0}

        engine = await self._get_engine()
        system_prompt = state.get("system_prompt")

        # F6 cache 风险：system_prompt 非 None（携带安全指令/证据）→ 临时禁用 generation 缓存
        cache_off = bool(self.disable_generation_cache and system_prompt is not None)
        if cache_off:
            self._set_llm_cache(False)
        try:
            qresult = await engine.query_with_context(
                state["question"],
                mode=state.get("mode", self.mode),
                system_prompt=system_prompt,
            )
        finally:
            if cache_off:
                self._set_llm_cache(True)

        return {
            "lightrag_evidence": (qresult or {}).get("data") or {},
            "llm_response": (qresult or {}).get("llm_response") or "",
            "llm_generation_count": 1,
        }

    # ----------------------------------------------------------
    # 节点 5: analyze（确定性信号链，零 LLM，E1-E4 等价）
    # ----------------------------------------------------------

    async def _analyze(self, state: AgentState) -> dict:
        question = state["question"]
        lr = state.get("lightrag_evidence") or {}
        answer = state.get("llm_response") or ""
        safety = state.get("safety_result")
        entities = lr.get("entities") or []
        chunks = lr.get("chunks") or []
        rels = lr.get("relationships") or []
        nm = self._get_name_map()

        # E1: detect_refusal（复用 runner）
        refusal = detect_refusal(answer, safety_check_result=safety)

        # E2: citation（复用 B3）
        try:
            citation = asdict(analyze_citations(
                answer=answer,
                retrieved_entities=entities,
                retrieved_chunks=chunks,
                entity_name_map=nm,
            ))
        except Exception as exc:
            citation = {"error": type(exc).__name__, "message": str(exc)[:200]}

        # E3: uncertainty signals（复用 B4，metrics=None 运行时无 GT）
        try:
            uncertainty = extract_signals(
                question=question,
                retrieved_entities=entities,
                retrieved_chunks=chunks,
                retrieved_relationships=rels,
                safety_check_result=safety,
                refusal_result=refusal,
                citation_analysis=citation,
                metrics=None,
                entity_name_map=nm,
                answer=answer,
            ).to_dict()
        except Exception as exc:
            uncertainty = {"error": type(exc).__name__, "message": str(exc)[:200]}

        # E4: M8-M12（复用 B5）
        m8 = uncertainty_score_metric(uncertainty)
        m9 = citation_quality(citation)
        m10 = safety_signal_presence(safety, uncertainty)
        m12 = combined_reliability(m8, m9, m10)
        metrics = {"M8": m8, "M9": m9, "M10": m10, "M12": m12}

        return {
            "refusal": refusal,
            "citation_analysis": citation,
            "uncertainty_analysis": uncertainty,
            "metrics": metrics,
            "intent": _derive_intent(state, uncertainty),
        }

    # ----------------------------------------------------------
    # 节点 6: decide（终端门，复用 5C，不重写规则）
    # ----------------------------------------------------------

    async def _decide(self, state: AgentState) -> dict:
        metrics = state.get("metrics") or {}
        decision = make_reliability_decision(
            question=state.get("question", ""),
            safety_check=state.get("safety_result"),
            uncertainty_analysis=state.get("uncertainty_analysis"),
            citation_analysis=state.get("citation_analysis"),
            m8=metrics.get("M8"),
            m9=metrics.get("M9"),
            m10=metrics.get("M10"),
            m12=metrics.get("M12"),
            refusal_result=state.get("refusal"),
            answer=state.get("llm_response"),
        )
        return {"reliability_decision": decision.to_dict()}

    # ----------------------------------------------------------
    # 节点 7: finalize（fuse_evidence + 展示层输出映射）
    # ----------------------------------------------------------

    async def _finalize(self, state: AgentState) -> dict:
        question = state.get("question", "")
        decision = state.get("reliability_decision") or {}
        safety = state.get("safety_result")
        answer = state.get("llm_response") or ""
        level = state.get("safety_level")
        errors = state.get("errors") or []

        # 完整融合上下文（审计 / provenance / coverage）——复用 F5 fuse_evidence
        try:
            fused = fuse_evidence(
                lightrag_data=state.get("lightrag_evidence"),
                neo4j_data=state.get("neo4j_evidence"),
                entity_name_map=self._get_name_map(),
                config=self._config,
            )
        except Exception as exc:
            fused = {
                "status": "error",
                "error": type(exc).__name__,
                "message": str(exc)[:200],
            }

        # ---- 展示层映射（不改决策语义）----
        d = decision.get("decision")
        is_refusal = bool(decision.get("is_final_refusal"))

        if level == "blocked":
            rejection = getattr(safety, "rejection_message", "") or ""
            final_answer = rejection or _refusal_answer(decision)
        elif is_refusal or d == "refuse":
            final_answer = _refusal_answer(decision)
        elif d == "revise":
            final_answer = f"{answer}\n\n---\n{MEDICAL_DISCLAIMER}" if answer else MEDICAL_DISCLAIMER
        else:
            final_answer = answer

        coverage = {
            "neo4j_has_evidence": bool(
                (fused.get("coverage") or {}).get("neo4j_has_evidence", False)
            ),
            "notes": (fused.get("coverage") or {}).get("notes", []),
            "neo4j_unavailable": state.get("neo4j_unavailable", False),
        }

        final_response = {
            "question": question,
            "answer": final_answer,
            "safety_level": level,
            "intent": state.get("intent"),
            "mentions": state.get("mentions", []),
            "neo4j_unavailable": state.get("neo4j_unavailable", False),
            "llm_generation_count": state.get("llm_generation_count", 0),
            "reliability_decision": decision,
            "citations": state.get("citation_analysis"),
            "fused_context": fused,
            "coverage": coverage,
            "errors": errors,
        }

        return {
            "final_response": final_response,
            "fused_context": fused,
            "coverage": coverage,
        }

    # ----------------------------------------------------------
    # 图构建 + 执行
    # ----------------------------------------------------------

    def build_graph(self) -> StateGraph:
        """构建 StateGraph（7 节点 + 1 条件边）。"""
        b = StateGraph(AgentState)

        b.add_node("safety_check", self._safety_check)
        b.add_node("neo4j_retrieve", self._neo4j_retrieve)
        b.add_node("build_prompt", self._build_prompt)
        b.add_node("generate", self._generate)
        b.add_node("analyze", self._analyze)
        b.add_node("decide", self._decide)
        b.add_node("finalize", self._finalize)

        b.add_edge(START, "safety_check")
        b.add_conditional_edges(
            "safety_check",
            route_after_safety,
            {"blocked": "decide", "continue": "neo4j_retrieve"},
        )
        b.add_edge("neo4j_retrieve", "build_prompt")
        b.add_edge("build_prompt", "generate")
        b.add_edge("generate", "analyze")
        b.add_edge("analyze", "decide")
        b.add_edge("decide", "finalize")
        b.add_edge("finalize", END)

        return b

    def _compiled(self):
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph().compile()
        return self._compiled_graph

    def _init_state(self, question: str) -> dict:
        return {
            "question": question,
            "mode": self.mode,
            "llm_generation_count": 0,
            "errors": [],
        }

    async def run_state(
        self, question: str, conversation_context: Optional[str] = None
    ) -> dict:
        """执行 Agent，返回完整最终 State（测试 / 调试用）。

        conversation_context: 多轮对话背景（Memory，仅追加到 generation prompt）。
            经 invoke config 传入，不进 AgentState；None/空 → 行为与单轮完全一致。
        """
        input_state = self._init_state(question)
        config = None
        if conversation_context:
            config = {"configurable": {"conversation_context": conversation_context}}
        return await self._compiled().ainvoke(input_state, config=config)

    async def run(
        self, question: str, conversation_context: Optional[str] = None
    ) -> dict:
        """执行 Agent，返回对外输出信封 final_response（或错误信封）。

        conversation_context: 多轮对话背景（Memory），仅作 generation prompt 背景，
            不影响 SafetyChecker/检索/证据/决策；None/空 → 与单轮行为一致。
        """
        try:
            state = await self.run_state(question, conversation_context=conversation_context)
            fr = state.get("final_response")
            if fr is None:
                return {
                    "question": question,
                    "answer": "",
                    "status": "error",
                    "errors": [{
                        "type": "MissingFinalResponse",
                        "message": "finalize 未产出 final_response",
                    }],
                }
            return fr
        except Exception as exc:
            return {
                "question": question,
                "answer": "",
                "status": "error",
                "errors": [{
                    "type": type(exc).__name__,
                    "message": str(exc)[:500],
                }],
            }
