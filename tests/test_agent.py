# -*- coding: utf-8 -*-
"""Phase 7 G1 — MedQAAgent 单元测试（stdlib unittest，mock 引擎/检索器，零真实 LLM / 零真实 DB）。

覆盖（对齐 G1 需求清单 16 项 + 设计审计 §10）:
    1.  State 初始化
    2.  SAFE workflow
    3.  CAUTION workflow
    4.  RESTRICTED workflow
    5.  BLOCKED 直接跳过 Neo4j / LLM
    6.  Neo4j 正常
    7.  Neo4j 不可用降级
    8.  Neo4j 无 evidence
    9.  evidence fusion 被调用
    10. ReliabilityDecision 被调用
    11. generate 最多一次
    12. analyze 不调用 LLM
    13. 不发生自动 retry
    14. cache 风险处理
    15. final response 结构完整
    16. graph 编译成功

运行:
    python -m unittest tests.test_agent -v
"""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent import MedQAAgent, AgentState, route_after_safety, _derive_intent
from src.medical_core.safety import SafetyChecker, SYSTEM_PROMPT_SAFE


# ============================================================
# Mock 组件
# ============================================================

LR_DATA = {
    "entities": [
        {"entity_name": "二甲双胍", "entity_type": "concept",
         "description": "二甲双胍 desc", "source_id": "c0", "file_path": "unknown"},
        {"entity_name": "2型糖尿病", "entity_type": "concept",
         "description": "T2DM desc", "source_id": "c1", "file_path": "unknown"},
    ],
    "relationships": [
        {"src_id": "二甲双胍", "tgt_id": "2型糖尿病", "description": "treats",
         "keywords": "treats", "weight": 1.0, "source_id": "r0", "file_path": "unknown"},
    ],
    "chunks": [
        {"chunk_id": "chunk-000", "content": "二甲双胍是2型糖尿病的一线用药",
         "file_path": "unknown"},
    ],
}


class FakeEngine:
    """mock RAGEngine：记录 query_with_context 调用次数 / system_prompt / 缓存状态。"""

    def __init__(self, llm_response="二甲双胍是治疗2型糖尿病的一线药物，具体用药请遵医嘱。",
                 lr_data=None):
        self.query_calls = 0
        self.last_system_prompt = None
        self.last_question = None
        self.cache_enabled_during_call = None
        self._llm_response = llm_response
        self._lr_data = lr_data if lr_data is not None else LR_DATA
        self.rag = SimpleNamespace(
            llm_response_cache=SimpleNamespace(global_config={"enable_llm_cache": True})
        )

    async def query_with_context(self, question, mode="mix", system_prompt=None,
                                 top_k=60, chunk_top_k=20):
        self.query_calls += 1
        self.last_question = question
        self.last_system_prompt = system_prompt
        self.cache_enabled_during_call = bool(
            self.rag.llm_response_cache.global_config.get("enable_llm_cache")
        )
        return {
            "status": "success",
            "data": self._lr_data,
            "metadata": {},
            "llm_response": self._llm_response,
        }


class FakeRetriever:
    """mock Neo4jGraphRetriever。mode ∈ {normal, empty, error}。"""

    def __init__(self, mode="normal"):
        self.calls = 0
        self.mode = mode

    def get_entity(self, name):
        self.calls += 1
        if self.mode == "error":
            raise RuntimeError("Neo4j 连接失败")
        if self.mode == "empty":
            return {"query": "", "count": 0, "nodes": [], "edges": [], "triples": [],
                    "resolution": {}}
        return {"query": "", "count": 1,
                "nodes": [{"id": name, "name": name, "labels": ["Drug"], "properties": {}}],
                "edges": [], "triples": [], "resolution": {}}

    def get_neighbors(self, name, relation_type=None, direction="both", limit=20):
        self.calls += 1
        if self.mode == "error":
            raise RuntimeError("Neo4j 连接失败")
        if self.mode == "empty":
            return {"query": "", "count": 0, "nodes": [], "edges": [], "triples": [],
                    "resolution": {}}
        return {"query": "", "count": 1,
                "nodes": [{"id": "2型糖尿病", "name": "2型糖尿病",
                           "labels": ["Disease"], "properties": {}}],
                "edges": [{"source": name, "target": "2型糖尿病",
                           "relation_type": "TREATS", "properties": {}}],
                "triples": [[name, "TREATS", "2型糖尿病"]],
                "resolution": {}}


def make_agent(engine=None, retriever=None, **kwargs) -> MedQAAgent:
    return MedQAAgent(
        engine=engine if engine is not None else FakeEngine(),
        retriever=retriever if retriever is not None else FakeRetriever(),
        safety_checker=SafetyChecker(),
        **kwargs,
    )


class SpySafetyChecker(SafetyChecker):
    """记录 check() 的每次输入，验证 memory 未进入 SafetyChecker。"""

    def __init__(self):
        super().__init__()
        self.checked_inputs: list[str] = []

    def check(self, user_input):
        self.checked_inputs.append(str(user_input))
        return super().check(user_input)


def run_state(agent: MedQAAgent, question: str, conversation_context=None) -> dict:
    return asyncio.run(agent.run_state(question, conversation_context=conversation_context))


def run(agent: MedQAAgent, question: str) -> dict:
    return asyncio.run(agent.run(question))


# ============================================================
# 1. State 初始化
# ============================================================

class TestStateInit(unittest.TestCase):
    def test_init_state(self):
        agent = MedQAAgent(engine=FakeEngine(), retriever=FakeRetriever())
        init = agent._init_state("高血压是什么")
        self.assertEqual(init["question"], "高血压是什么")
        self.assertEqual(init["mode"], "mix")
        self.assertEqual(init["llm_generation_count"], 0)
        self.assertEqual(init["errors"], [])

    def test_agent_state_fieldset(self):
        # 设计文档要求的状态字段全部声明
        required = {
            "question", "mode", "safety_result", "safety_check", "safety_level",
            "base_safety_prompt", "mentions", "neo4j_evidence", "neo4j_unavailable",
            "system_prompt", "lightrag_evidence", "llm_response",
            "llm_generation_count", "refusal", "citation_analysis",
            "uncertainty_analysis", "metrics", "intent", "reliability_decision",
            "fused_context", "coverage", "final_response", "errors",
        }
        self.assertTrue(required.issubset(set(AgentState.__annotations__.keys())))

    def test_route_after_safety(self):
        self.assertEqual(route_after_safety({"safety_level": "blocked"}), "blocked")
        for level in ("safe", "caution", "restricted"):
            self.assertEqual(route_after_safety({"safety_level": level}), "continue")
        self.assertEqual(route_after_safety({}), "continue")


# ============================================================
# 2-4. SAFE / CAUTION / RESTRICTED workflow
# ============================================================

class TestSafetyWorkflows(unittest.TestCase):
    def test_safe_workflow(self):
        # "什么是高血压" 无安全模式 → SAFE；无 curated 命中外的额外证据仍走全流程
        agent = make_agent()
        state = run_state(agent, "什么是高血压")
        self.assertEqual(state["safety_level"], "safe")
        self.assertIsNone(state["base_safety_prompt"])
        self.assertEqual(state["llm_generation_count"], 1)
        self.assertIn("reliability_decision", state)
        self.assertEqual(state["final_response"]["question"], "什么是高血压")

    def test_caution_workflow(self):
        agent = make_agent()
        state = run_state(agent, "高血压怎么治")
        self.assertEqual(state["safety_level"], "caution")
        self.assertEqual(state["base_safety_prompt"], SYSTEM_PROMPT_SAFE)
        # SAFE 问题同样只发生一次 generation
        self.assertEqual(state["llm_generation_count"], 1)

    def test_restricted_workflow(self):
        agent = make_agent()
        state = run_state(agent, "二甲双胍吃几片")
        self.assertEqual(state["safety_level"], "restricted")
        self.assertIn("切勿提供具体的用药方案", state["base_safety_prompt"])
        self.assertEqual(state["llm_generation_count"], 1)


# ============================================================
# 5. BLOCKED 直接跳过 Neo4j / LLM
# ============================================================

class TestBlocked(unittest.TestCase):
    def test_blocked_skips_neo4j_and_llm(self):
        eng = FakeEngine()
        ret = FakeRetriever()
        agent = make_agent(engine=eng, retriever=ret)
        state = run_state(agent, "我不想活了")
        self.assertEqual(state["safety_level"], "blocked")
        self.assertEqual(eng.query_calls, 0)          # 0 次 LLM generation
        self.assertEqual(ret.calls, 0)                # 0 次 Neo4j 检索
        self.assertEqual(state["llm_generation_count"], 0)
        d = state["reliability_decision"]
        self.assertEqual(d["decision"], "refuse")
        self.assertEqual(d["reason_code"], "SAFETY_BLOCKED")
        self.assertTrue(d["is_final_refusal"])
        # 最终回答 = rejection_message（含心理援助热线）
        self.assertIn("心理健康", state["final_response"]["answer"])

    def test_blocked_run_envelope(self):
        agent = make_agent()
        fr = run(agent, "我不想活了")
        self.assertEqual(fr["llm_generation_count"], 0)
        self.assertEqual(fr["reliability_decision"]["reason_code"], "SAFETY_BLOCKED")
        self.assertIn("answer", fr)


# ============================================================
# 6-8. Neo4j 正常 / 不可用 / 无 evidence
# ============================================================

class TestNeo4j(unittest.TestCase):
    def test_neo4j_normal(self):
        agent = make_agent(retriever=FakeRetriever(mode="normal"))
        state = run_state(agent, "高血压怎么治")
        self.assertFalse(state["neo4j_unavailable"])
        # 证据段进入 system_prompt
        self.assertIn("TREATS", state["system_prompt"])
        self.assertIn("高血压", state["system_prompt"])
        self.assertTrue(state["fused_context"]["coverage"]["neo4j_has_evidence"])

    def test_neo4j_unavailable_degradation(self):
        agent = make_agent(retriever=FakeRetriever(mode="error"))
        state = run_state(agent, "高血压怎么治")
        self.assertTrue(state["neo4j_unavailable"])
        # 降级：system_prompt = base（SYSTEM_PROMPT_SAFE），无证据段
        self.assertEqual(state["system_prompt"], SYSTEM_PROMPT_SAFE)
        self.assertFalse(state["fused_context"]["coverage"]["neo4j_has_evidence"])
        # 不崩溃，决策仍产出
        self.assertIn("reliability_decision", state)
        self.assertIn("final_response", state)
        self.assertIn("不可用", " ".join(state["fused_context"]["coverage"]["notes"]))

    def test_neo4j_no_evidence(self):
        # 无 curated mention → 无证据，覆盖如实
        agent = make_agent(retriever=FakeRetriever(mode="empty"))
        state = run_state(agent, "今天天气怎么样")
        self.assertEqual(state["mentions"], [])
        self.assertFalse(state["neo4j_unavailable"])
        self.assertIsNone(state["system_prompt"])       # SAFE + 无证据 → base None
        self.assertFalse(state["fused_context"]["coverage"]["neo4j_has_evidence"])


# ============================================================
# 9. evidence fusion 被调用
# ============================================================

class TestEvidenceFusion(unittest.TestCase):
    def test_fusion_called_structure(self):
        agent = make_agent()
        state = run_state(agent, "二甲双胍治什么病")
        fused = state["final_response"]["fused_context"]
        for key in ("status", "evidence_sources", "evidence", "summary", "coverage",
                    "prompt_section"):
            self.assertIn(key, fused)
        self.assertEqual(fused["status"], "ok")
        self.assertGreaterEqual(len(fused["evidence"]), 1)


# ============================================================
# 10. ReliabilityDecision 被调用
# ============================================================

class TestReliabilityDecision(unittest.TestCase):
    def test_decision_called(self):
        agent = make_agent()
        state = run_state(agent, "什么是高血压")
        d = state["reliability_decision"]
        self.assertIn(d["decision"], ("allow", "revise", "refuse"))
        self.assertIsInstance(d["is_final_refusal"], bool)
        self.assertIn("reason_code", d)
        # 决策字段完整
        for key in ("confidence", "safety_level", "explanation", "triggered_rules"):
            self.assertIn(key, d)

    def test_dosage_question_refuses(self):
        # restricted + 剂量信号 → R2（优先级高于 R4）→ REFUSE（确定性，不依赖回答）
        agent = make_agent()
        state = run_state(agent, "二甲双胍吃几片")
        d = state["reliability_decision"]
        self.assertEqual(d["decision"], "refuse")
        self.assertEqual(d["reason_code"], "SAFETY_RESTRICTED_PERSONAL")
        self.assertIn("R4", d["triggered_rules"])   # 剂量规则 R4 被触发
        self.assertTrue(d["is_final_refusal"])


# ============================================================
# 11. generate 最多一次
# ============================================================

class TestGenerateAtMostOnce(unittest.TestCase):
    def test_non_blocked_single_generation(self):
        eng = FakeEngine()
        agent = make_agent(engine=eng)
        state = run_state(agent, "高血压怎么治")
        self.assertEqual(eng.query_calls, 1)
        self.assertEqual(state["llm_generation_count"], 1)

    def test_blocked_zero_generation(self):
        eng = FakeEngine()
        agent = make_agent(engine=eng)
        run_state(agent, "我不想活了")
        self.assertEqual(eng.query_calls, 0)


# ============================================================
# 12. analyze 不调用 LLM
# ============================================================

class TestAnalyzeNoLLM(unittest.TestCase):
    def test_analyze_source_no_llm(self):
        src = Path(_PROJECT_ROOT / "src" / "agent.py").read_text(encoding="utf-8")
        # 提取 _analyze 方法体（到下一个顶层 def 或类方法）
        start = src.find("async def _analyze")
        self.assertGreater(start, -1)
        end = src.find("\n    async def ", start + 1)
        body = src[start: end if end > 0 else len(src)]
        for forbidden in ("query_with_context", "llm_model_func", "openai", "chat(",
                          "ainvoke", "invoke"):
            self.assertNotIn(forbidden, body, f"_analyze 不应调用 {forbidden}")

    def test_analyze_does_not_add_generation(self):
        eng = FakeEngine()
        agent = make_agent(engine=eng)
        run_state(agent, "高血压怎么治")
        self.assertEqual(eng.query_calls, 1)  # analyze 未新增任何 generation


# ============================================================
# 13. 不发生自动 retry
# ============================================================

class TestNoAutoRetry(unittest.TestCase):
    def test_refusal_does_not_trigger_retry(self):
        # 即使决策为 refuse（剂量题），也只生成一次，无重试
        eng = FakeEngine()
        agent = make_agent(engine=eng)
        state = run_state(agent, "二甲双胍吃几片")
        self.assertEqual(state["reliability_decision"]["decision"], "refuse")
        self.assertEqual(eng.query_calls, 1)


# ============================================================
# 14. cache 风险处理
# ============================================================

class TestCacheRisk(unittest.TestCase):
    def test_system_prompt_disables_generation_cache(self):
        # 动态 system_prompt（携带证据/安全指令）→ 生成期缓存被禁用（证据不被旧 cache 吞掉）
        eng = FakeEngine()
        agent = make_agent(engine=eng, retriever=FakeRetriever(mode="normal"))
        state = run_state(agent, "高血压怎么治")   # CAUTION + Neo4j 证据 → system_prompt 非 None
        self.assertIsNotNone(state["system_prompt"])
        self.assertFalse(eng.cache_enabled_during_call)
        # 调用后缓存恢复开启
        self.assertTrue(eng.rag.llm_response_cache.global_config["enable_llm_cache"])

    def test_safe_no_prompt_keeps_cache(self):
        # system_prompt None（纯 SAFE 无证据）→ 缓存保持开启（命中安全）
        eng = FakeEngine()
        agent = make_agent(engine=eng, retriever=FakeRetriever(mode="empty"))
        state = run_state(agent, "今天天气怎么样")
        self.assertIsNone(state["system_prompt"])
        self.assertTrue(eng.cache_enabled_during_call)

    def test_set_llm_cache_restores(self):
        eng = FakeEngine()
        agent = make_agent(engine=eng)
        agent._set_llm_cache(False)
        self.assertFalse(eng.rag.llm_response_cache.global_config["enable_llm_cache"])
        agent._set_llm_cache(True)
        self.assertTrue(eng.rag.llm_response_cache.global_config["enable_llm_cache"])


# ============================================================
# 15. final response 结构完整
# ============================================================

class TestFinalResponse(unittest.TestCase):
    def test_structure(self):
        agent = make_agent()
        fr = run(agent, "高血压怎么治")
        for key in ("question", "answer", "safety_level", "intent", "mentions",
                    "neo4j_unavailable", "llm_generation_count",
                    "reliability_decision", "citations", "fused_context",
                    "coverage", "errors"):
            self.assertIn(key, fr)
        self.assertEqual(fr["llm_generation_count"], 1)
        self.assertIn("safety_level", fr)

    def test_blocked_structure(self):
        agent = make_agent()
        fr = run(agent, "我不想活了")
        self.assertEqual(fr["llm_generation_count"], 0)
        self.assertIn("answer", fr)
        self.assertIn("reliability_decision", fr)


# ============================================================
# 16. graph 编译成功
# ============================================================

class TestGraphCompile(unittest.TestCase):
    def test_compile(self):
        agent = make_agent()
        graph = agent.build_graph().compile()
        self.assertIsNotNone(graph)

    def test_node_names(self):
        agent = make_agent()
        b = agent.build_graph()
        expected = {"safety_check", "neo4j_retrieve", "build_prompt", "generate",
                    "analyze", "decide", "finalize"}
        self.assertTrue(expected.issubset(set(b.nodes.keys())))


# ============================================================
# 辅助: intent 推导（确定性）
# ============================================================

class TestDeriveIntent(unittest.TestCase):
    def test_intent(self):
        self.assertEqual(_derive_intent({"safety_level": "restricted"}, None),
                         "personal_health")
        self.assertEqual(_derive_intent({"safety_level": "safe", "mentions": ["高血压"]},
                                        None), "medical_knowledge")
        self.assertEqual(_derive_intent({"safety_level": "safe", "mentions": []}, None),
                         "general")
        # signal-11 诊断标志
        uncertainty = {
            "signals": [{
                "name": "safety_sensitive_indicators",
                "value": {"has_diagnosis_question": True},
            }],
        }
        self.assertEqual(_derive_intent({"safety_level": "safe"}, uncertainty),
                         "personal_health")


# ============================================================
# Phase 9 P2: conversation_context（多轮 Memory）安全边界
# ============================================================

class TestConversationMemory(unittest.TestCase):
    """Memory 只进 generation prompt 背景，绝不影响当前轮安全/检索/证据/决策。"""

    _MEM = "用户问题：二甲双胍用于治疗什么疾病？\n助手回答：二甲双胍用于治疗2型糖尿病。"

    def _two(self, q, retriever_mode="normal"):
        """同一问题：无 memory vs 有 memory 两次独立运行，返回 (state0, state1)。"""
        s0 = run_state(make_agent(retriever=FakeRetriever(mode=retriever_mode)), q)
        s1 = run_state(make_agent(retriever=FakeRetriever(mode=retriever_mode)), q,
                       conversation_context=self._MEM)
        return s0, s1

    # ---- 结构 / 兼容 ----
    def test_no_agentstate_field(self):
        # Memory 不得成为 AgentState 字段（23 字段结构不变）
        self.assertNotIn("conversation_context", AgentState.__annotations__)

    def test_default_no_memory_compatible(self):
        a = make_agent(retriever=FakeRetriever(mode="empty"))
        s = run_state(a, "今天天气怎么样")
        self.assertIsNone(s["system_prompt"])          # SAFE 无证据 → None（默认不变）
        b = make_agent(retriever=FakeRetriever(mode="empty"))
        s2 = run_state(b, "今天天气怎么样", conversation_context=None)
        self.assertIsNone(s2["system_prompt"])         # 显式 None 与缺省一致

    def test_empty_memory_ignored(self):
        a = make_agent(retriever=FakeRetriever(mode="empty"))
        s = run_state(a, "今天天气怎么样", conversation_context="")
        self.assertIsNone(s["system_prompt"])

    # ---- 注入位置 ----
    def test_memory_appended_to_system_prompt(self):
        a = make_agent()
        s = run_state(a, "高血压怎么治", conversation_context=self._MEM)
        self.assertIsNotNone(s["system_prompt"])
        self.assertIn("<conversation_context>", s["system_prompt"])
        self.assertIn(self._MEM, s["system_prompt"])

    def test_memory_after_evidence_section(self):
        a = make_agent()
        s = run_state(a, "高血压怎么治", conversation_context=self._MEM)
        self.assertIn("TREATS", s["system_prompt"])
        self.assertLess(s["system_prompt"].index("TREATS"),
                        s["system_prompt"].index("<conversation_context>"))

    def test_memory_safe_no_evidence_keeps_context_data(self):
        a = make_agent(retriever=FakeRetriever(mode="empty"))
        s = run_state(a, "今天天气怎么样", conversation_context=self._MEM)
        self.assertIsNotNone(s["system_prompt"])
        self.assertIn("{context_data}", s["system_prompt"])   # 保留当前轮检索占位符
        self.assertIn("<conversation_context>", s["system_prompt"])

    # ---- 安全 / 检索 / 证据 / 决策 不受 Memory 影响 ----
    def test_memory_does_not_change_safety(self):
        for q in ("什么是高血压", "高血压怎么治", "二甲双胍吃几片", "我不想活了"):
            s0, s1 = self._two(q)
            self.assertEqual(s0["safety_level"], s1["safety_level"], q)

    def test_memory_does_not_change_retrieval(self):
        s0, s1 = self._two("高血压怎么治")
        self.assertEqual(s0["mentions"], s1["mentions"])
        self.assertEqual(s0["neo4j_unavailable"], s1["neo4j_unavailable"])
        self.assertEqual(s0["lightrag_evidence"], s1["lightrag_evidence"])
        self.assertEqual(s0["neo4j_evidence"], s1["neo4j_evidence"])

    def test_memory_not_in_fused_context(self):
        s0, s1 = self._two("高血压怎么治")
        self.assertEqual(s0["fused_context"], s1["fused_context"])
        self.assertNotIn(self._MEM, str(s1["fused_context"]))
        self.assertNotIn("conversation_context", str(s1["fused_context"]))

    def test_memory_not_in_citations(self):
        s0, s1 = self._two("高血压怎么治")
        self.assertEqual(s0["citation_analysis"], s1["citation_analysis"])

    def test_memory_not_in_coverage(self):
        s0, s1 = self._two("高血压怎么治")
        self.assertEqual(s0["coverage"], s1["coverage"])

    def test_memory_not_in_reliability(self):
        s0, s1 = self._two("二甲双胍吃几片")   # restricted + 剂量 → refuse
        self.assertEqual(s0["reliability_decision"], s1["reliability_decision"])

    def test_memory_does_not_add_generation(self):
        eng = FakeEngine()
        agent = make_agent(engine=eng)
        run_state(agent, "高血压怎么治", conversation_context=self._MEM)
        self.assertEqual(eng.query_calls, 1)

    # ---- BLOCKED / 诊断拒答 不被 Memory 绕过 ----
    def test_blocked_not_bypassed_by_memory(self):
        eng = FakeEngine()
        agent = make_agent(engine=eng)
        s = run_state(agent, "我不想活了", conversation_context=self._MEM)
        self.assertEqual(s["safety_level"], "blocked")
        self.assertEqual(eng.query_calls, 0)          # 0 次 generation
        self.assertEqual(s["llm_generation_count"], 0)
        self.assertEqual(s["reliability_decision"]["reason_code"], "SAFETY_BLOCKED")

    def test_diagnosis_not_bypassed_by_memory(self):
        agent = make_agent()
        s = run_state(agent, "我最近总是头晕，是不是得了高血压？",
                      conversation_context=self._MEM)
        d = s["reliability_decision"]
        self.assertEqual(d["decision"], "refuse")
        self.assertEqual(d["reason_code"], "DIAGNOSIS_INTENT")
        self.assertTrue(d["is_final_refusal"])


# ============================================================
# Phase 9 P3: 调用边界 spy + 多轮流程验收
# ============================================================

class TestMemoryBoundaries(unittest.TestCase):
    """Phase 9 P3 — spy 验证 memory 不进入 SafetyChecker / retrieval query 等调用边界。"""

    def test_memory_not_in_safety_checker_input(self):
        checker = SpySafetyChecker()
        agent = MedQAAgent(engine=FakeEngine(), retriever=FakeRetriever(),
                           safety_checker=checker)
        run_state(agent, "高血压怎么治", conversation_context="MEMORY_SECRET")
        self.assertEqual(checker.checked_inputs, ["高血压怎么治"])
        self.assertNotIn("MEMORY_SECRET", "".join(checker.checked_inputs))

    def test_memory_not_in_retrieval_query(self):
        eng = FakeEngine()
        agent = make_agent(engine=eng)
        run_state(agent, "高血压怎么治", conversation_context="MEMORY_SECRET")
        self.assertEqual(eng.last_question, "高血压怎么治")   # retrieval 只针对当前问题
        self.assertNotIn("MEMORY_SECRET", str(eng.last_question))

    def test_memory_only_in_generation_system_prompt(self):
        eng = FakeEngine()
        agent = make_agent(engine=eng)
        run_state(agent, "高血压怎么治", conversation_context="MEMORY_SECRET")
        self.assertIn("MEMORY_SECRET", str(eng.last_system_prompt))   # 只进 prompt

    def test_memory_not_in_neo4j_retrieval_input(self):
        ret = FakeRetriever()
        agent = make_agent(retriever=ret)
        run_state(agent, "高血压怎么治", conversation_context="MEMORY_SECRET")
        # mentions 来自 extract_curated_mentions(question)，不含 memory
        self.assertEqual(run_state(make_agent(), "高血压怎么治")["mentions"],
                         run_state(make_agent(), "高血压怎么治", conversation_context="MEMORY_SECRET")["mentions"])


class TestMemoryFlowAgent(unittest.TestCase):
    """Phase 9 P3 — Agent 层多轮流程：BLOCKED/refusal 后恢复、换主题检索当前问题。"""

    _MEM = "用户问题：二甲双胍用于治疗什么疾病？\n助手回答：二甲双胍用于治疗2型糖尿病。"

    def test_blocked_then_recover(self):
        s1 = run_state(make_agent(), "我不想活了", conversation_context=self._MEM)
        self.assertEqual(s1["safety_level"], "blocked")
        self.assertEqual(s1["reliability_decision"]["reason_code"], "SAFETY_BLOCKED")
        self.assertEqual(s1["llm_generation_count"], 0)
        # Q3 正常（无 memory / 带 memory 均不应受 BLOCKED 污染）
        s3 = run_state(make_agent(), "高血压怎么治", conversation_context=self._MEM)
        self.assertNotEqual(s3["safety_level"], "blocked")
        self.assertEqual(s3["llm_generation_count"], 1)

    def test_refusal_then_recover(self):
        s1 = run_state(make_agent(), "我最近总是头晕，是不是得了高血压？",
                       conversation_context=self._MEM)
        self.assertTrue(s1["reliability_decision"]["is_final_refusal"])
        s3 = run_state(make_agent(), "高血压怎么治", conversation_context=self._MEM)
        self.assertFalse(s3["reliability_decision"]["is_final_refusal"])
        self.assertEqual(s3["llm_generation_count"], 1)

    def test_topic_switch_retrieval_current_question(self):
        # 换主题：memory 来自 Q1（二甲双胍），Q2 检索仍只针对当前问题
        agent = make_agent()
        s = run_state(agent, "高血压有哪些常见并发症？",
                      conversation_context=self._MEM)
        self.assertIn("高血压", s["mentions"])
        self.assertNotIn("二甲双胍", s["mentions"])     # 不错误继承 Q1 实体


if __name__ == "__main__":
    unittest.main()
