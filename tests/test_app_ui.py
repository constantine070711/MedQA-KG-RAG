# -*- coding: utf-8 -*-
"""Phase 8 P1 — Streamlit UI 层测试（stdlib unittest，mock agent，零真实 LLM / Neo4j / LightRAG）。

覆盖（对齐 P1 需求清单）:
    1.  Agent 正常回答 → UI 数据映射正确
    2-5. safe / caution / restricted / blocked 状态
    6.  refusal 正确展示
    7-8. citations 存在 / 缺失
    9-10. Neo4j available / unavailable
    11. coverage 正确映射
    12. mentions 正确映射
    13. generation count 正确映射
    14. Agent 返回 error envelope
    15. Agent 抛异常
    16. errors 被 sanitize
    17-19. 不泄漏 API key / password / traceback
    20. 缺失字段不会崩溃
    21. 空 response 不崩溃
    22. Session state 纯函数验证
    23. UI 不改变 final_response 语义

运行:
    python -m unittest tests.test_app_ui -v
"""

import json
import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.ui_helpers import (
    _display_evidence_item,
    _get_event_loop,
    add_message,
    build_conversation_context,
    build_display_sections,
    build_kg_graph_data,
    build_kg_layout,
    format_citations,
    format_coverage,
    format_evidence_overview,
    format_lightrag_evidence,
    format_neo4j_evidence,
    format_reliability_decision,
    format_safety_level,
    format_system_status,
    inject_kg_centering,
    lazy_init,
    mask_sensitive,
    memory_used_hint,
    normalize_mental_health_hotline,
    reset_chat_state,
    resolve_question,
    run_agent_sync,
    sanitize_errors,
)


# ============================================================
# Fixtures
# ============================================================

_DECISION = {
    "decision": "revise",
    "reason_code": "RELIABILITY_MEDIUM_REVISE",
    "reason": "中等可靠性",
    "triggered_rules": ["R7"],
    "confidence": 0.65,
    "safety_level": "caution",
    "is_final_refusal": False,
    "reliability_score": 0.62,
    "uncertainty_level": "medium",
    "citation_quality": 0.3,
    "evidence": {},
    "explanation": "中等可靠性 → REVISE",
    "disclaimer_needed": True,
    "cancel_posthoc_refusal": False,
    "note": None,
}

_CITATIONS = {
    "citation_present": True,
    "citation_format": "references_section",
    "reference_items": [
        {"index": 1, "raw_text": "高血压 (Hypertension) 定义", "entity_name": "高血压",
         "entity_type": None, "source_file": "hypertension.md"},
    ],
    "citation_grounded": True,
    "grounded_entities": ["高血压"],
    "ungrounded_entities": [],
    "grounded_ratio": 1.0,
    "total_references": 1,
    "grounding_method": "entity_name_map",
}

_FUSED = {
    "status": "ok",
    "evidence_sources": {
        "lightrag_vector": 14,
        "lightrag_graph": {"entities": 3, "relationships": 2},
        "neo4j_graph": {"nodes": 5, "edges": 4, "triples": 3},
    },
    "evidence": [],
    "summary": {"total": 10, "deduplicated": 0, "truncated": 0},
    "coverage": {"neo4j_has_evidence": True, "notes": ["命中图谱"]},
    "prompt_section": "x",
}

_COVERAGE = {"neo4j_has_evidence": True, "notes": ["命中图谱"], "neo4j_unavailable": False}


def make_response(**overrides) -> dict:
    base = {
        "question": "高血压怎么治？",
        "answer": "高血压需要综合管理，具体用药请遵医嘱。",
        "safety_level": "caution",
        "intent": "medical_knowledge",
        "mentions": ["高血压"],
        "neo4j_unavailable": False,
        "llm_generation_count": 1,
        "reliability_decision": dict(_DECISION),
        "citations": dict(_CITATIONS),
        "fused_context": dict(_FUSED),
        "coverage": dict(_COVERAGE),
        "errors": [],
    }
    base.update(overrides)
    return base


class FakeAgent:
    """mock agent：async run(question)。可返回自定义 response 或抛异常。"""

    def __init__(self, response=None, raise_error=False):
        self.calls = 0
        self.response = response
        self.raise_error = raise_error

    async def run(self, question):
        self.calls += 1
        if self.raise_error:
            raise RuntimeError("内部错误 boom")
        return self.response if self.response is not None else make_response()


# ============================================================
# 1. Agent 正常回答 → UI 数据映射正确
# ============================================================

class TestAgentMapping(unittest.TestCase):
    def test_run_agent_sync_calls_once_and_maps(self):
        a = FakeAgent()
        fr = run_agent_sync(a, "高血压怎么治？")
        self.assertEqual(a.calls, 1)                       # 恰一次 Agent 调用
        self.assertNotEqual(fr.get("status"), "error")     # 非错误信封

        sections = build_display_sections(fr)
        self.assertEqual(sections["answer"], fr["answer"])
        self.assertEqual(sections["safety"]["level"], "caution")
        self.assertEqual(sections["decision"]["decision"], "revise")
        self.assertEqual(sections["decision"]["reason_code"], "RELIABILITY_MEDIUM_REVISE")
        self.assertTrue(sections["citations"]["present"])
        self.assertTrue(sections["coverage"]["neo4j_has_evidence"])
        self.assertEqual(sections["system"]["llm_generation_count"], 1)
        self.assertFalse(sections["is_error"])
        self.assertEqual(sections["errors"], [])

    def test_ui_does_not_change_semantics(self):
        # UI 输出只是 final_response 的派生展示，不改写 Agent 语义
        fr = make_response()
        sections = build_display_sections(fr)
        self.assertEqual(sections["answer"], fr["answer"])
        self.assertEqual(sections["decision"]["decision"], fr["reliability_decision"]["decision"])
        self.assertEqual(sections["decision"]["is_final_refusal"],
                         fr["reliability_decision"]["is_final_refusal"])


# ============================================================
# 2-5. safety 各状态
# ============================================================

class TestSafetyLevels(unittest.TestCase):
    def test_all_levels(self):
        expect_tone = {"safe": "ok", "caution": "warn", "restricted": "warn", "blocked": "danger"}
        for level, tone in expect_tone.items():
            s = format_safety_level(level)
            self.assertEqual(s["level"], level)
            self.assertEqual(s["tone"], tone)
            self.assertTrue(s["label"])

    def test_labels(self):
        self.assertIn("正常", format_safety_level("safe")["label"])
        self.assertIn("谨慎", format_safety_level("caution")["label"])
        self.assertIn("受限", format_safety_level("restricted")["label"])
        self.assertIn("安全拦截", format_safety_level("blocked")["label"])

    def test_blocked_sections(self):
        sections = build_display_sections(make_response(safety_level="blocked"))
        self.assertEqual(sections["safety"]["level"], "blocked")
        self.assertEqual(sections["safety"]["tone"], "danger")


# ============================================================
# 6. refusal 正确展示
# ============================================================

class TestRefusal(unittest.TestCase):
    def test_refuse_decision(self):
        d = dict(_DECISION, decision="refuse", is_final_refusal=True,
                 reason_code="DIAGNOSIS_INTENT")
        sections = build_display_sections(make_response(reliability_decision=d))
        self.assertTrue(sections["decision"]["is_final_refusal"])
        self.assertEqual(sections["decision"]["decision"], "refuse")
        self.assertEqual(sections["decision"]["reason_code"], "DIAGNOSIS_INTENT")

    def test_final_refusal_preserved_even_if_allow(self):
        # Q026 场景：decision=allow 但 is_final_refusal=True（LLM 自拒答保留）
        d = dict(_DECISION, decision="allow", reason_code="RELIABLE_ANSWER",
                 is_final_refusal=True)
        sections = build_display_sections(make_response(reliability_decision=d))
        self.assertTrue(sections["decision"]["is_final_refusal"])


# ============================================================
# 7-8. citations 存在 / 缺失
# ============================================================

class TestCitations(unittest.TestCase):
    def test_citations_present(self):
        c = format_citations(make_response()["citations"])
        self.assertTrue(c["present"])
        self.assertEqual(len(c["items"]), 1)
        self.assertTrue(c["items"][0]["grounded"])
        self.assertEqual(c["grounded_count"], 1)

    def test_citations_missing(self):
        empty = {"citation_present": False, "reference_items": [],
                 "grounded_entities": [], "ungrounded_entities": [],
                 "grounded_ratio": 0.0, "total_references": 0, "grounding_method": "none"}
        c = format_citations(empty)
        self.assertFalse(c["present"])
        self.assertEqual(c["items"], [])
        self.assertIsNone(c["note"])

    def test_citations_error_dict(self):
        c = format_citations({"error": "X", "message": "boom"})
        self.assertTrue(c["has_error"])
        self.assertFalse(c["present"])


# ============================================================
# 9-10. Neo4j available / unavailable
# ============================================================

class TestNeo4j(unittest.TestCase):
    def test_available(self):
        fr = make_response()
        cov = format_coverage(fr)
        self.assertTrue(cov["neo4j_has_evidence"])
        self.assertFalse(cov["neo4j_unavailable"])
        sys_st = format_system_status(fr)
        self.assertEqual(sys_st["neo4j_status"], "hit")
        self.assertIn("已提供", sys_st["neo4j_label"])

    def test_unavailable(self):
        fr = make_response(
            neo4j_unavailable=True,
            coverage={"neo4j_has_evidence": False, "notes": ["Neo4j 容器不可用"],
                      "neo4j_unavailable": True},
            fused_context=dict(_FUSED, evidence_sources={
                "lightrag_vector": 14,
                "lightrag_graph": {"entities": 3, "relationships": 2},
                "neo4j_graph": {"nodes": 0, "edges": 0, "triples": 0},
            }),
        )
        cov = format_coverage(fr)
        self.assertTrue(cov["neo4j_unavailable"])
        self.assertFalse(cov["neo4j_has_evidence"])
        self.assertEqual(cov["neo4j_triples"], 0)
        sys_st = format_system_status(fr)
        self.assertEqual(sys_st["neo4j_status"], "unavailable")
        self.assertIn("不可用", sys_st["neo4j_label"])


# ============================================================
# 11-13. coverage / mentions / generation count
# ============================================================

class TestCoverageAndStatus(unittest.TestCase):
    def test_coverage_counts(self):
        cov = format_coverage(make_response())
        self.assertEqual(cov["lightrag_vector_chunks"], 14)
        self.assertEqual(cov["lightrag_graph_entities"], 3)
        self.assertEqual(cov["lightrag_graph_relationships"], 2)
        self.assertEqual(cov["neo4j_nodes"], 5)
        self.assertEqual(cov["neo4j_edges"], 4)
        self.assertEqual(cov["neo4j_triples"], 3)

    def test_mentions(self):
        sections = build_display_sections(make_response(mentions=["高血压", "脑卒中"]))
        self.assertEqual(sections["coverage"]["mentions"], ["高血压", "脑卒中"])

    def test_generation_count(self):
        sys_st = format_system_status(make_response(llm_generation_count=1))
        self.assertEqual(sys_st["llm_generation_count"], 1)
        sys_st0 = format_system_status(make_response(llm_generation_count=0))
        self.assertEqual(sys_st0["llm_generation_count"], 0)


# ============================================================
# 14-15. error envelope / 抛异常
# ============================================================

class TestErrors(unittest.TestCase):
    def test_error_envelope_passthrough(self):
        a = FakeAgent(response={"status": "error", "question": "q", "answer": "",
                                "errors": [{"type": "X", "message": "boom"}]})
        fr = run_agent_sync(a, "q")
        self.assertEqual(fr.get("status"), "error")
        sections = build_display_sections(fr)
        self.assertTrue(sections["is_error"])
        self.assertEqual(sections["errors"][0]["message"], "boom")

    def test_agent_raises(self):
        a = FakeAgent(raise_error=True)
        fr = run_agent_sync(a, "q")
        self.assertEqual(fr.get("status"), "error")
        self.assertTrue(fr.get("errors"))
        # 不泄漏原始 traceback
        joined = " ".join(str(e.get("message", "")) for e in fr["errors"])
        self.assertNotIn("Traceback", joined)
        self.assertNotIn("File \"", joined)

    def test_empty_question(self):
        fr = run_agent_sync(FakeAgent(), "   ")
        self.assertEqual(fr.get("status"), "error")

    def test_agent_none(self):
        fr = run_agent_sync(None, "q")
        self.assertEqual(fr.get("status"), "error")


# ============================================================
# 16-19. sanitize / 不泄漏敏感信息
# ============================================================

class TestSanitize(unittest.TestCase):
    def test_sanitize_drops_traceback_field(self):
        cleaned = sanitize_errors([
            {"type": "E", "message": "boom", "traceback": "Traceback ...", "raw": "x"},
        ])
        self.assertEqual(len(cleaned), 1)
        self.assertNotIn("traceback", cleaned[0])
        self.assertNotIn("raw", cleaned[0])

    def test_mask_api_key(self):
        msg = mask_sensitive("auth failed api key=sk-abcdef1234567890")
        self.assertNotIn("sk-abcdef1234567890", msg)

    def test_mask_password_and_token(self):
        self.assertNotIn("secret", mask_sensitive("password=supersecret token=xyz").lower())
        self.assertIn("隐藏", mask_sensitive("password=supersecret token=xyz"))

    def test_mask_uri_credentials(self):
        masked = mask_sensitive("bolt://neo4j:pass123@localhost:7687")
        self.assertNotIn("pass123", masked)
        self.assertIn("***", masked)

    def test_no_leak_in_display(self):
        fr = make_response(errors=[{"type": "E", "message": "key=sk-abc123",
                                    "traceback": "Traceback"}])
        sections = build_display_sections(fr)
        dump = str(sections["debug"])
        self.assertNotIn("sk-abc123", dump)
        self.assertNotIn("Traceback", dump)


# ============================================================
# 20-21. 缺失字段 / 空 response 不崩溃
# ============================================================

class TestRobustness(unittest.TestCase):
    def test_empty_dict(self):
        sections = build_display_sections({})
        self.assertEqual(sections["answer"], "")
        self.assertIn("safety", sections)
        self.assertIn("decision", sections)
        self.assertIn("citations", sections)
        self.assertIn("coverage", sections)
        self.assertIn("system", sections)
        self.assertFalse(sections["is_error"])

    def test_none_response(self):
        sections = build_display_sections(None)
        self.assertEqual(sections["answer"], "")
        self.assertEqual(sections["errors"], [])

    def test_missing_fields(self):
        sections = build_display_sections({"answer": "hi"})
        self.assertEqual(sections["answer"], "hi")
        self.assertEqual(sections["safety"]["level"], "unknown")
        self.assertFalse(sections["decision"]["is_final_refusal"])
        self.assertFalse(sections["citations"]["present"])
        self.assertEqual(sections["coverage"]["neo4j_triples"], 0)

    def test_empty_agent_response(self):
        a = FakeAgent(response={})
        fr = run_agent_sync(a, "q")
        sections = build_display_sections(fr)
        self.assertEqual(sections["answer"], "")


# ============================================================
# 22. Session state 纯函数验证
# ============================================================

class TestSessionState(unittest.TestCase):
    def test_add_message(self):
        msgs = add_message([], "user", "q1")
        self.assertEqual(msgs, [{"role": "user", "content": "q1"}])
        msgs = add_message(msgs, "assistant", "a1")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[1]["role"], "assistant")

    def test_add_message_does_not_mutate_input(self):
        original = [{"role": "user", "content": "x"}]
        new = add_message(original, "user", "y")
        self.assertEqual(len(original), 1)   # 原列表不被修改
        self.assertEqual(len(new), 2)

    def test_reset_chat_state(self):
        state = reset_chat_state()
        self.assertEqual(state, {"messages": [], "history": [], "last_response": None})


# ============================================================
# 23. 静态：app 层不 import 禁改核心模块 / 不触 DB·LLM
# ============================================================

class TestNoCoreLeak(unittest.TestCase):
    def test_app_does_not_import_core_modules(self):
        for mod in ("app/streamlit_app.py", "app/ui_helpers.py"):
            src = Path(_PROJECT_ROOT / mod).read_text(encoding="utf-8")
            import_lines = [ln for ln in src.splitlines()
                            if ln.lstrip().startswith(("import ", "from "))]
            forbidden = ("lightrag", "src.rag_engine", "src.neo4j", "src.llm_client",
                         "src.evidence_fusion", "reliability_decision", "src.medical_core")
            for line in import_lines:
                for kw in forbidden:
                    self.assertNotIn(kw, line, f"{mod} 不应 import {kw}: {line}")

    def test_ui_helpers_no_streamlit_import(self):
        src = Path(_PROJECT_ROOT / "app" / "ui_helpers.py").read_text(encoding="utf-8")
        import_lines = [ln for ln in src.splitlines()
                        if ln.lstrip().startswith(("import ", "from "))]
        for line in import_lines:
            self.assertNotIn("streamlit", line)


# ============================================================
# P2: 聊天消息构建 + 历史顺序 + 响应绑定
# ============================================================

def run_turn(agent, msgs, question):
    """模拟一轮问答（纯 helper 层）：user → agent.run → assistant(response)。"""
    msgs = add_message(msgs, "user", question)
    fr = run_agent_sync(agent, question)
    content = (fr.get("answer") or "") if fr.get("status") != "error" else "本次请求暂时无法完成，请稍后重试。"
    msgs = add_message(msgs, "assistant", content, response=fr)
    return msgs, fr


class TestChatMessageBuilding(unittest.TestCase):
    def test_first_turn_user_assistant(self):
        agent = FakeAgent()
        msgs, fr = run_turn(agent, [], "Q1")
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[0]["content"], "Q1")
        self.assertIs(msgs[1]["response"], fr)          # response 绑定本轮
        self.assertEqual(agent.calls, 1)

    def test_second_turn(self):
        agent = FakeAgent()
        msgs, _ = run_turn(agent, [], "Q1")
        msgs, fr2 = run_turn(agent, msgs, "Q2")
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user", "assistant"])
        self.assertEqual(msgs[2]["content"], "Q2")
        self.assertIs(msgs[3]["response"], fr2)
        self.assertEqual(agent.calls, 2)

    def test_response_bound_to_question(self):
        # 每轮 response 与对应 question 绑定：assistant.response 源自该轮 agent.run
        agent = FakeAgent()
        msgs, fr1 = run_turn(agent, [], "Q1")
        msgs, fr2 = run_turn(agent, msgs, "Q2")
        self.assertIsNot(fr1, fr2)
        self.assertIs(msgs[1]["response"], fr1)
        self.assertIs(msgs[3]["response"], fr2)


# ============================================================
# P2: 清空对话（保留 Agent）
# ============================================================

class TestClearConversation(unittest.TestCase):
    def test_reset_clears_chat(self):
        state = reset_chat_state()
        self.assertEqual(state, {"messages": [], "history": [], "last_response": None})

    def test_reset_keeps_agent(self):
        calls = []

        def factory():
            calls.append(1)
            return "AGENT"

        state = {}
        lazy_init(state, "agent", factory)
        fresh = reset_chat_state()
        self.assertNotIn("agent", fresh)               # 清空不涉及 agent
        self.assertIs(lazy_init(state, "agent", factory), "AGENT")  # agent 仍复用
        self.assertEqual(len(calls), 1)


# ============================================================
# P2: 输入解析 / 防重复 / 空输入
# ============================================================

class TestInputParsing(unittest.TestCase):
    def test_resolve_question_priority(self):
        self.assertEqual(resolve_question("你好", None), "你好")
        self.assertEqual(resolve_question("", "示例题"), "示例题")
        self.assertEqual(resolve_question("你好", "示例题"), "你好")   # prompt 优先
        self.assertIsNone(resolve_question(None, None))
        self.assertIsNone(resolve_question("   ", "   "))

    def test_empty_question_no_agent_call(self):
        agent = FakeAgent()
        self.assertIsNone(resolve_question("", None))
        self.assertEqual(agent.calls, 0)               # 空输入不调用 Agent

    def test_normal_question_one_agent_call(self):
        agent = FakeAgent()
        run_agent_sync(agent, "问题")
        self.assertEqual(agent.calls, 1)


# ============================================================
# P2: 错误后继续下一轮
# ============================================================

class TestErrorRecovery(unittest.TestCase):
    def test_error_then_continue(self):
        agent = FakeAgent(raise_error=True)
        fr1 = run_agent_sync(agent, "Q1")
        self.assertEqual(fr1.get("status"), "error")   # 第一轮 error

        agent.raise_error = False                      # 第二轮恢复
        msgs, fr2 = run_turn(agent, [], "Q2")
        self.assertNotEqual(fr2.get("status"), "error")
        self.assertEqual(len(msgs), 2)                 # 历史结构正常
        self.assertEqual(agent.calls, 2)


# ============================================================
# P2: 每轮 response 独立（不污染下一轮）
# ============================================================

class TestPerTurnIsolation(unittest.TestCase):
    def _unavailable_response(self):
        return make_response(
            question="Q1",
            neo4j_unavailable=True,
            coverage={"neo4j_has_evidence": False, "notes": ["Neo4j 容器不可用"],
                      "neo4j_unavailable": True},
            fused_context=dict(_FUSED, evidence_sources={
                "lightrag_vector": 14,
                "lightrag_graph": {"entities": 3, "relationships": 2},
                "neo4j_graph": {"nodes": 0, "edges": 0, "triples": 0},
            }),
            llm_generation_count=1,
        )

    def _hit_response(self):
        return make_response(
            question="Q2",
            neo4j_unavailable=False,
            coverage={"neo4j_has_evidence": True, "notes": ["命中图谱"],
                      "neo4j_unavailable": False},
            llm_generation_count=1,
        )

    def test_neo4j_status_not_polluted(self):
        s1 = build_display_sections(self._unavailable_response())
        s2 = build_display_sections(self._hit_response())
        self.assertTrue(s1["coverage"]["neo4j_unavailable"])
        self.assertFalse(s2["coverage"]["neo4j_unavailable"])
        self.assertEqual(s1["system"]["neo4j_status"], "unavailable")
        self.assertEqual(s2["system"]["neo4j_status"], "hit")

    def test_citations_not_polluted(self):
        s1 = build_display_sections(self._unavailable_response())  # citations present
        no_cit = make_response(citations={"citation_present": False, "reference_items": [],
                                          "grounded_entities": [], "ungrounded_entities": [],
                                          "grounded_ratio": 0.0, "total_references": 0,
                                          "grounding_method": "none"})
        s2 = build_display_sections(no_cit)
        self.assertTrue(s1["citations"]["present"])
        self.assertFalse(s2["citations"]["present"])

    def test_coverage_not_polluted(self):
        s1 = build_display_sections(self._unavailable_response())
        s2 = build_display_sections(self._hit_response())
        self.assertEqual(s1["coverage"]["neo4j_triples"], 0)
        self.assertEqual(s2["coverage"]["neo4j_triples"], 3)

    def test_generation_count_per_response(self):
        blk = make_response(safety_level="blocked", llm_generation_count=0)
        ok = make_response(safety_level="safe", llm_generation_count=1)
        self.assertEqual(build_display_sections(blk)["system"]["llm_generation_count"], 0)
        self.assertEqual(build_display_sections(ok)["system"]["llm_generation_count"], 1)


# ============================================================
# P2: 生命周期（Agent / event loop 复用）
# ============================================================

class TestLifecycle(unittest.TestCase):
    def test_lazy_init_reuses(self):
        calls = []

        def factory():
            calls.append(1)
            return object()

        state = {}
        a1 = lazy_init(state, "agent", factory)
        a2 = lazy_init(state, "agent", factory)
        self.assertIs(a1, a2)
        self.assertEqual(len(calls), 1)                # factory 只调用一次

    def test_event_loop_stable(self):
        ids = {id(_get_event_loop()) for _ in range(3)}
        self.assertEqual(len(ids), 1)                  # 同一持久事件循环

    def test_three_queries_same_lifecycle(self):
        agent = FakeAgent()
        for i in range(3):
            fr = run_agent_sync(agent, f"Q{i}")
            self.assertNotEqual(fr.get("status"), "error")
        self.assertEqual(agent.calls, 3)               # 同一 agent 实例连续 3 次
        # 连续 3 次查询均走同一持久事件循环（无跨循环错误）
        self.assertEqual(id(_get_event_loop()), id(_get_event_loop()))

    def test_messages_three_turns(self):
        agent = FakeAgent()
        msgs, _ = run_turn(agent, [], "Q1")
        msgs, _ = run_turn(agent, msgs, "Q2")
        msgs, _ = run_turn(agent, msgs, "Q3")
        self.assertEqual([m["role"] for m in msgs],
                         ["user", "assistant"] * 3)
        self.assertEqual(agent.calls, 3)


# ============================================================
# P2: 展示 & 泄漏
# ============================================================

class TestChatDisplayAndLeak(unittest.TestCase):
    def test_loading_state_present(self):
        src = Path(_PROJECT_ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("st.spinner", src)
        self.assertIn("正在检索医疗知识并生成回答", src)

    def test_examples_present(self):
        src = Path(_PROJECT_ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("EXAMPLES", src)
        self.assertIn("2型糖尿病的常见症状有哪些？", src)

    def test_no_prompt_or_system_prompt_in_debug(self):
        sections = build_display_sections(make_response())
        dbg = sections["debug"]
        self.assertNotIn("question", dbg)
        self.assertNotIn("system_prompt", dbg)
        self.assertNotIn("prompt", dbg)

    def test_no_leak_after_error_in_chat(self):
        # error 信封：debug 只有清洗后 errors，无 traceback / api / password / prompt
        fr = {"status": "error", "errors": [
            {"type": "E", "message": "boom key=sk-abc123", "traceback": "Traceback ..."}]}
        cleaned = sanitize_errors(fr.get("errors"))
        dump = str(cleaned)
        self.assertNotIn("sk-abc123", dump)
        self.assertNotIn("Traceback", dump)
        self.assertNotIn("password", dump.lower())

    def test_blocked_and_refuse_displayed_from_response(self):
        # UI 只根据 final_response 展示，不自作判断
        blocked = build_display_sections(make_response(safety_level="blocked"))
        refuse = build_display_sections(make_response(reliability_decision=dict(
            _DECISION, decision="refuse", is_final_refusal=True, reason_code="DIAGNOSIS_INTENT")))
        self.assertEqual(blocked["safety"]["level"], "blocked")
        self.assertEqual(refuse["decision"]["decision"], "refuse")
        self.assertTrue(refuse["decision"]["is_final_refusal"])


# ============================================================
# P3: 证据展示 fixtures
# ============================================================

def _ev(kind, name, payload, sources, priority):
    return {"id": f"{kind}::{name}", "kind": kind, "name": name,
            "payload": payload, "sources": sources, "priority": priority}


LR_EVIDENCE = [
    _ev("lightrag_entity", "二甲双胍",
        {"entity_name": "二甲双胍", "entity_type": "drug", "description": "双胍类降糖药",
         "source_id": "c0", "file_path": "metformin.md"}, ["lightrag_graph"], 3),
    _ev("lightrag_relation", "二甲双胍 -treats-> 2型糖尿病",
        {"src_id": "二甲双胍", "tgt_id": "2型糖尿病", "description": "treats", "keywords": "treats",
         "weight": 1.0, "source_id": "r0", "file_path": "x.md"}, ["lightrag_graph"], 2),
    _ev("lightrag_chunk", "chunk-000",
        {"chunk_id": "chunk-000", "content": "二甲双胍是2型糖尿病的一线用药", "file_path": "x.md"},
        ["lightrag_vector"], 4),
]
N4_EVIDENCE = [
    _ev("neo4j_triple", "二甲双胍 -TREATS-> 2型糖尿病",
        {"source": "二甲双胍", "relation_type": "TREATS", "target": "2型糖尿病",
         "canonical_source": "二甲双胍", "canonical_target": "2型糖尿病"},
        ["neo4j_graph"], 0),
    _ev("neo4j_triple", "高脂血症 -HAS_COMPLICATION-> 冠心病",
        {"source": "高脂血症", "relation_type": "HAS_COMPLICATION", "target": "冠心病",
         "canonical_source": "高脂血症", "canonical_target": "冠心病"},
        ["neo4j_graph", "lightrag_graph"], 0),   # 跨源合并 provenance
    _ev("neo4j_node", "二甲双胍",
        {"id": "二甲双胍", "name": "二甲双胍", "entity_type": "drug",
         "labels": ["Drug"], "properties": {}}, ["neo4j_graph"], 1),
]


def _fused_lr():
    return {"status": "ok",
            "evidence_sources": {"lightrag_vector": 1,
                                 "lightrag_graph": {"entities": 1, "relationships": 1},
                                 "neo4j_graph": {"nodes": 0, "edges": 0, "triples": 0}},
            "evidence": LR_EVIDENCE, "summary": {},
            "coverage": {"neo4j_has_evidence": False, "notes": []}, "prompt_section": None}


def _fused_n4():
    return {"status": "ok",
            "evidence_sources": {"lightrag_vector": 0,
                                 "lightrag_graph": {"entities": 0, "relationships": 0},
                                 "neo4j_graph": {"nodes": 1, "edges": 2, "triples": 2}},
            "evidence": N4_EVIDENCE, "summary": {},
            "coverage": {"neo4j_has_evidence": True, "notes": ["命中图谱"]}, "prompt_section": None}


def _fused_empty():
    return {"status": "ok",
            "evidence_sources": {"lightrag_vector": 0,
                                 "lightrag_graph": {"entities": 0, "relationships": 0},
                                 "neo4j_graph": {"nodes": 0, "edges": 0, "triples": 0}},
            "evidence": [], "summary": {},
            "coverage": {"neo4j_has_evidence": False, "notes": []}, "prompt_section": None}


def _resp_with_fused(fused, **overrides):
    cov = {
        "neo4j_has_evidence": (fused.get("coverage") or {}).get("neo4j_has_evidence", False),
        "notes": (fused.get("coverage") or {}).get("notes", []),
        "neo4j_unavailable": False,
    }
    if "coverage" in overrides and isinstance(overrides["coverage"], dict):
        cov.update(overrides["coverage"])
    overrides["coverage"] = cov
    return make_response(fused_context=fused, **overrides)


# ============================================================
# P3: 证据展示
# ============================================================

class TestEvidenceDisplay(unittest.TestCase):
    def test_no_evidence(self):
        r = _resp_with_fused(_fused_empty())
        ov = format_evidence_overview(r)
        self.assertFalse(ov["has_evidence"])
        self.assertEqual(ov["source_labels"], [])
        self.assertEqual(format_lightrag_evidence(r)["total"], 0)
        self.assertEqual(format_neo4j_evidence(r)["status"], "no_hit")

    def test_only_lightrag_evidence(self):
        r = _resp_with_fused(_fused_lr())
        ov = format_evidence_overview(r)
        self.assertTrue(ov["lightrag_has"])
        self.assertFalse(ov["neo4j_has"])
        self.assertEqual(ov["source_labels"], ["LightRAG"])
        self.assertEqual(format_lightrag_evidence(r)["total"], 3)
        self.assertEqual(format_neo4j_evidence(r)["status"], "no_hit")

    def test_neo4j_hit(self):
        r = _resp_with_fused(_fused_n4())
        ov = format_evidence_overview(r)
        self.assertTrue(ov["neo4j_has"])
        self.assertIn("Neo4j", ov["source_labels"])
        n4 = format_neo4j_evidence(r)
        self.assertEqual(n4["status"], "hit")
        self.assertEqual(n4["nodes"], 1)
        self.assertEqual(n4["edges"], 2)
        self.assertEqual(n4["triples_count"], 2)

    def test_neo4j_unavailable(self):
        r = _resp_with_fused(_fused_empty(),
                             neo4j_unavailable=True,
                             coverage={"neo4j_has_evidence": False, "notes": ["Neo4j 容器不可用"],
                                       "neo4j_unavailable": True})
        n4 = format_neo4j_evidence(r)
        self.assertEqual(n4["status"], "unavailable")
        self.assertTrue(n4["unavailable"])
        self.assertEqual(n4["triples"], [])           # 不伪造 triple

    def test_neo4j_available_but_no_hit(self):
        r = _resp_with_fused(_fused_empty())
        n4 = format_neo4j_evidence(r)
        self.assertEqual(n4["status"], "no_hit")      # 与 unavailable 严格区分
        self.assertFalse(n4["unavailable"])

    def test_triple_display(self):
        r = _resp_with_fused(_fused_n4())
        n4 = format_neo4j_evidence(r)
        self.assertGreaterEqual(len(n4["triples"]), 1)
        t = n4["triples"][0]
        self.assertEqual(t["source"], "二甲双胍")
        self.assertEqual(t["relation"], "TREATS")
        self.assertEqual(t["target"], "2型糖尿病")

    def test_node_relation_triple_counts(self):
        r = _resp_with_fused(_fused_n4())
        n4 = format_neo4j_evidence(r)
        self.assertEqual((n4["nodes"], n4["edges"], n4["triples_count"]), (1, 2, 2))

    def test_citations_with_source(self):
        c = format_citations(make_response()["citations"])
        self.assertTrue(c["present"])
        self.assertEqual(c["items"][0]["entity_name"], "高血压")
        self.assertTrue(c["items"][0]["grounded"])
        self.assertEqual(c["items"][0]["source_file"], "hypertension.md")
        self.assertEqual(c["grounding_method"], "entity_name_map")

    def test_citations_empty(self):
        c = format_citations({"citation_present": False, "reference_items": []})
        self.assertFalse(c["present"])
        self.assertEqual(c["items"], [])

    def test_coverage_notes(self):
        r = _resp_with_fused(_fused_n4())
        cov = format_coverage(r)
        self.assertIn("命中图谱", cov["notes"])       # 后端事实原样保留

    def test_mentions_mapping(self):
        r = _resp_with_fused(_fused_n4(), mentions=["高血压", "脑卒中"])
        self.assertEqual(format_coverage(r)["mentions"], ["高血压", "脑卒中"])

    def test_provenance_display(self):
        item = _display_evidence_item(N4_EVIDENCE[1])  # 跨源合并 ["neo4j_graph","lightrag_graph"]
        self.assertEqual(item["provenance"], ["Neo4j 图", "LightRAG 图"])
        self.assertEqual(item["kind_label"], "图谱三元组")

    def test_evidence_count_limited(self):
        r = _resp_with_fused(_fused_lr())
        lr = format_lightrag_evidence(r, max_items=2)
        self.assertEqual(len(lr["items"]), 2)
        self.assertEqual(lr["total"], 3)
        self.assertTrue(lr["has_more"])

    def test_display_does_not_mutate_original(self):
        r = _resp_with_fused(_fused_lr())
        before = list((r["fused_context"] or {}).get("evidence", []))
        format_lightrag_evidence(r, max_items=2)
        after = (r["fused_context"] or {}).get("evidence", [])
        self.assertEqual(before, after)               # 不改变原始 evidence

    def test_no_fake_neo4j_evidence_when_unavailable(self):
        r = _resp_with_fused(_fused_empty(), neo4j_unavailable=True,
                             coverage={"neo4j_has_evidence": False, "notes": ["不可用"],
                                       "neo4j_unavailable": True})
        n4 = format_neo4j_evidence(r)
        self.assertEqual(n4["status"], "unavailable")
        self.assertEqual(n4["triples"], [])
        self.assertFalse(n4["has_evidence"])

    def test_unavailable_not_treated_as_no_hit(self):
        r = _resp_with_fused(_fused_empty(), neo4j_unavailable=True,
                             coverage={"neo4j_has_evidence": False, "notes": ["不可用"],
                                       "neo4j_unavailable": True})
        self.assertNotEqual(format_neo4j_evidence(r)["status"], "no_hit")

    def test_no_leak_in_evidence(self):
        leak_item = _ev("lightrag_chunk", "c1",
                        {"chunk_id": "c1", "content": "password=supersecret key=sk-abc123"},
                        ["lightrag_vector"], 4)
        r = _resp_with_fused({"status": "ok",
                              "evidence_sources": {"lightrag_vector": 1,
                                                   "lightrag_graph": {"entities": 0, "relationships": 0},
                                                   "neo4j_graph": {"nodes": 0, "edges": 0, "triples": 0}},
                              "evidence": [leak_item], "summary": {},
                              "coverage": {"neo4j_has_evidence": False, "notes": []},
                              "prompt_section": None})
        lr = format_lightrag_evidence(r)
        dump = str(lr["items"])
        self.assertNotIn("supersecret", dump)
        self.assertNotIn("sk-abc123", dump)

    def test_blocked_evidence_no_crash(self):
        fr = make_response(safety_level="blocked", llm_generation_count=0,
                           fused_context=_fused_empty(),
                           coverage={"neo4j_has_evidence": False, "notes": [], "neo4j_unavailable": False})
        s = build_display_sections(fr)
        self.assertFalse(s["evidence_overview"]["has_evidence"])
        self.assertEqual(s["neo4j_evidence"]["status"], "not_checked")

    def test_refuse_evidence_no_crash(self):
        fr = make_response(reliability_decision=dict(_DECISION, decision="refuse",
                                                     is_final_refusal=True,
                                                     reason_code="DIAGNOSIS_INTENT"),
                           fused_context=_fused_empty(),
                           coverage={"neo4j_has_evidence": False, "notes": [], "neo4j_unavailable": False})
        s = build_display_sections(fr)
        self.assertTrue(s["decision"]["is_final_refusal"])
        self.assertIn("evidence_overview", s)

    def test_missing_fields_no_crash(self):
        r = {"answer": "hi", "fused_context": {"evidence_sources": {}}, "coverage": {},
             "citations": None, "reliability_decision": None, "safety_level": None}
        s = build_display_sections(r)
        self.assertFalse(s["evidence_overview"]["has_evidence"])
        self.assertEqual(s["neo4j_evidence"]["status"], "no_hit")
        self.assertEqual(s["lightrag_evidence"]["items"], [])

    def test_malformed_evidence_no_crash(self):
        fused = {"status": "ok",
                 "evidence_sources": {"lightrag_vector": 0, "lightrag_graph": {"entities": 0, "relationships": 0},
                                      "neo4j_graph": {"nodes": 0, "edges": 0, "triples": 0}},
                 "evidence": ["bad", None, {"kind": "neo4j_triple"}, {"kind": "lightrag_chunk", "payload": None},
                              {"kind": "unknown_kind", "name": "x", "payload": {"a": "b"}, "sources": None}],
                 "summary": {}, "coverage": {"neo4j_has_evidence": False, "notes": []}, "prompt_section": None}
        r = _resp_with_fused(fused)
        lr = format_lightrag_evidence(r)   # 仅 lightrag_chunk 计入，malformed 被跳过
        self.assertGreaterEqual(len(lr["items"]), 1)
        n4 = format_neo4j_evidence(r)      # 不崩溃
        self.assertIn(n4["status"], ("hit", "no_hit", "unavailable"))


# ============================================================
# P6-B: Neo4j 四态语义（not_checked / unavailable / hit / no_hit）
# ============================================================

class TestNeo4jFourStates(unittest.TestCase):
    def _resp(self, safety="safe", n4_unavail=False, has=False, **kw):
        return make_response(
            safety_level=safety,
            neo4j_unavailable=n4_unavail,
            fused_context=_fused_empty(),
            coverage={"neo4j_has_evidence": has, "notes": [],
                      "neo4j_unavailable": n4_unavail},
            **kw,
        )

    def test_blocked_is_not_checked(self):
        r = self._resp(safety="blocked")
        n4 = format_neo4j_evidence(r)
        self.assertEqual(n4["status"], "not_checked")
        self.assertEqual(n4["label"], "安全拦截，未执行图谱检索")

    def test_blocked_even_if_neo4j_unavailable_false(self):
        # BLOCKED + neo4j_unavailable=False → 仍 not_checked（不误判 no_hit）
        r = self._resp(safety="blocked", n4_unavail=False, has=False)
        self.assertEqual(format_neo4j_evidence(r)["status"], "not_checked")

    def test_blocked_produces_no_counts_or_triples(self):
        r = self._resp(safety="blocked")
        n4 = format_neo4j_evidence(r)
        self.assertEqual((n4["nodes"], n4["edges"], n4["triples_count"]), (0, 0, 0))
        self.assertEqual(n4["triples"], [])

    def test_blocked_not_no_hit(self):
        r = self._resp(safety="blocked")
        self.assertNotEqual(format_neo4j_evidence(r)["status"], "no_hit")

    def test_blocked_wins_over_unavailable_and_hit(self):
        # 优先级：blocked > 其它（即使字段表明 unavailable/hit 也归 not_checked）
        r_unavail = self._resp(safety="blocked", n4_unavail=True)
        r_hit = self._resp(safety="blocked", has=True)
        self.assertEqual(format_neo4j_evidence(r_unavail)["status"], "not_checked")
        self.assertEqual(format_neo4j_evidence(r_hit)["status"], "not_checked")

    def test_unavailable_unchanged(self):
        r = self._resp(safety="safe", n4_unavail=True)
        n4 = format_neo4j_evidence(r)
        self.assertEqual(n4["status"], "unavailable")
        self.assertEqual(n4["label"], "不可用（未使用结构化图谱证据）")

    def test_no_hit_unchanged(self):
        r = self._resp(safety="safe", n4_unavail=False, has=False)
        self.assertEqual(format_neo4j_evidence(r)["status"], "no_hit")

    def test_hit_unchanged(self):
        r = self._resp(safety="safe", n4_unavail=False, has=True)
        n4 = format_neo4j_evidence(r)
        self.assertEqual(n4["status"], "hit")
        self.assertEqual(n4["label"], "已提供结构化图谱证据")

    def test_priority_non_blocked(self):
        # 非 BLOCKED：unavailable > hit > no_hit
        self.assertEqual(format_neo4j_evidence(self._resp("safe", True, True))["status"], "unavailable")
        self.assertEqual(format_neo4j_evidence(self._resp("safe", False, True))["status"], "hit")
        self.assertEqual(format_neo4j_evidence(self._resp("safe", False, False))["status"], "no_hit")

    def test_no_fake_triple(self):
        # 非 hit 状态 triples 为空（不伪造）
        r = self._resp(safety="blocked")
        self.assertEqual(format_neo4j_evidence(r)["triples"], [])
        self.assertEqual(format_neo4j_evidence(self._resp("safe", False, False))["triples"], [])

    def test_other_sections_untouched_for_blocked(self):
        r = self._resp(safety="blocked", citations=dict(_CITATIONS),
                       reliability_decision=dict(_DECISION))
        s = build_display_sections(r)
        self.assertTrue(s["citations"]["present"])          # citations 展示不受影响
        self.assertEqual(s["decision"]["decision"], "revise")  # decision 展示不受影响
        self.assertEqual(s["safety"]["level"], "blocked")      # safety 展示不受影响
        self.assertEqual(s["system"]["neo4j_status"], "not_checked")  # 系统状态同步四态

    def test_missing_fields_not_crash(self):
        self.assertEqual(format_neo4j_evidence({})["status"], "no_hit")
        self.assertIn(format_neo4j_evidence(None)["status"], ("no_hit", "unavailable"))


# ============================================================
# P6-C: KG 可视化图数据（build_kg_graph_data）
# ============================================================

def _fused_big(n_triples=25):
    ev = (
        [_ev("neo4j_triple", f"t{i}",
             {"source": f"实体{i}", "relation_type": "TREATS", "target": "2型糖尿病",
              "canonical_source": f"实体{i}", "canonical_target": "2型糖尿病"},
             ["neo4j_graph"], 0) for i in range(n_triples)]
        + [_ev("neo4j_node", f"实体{i}",
               {"id": f"实体{i}", "name": f"实体{i}", "entity_type": "disease",
                "labels": ["Disease"], "properties": {}},
               ["neo4j_graph"], 1) for i in range(n_triples)]
    )
    return {"status": "ok",
            "evidence_sources": {"lightrag_vector": 0,
                                 "lightrag_graph": {"entities": 0, "relationships": 0},
                                 "neo4j_graph": {"nodes": n_triples, "edges": n_triples, "triples": n_triples}},
            "evidence": ev, "summary": {},
            "coverage": {"neo4j_has_evidence": True, "notes": []}, "prompt_section": None}


class TestKgGraph(unittest.TestCase):
    def test_hit_one_triple_generates_graph(self):
        r = _resp_with_fused(_fused_n4())
        kg = build_kg_graph_data(r)
        self.assertEqual(kg["status"], "hit")
        self.assertTrue(kg["has_graph"])
        ids = {n["id"] for n in kg["nodes"]}
        self.assertIn("二甲双胍", ids)
        self.assertIn("2型糖尿病", ids)
        self.assertTrue(any(e["source"] == "二甲双胍" and e["target"] == "2型糖尿病"
                            and e["label"] == "TREATS" for e in kg["edges"]))

    def test_treats_label_preserved(self):
        r = _resp_with_fused(_fused_n4())
        kg = build_kg_graph_data(r)
        self.assertEqual([e["label"] for e in kg["edges"] if e["label"] == "TREATS"], ["TREATS"])

    def test_has_complication_label_preserved(self):
        r = _resp_with_fused(_fused_n4())
        kg = build_kg_graph_data(r)
        self.assertTrue(any(e["label"] == "HAS_COMPLICATION" for e in kg["edges"]))

    def test_nodes_dedup(self):
        r = _resp_with_fused(_fused_n4())
        kg = build_kg_graph_data(r)
        ids = [n["id"] for n in kg["nodes"]]
        self.assertEqual(len(ids), len(set(ids)))   # 去重

    def test_edges_dedup(self):
        fused = dict(_fused_n4())
        # 重复一份 TREATS triple
        dup = _ev("neo4j_triple", "dup",
                  {"source": "二甲双胍", "relation_type": "TREATS", "target": "2型糖尿病",
                   "canonical_source": "二甲双胍", "canonical_target": "2型糖尿病"},
                  ["neo4j_graph"], 0)
        fused = dict(fused, evidence=fused["evidence"] + [dup])
        kg = build_kg_graph_data(_resp_with_fused(fused))
        treats = [e for e in kg["edges"] if e["label"] == "TREATS"]
        self.assertEqual(len(treats), 1)             # (s,rel,t) 去重

    def test_max_nodes(self):
        kg = build_kg_graph_data(_resp_with_fused(_fused_big(25)), max_nodes=20, max_edges=20)
        self.assertLessEqual(len(kg["nodes"]), 20)

    def test_max_edges(self):
        kg = build_kg_graph_data(_resp_with_fused(_fused_big(25)), max_nodes=20, max_edges=20)
        self.assertLessEqual(len(kg["edges"]), 20)

    def test_truncation_reported(self):
        kg = build_kg_graph_data(_resp_with_fused(_fused_big(25)), max_nodes=20, max_edges=20)
        self.assertTrue(kg["has_more"])
        self.assertEqual(kg["node_truncated"], kg["node_total"] - len(kg["nodes"]))
        self.assertEqual(kg["edge_truncated"], kg["edge_total"] - len(kg["edges"]))

    def test_unavailable_no_graph(self):
        r = _resp_with_fused(_fused_empty(), neo4j_unavailable=True,
                             coverage={"neo4j_has_evidence": False, "notes": ["不可用"],
                                       "neo4j_unavailable": True})
        kg = build_kg_graph_data(r)
        self.assertFalse(kg["has_graph"])
        self.assertEqual(kg["nodes"], [])
        self.assertEqual(kg["status"], "unavailable")

    def test_no_hit_no_graph(self):
        kg = build_kg_graph_data(_resp_with_fused(_fused_empty()))
        self.assertFalse(kg["has_graph"])
        self.assertEqual(kg["nodes"], [])
        self.assertEqual(kg["status"], "no_hit")

    def test_not_checked_no_graph(self):
        r = _resp_with_fused(_fused_empty(), safety_level="blocked")
        kg = build_kg_graph_data(r)
        self.assertFalse(kg["has_graph"])
        self.assertEqual(kg["status"], "not_checked")

    def test_blocked_no_graph(self):
        r = _resp_with_fused(_fused_n4(), safety_level="blocked")
        kg = build_kg_graph_data(r)
        self.assertFalse(kg["has_graph"])          # 即使有 evidence，blocked 也优先
        self.assertEqual(kg["status"], "not_checked")

    def test_lightrag_only_no_graph(self):
        kg = build_kg_graph_data(_resp_with_fused(_fused_lr()))
        self.assertFalse(kg["has_graph"])          # 仅 LightRAG 证据 → 不出 Neo4j 图

    def test_mentions_not_used_to_build_graph(self):
        # mentions 存在但无 neo4j evidence → 不据此构造节点/边
        r = _resp_with_fused(_fused_empty(), mentions=["高血压", "脑卒中"])
        kg = build_kg_graph_data(r)
        self.assertFalse(kg["has_graph"])
        self.assertEqual(kg["nodes"], [])

    def test_does_not_mutate_input(self):
        fused = _fused_n4()
        before = json.dumps(fused, ensure_ascii=False, sort_keys=True)
        build_kg_graph_data(_resp_with_fused(fused))
        self.assertEqual(json.dumps(fused, ensure_ascii=False, sort_keys=True), before)

    def test_malformed_evidence_no_crash(self):
        fused = dict(_fused_n4(),
                     evidence=["bad", None, {"kind": "neo4j_triple"},
                               {"kind": "neo4j_triple", "sources": ["neo4j_graph"], "payload": None},
                               {"kind": "neo4j_node", "sources": ["neo4j_graph"], "payload": {}}])
        kg = build_kg_graph_data(_resp_with_fused(fused))
        self.assertEqual(kg["status"], "hit")       # 有合法 triple 仍 hit
        self.assertIsInstance(kg["nodes"], list)
        self.assertIsInstance(kg["edges"], list)

    def test_sensitive_masked(self):
        leak = _ev("neo4j_node", "secret_node",
                   {"id": "secret_node", "name": "password=supersecret", "entity_type": "x",
                    "labels": [], "properties": {}}, ["neo4j_graph"], 1)
        fused = dict(_fused_empty(), evidence=[leak])
        fused["coverage"]["neo4j_has_evidence"] = True
        kg = build_kg_graph_data(_resp_with_fused(fused))
        dump = json.dumps(kg["nodes"], ensure_ascii=False)
        self.assertNotIn("supersecret", dump)

    def test_normal_medical_names_not_masked(self):
        r = _resp_with_fused(_fused_n4())
        kg = build_kg_graph_data(r)
        labels = [n["label"] for n in kg["nodes"]]
        self.assertIn("二甲双胍", labels)            # 正常医学实体不误 mask
        self.assertIn("2型糖尿病", labels)

    def test_kg_section_in_display(self):
        s = build_display_sections(_resp_with_fused(_fused_n4()))
        self.assertIn("kg_graph", s)
        self.assertTrue(s["kg_graph"]["has_graph"])


class TestPhase9P0MentalHotline(unittest.TestCase):
    """Phase 9 P0 — 心理危机拦截文案展示校正（normalize_mental_health_hotline）。"""

    _CRISIS_MSG = (
        "检测到您的问题涉及心理健康危机。\n\n"
        "❤️ **心理健康提醒**\n\n"
        "如果您正在经历心理困扰，请知道您并不孤单。\n"
        "- 全国24小时心理援助热线：400-161-9995\n"
        "- 北京心理危机研究与干预中心：010-82951332\n"
        "- 生命热线：400-821-1215\n\n"
        "请尽快联系专业心理咨询师或精神科医生。"
    )

    def test_removes_beijing_hotline_line(self):
        out = normalize_mental_health_hotline(self._CRISIS_MSG)
        self.assertNotIn("北京心理危机", out)
        self.assertNotIn("010-82951332", out)

    def test_adds_national_unified_hotline(self):
        out = normalize_mental_health_hotline(self._CRISIS_MSG)
        self.assertIn("全国统一心理援助热线：12356", out)

    def test_keeps_other_hotlines(self):
        out = normalize_mental_health_hotline(self._CRISIS_MSG)
        self.assertIn("400-161-9995", out)
        self.assertIn("400-821-1215", out)

    def test_hotline_bullet_position_after_first_hotline(self):
        out = normalize_mental_health_hotline(self._CRISIS_MSG)
        lines = out.splitlines()
        i_24 = next(i for i, ln in enumerate(lines) if "400-161-9995" in ln)
        self.assertEqual(lines[i_24 + 1].strip(), "- 全国统一心理援助热线：12356")

    def test_idempotent_no_duplicate_national_line(self):
        once = normalize_mental_health_hotline(self._CRISIS_MSG)
        twice = normalize_mental_health_hotline(once)
        self.assertEqual(once.count("12356"), 1)
        self.assertEqual(twice, once)

    def test_already_has_12356_no_duplicate(self):
        msg = "- 全国统一心理援助热线：12356\n请及时求助。"
        out = normalize_mental_health_hotline(msg)
        self.assertEqual(out.count("12356"), 1)

    def test_non_crisis_text_unchanged(self):
        msg = "二甲双胍主要用于治疗2型糖尿病。"
        self.assertEqual(normalize_mental_health_hotline(msg), msg)

    def test_empty_text_unchanged(self):
        self.assertEqual(normalize_mental_health_hotline(""), "")
        self.assertEqual(normalize_mental_health_hotline(None), None)


class TestPhase9P0KgCentering(unittest.TestCase):
    """Phase 9 P0 — pyvis HTML 注入居中脚本（inject_kg_centering，同作用域注入）。"""

    _PLAIN_HTML = ("<html><head></head><body><div id='mynetwork'></div>"
                   "<script>var network; drawGraph();</script></body></html>")

    def test_injects_into_last_script_same_scope(self):
        out = inject_kg_centering(self._PLAIN_HTML)
        self.assertIn("Phase 9 P0: 图谱居中", out)
        self.assertLess(out.index("Phase 9 P0"), out.rindex("</script>"))
        # 注入点在 drawGraph() 之后、同一 <script> 内
        self.assertLess(out.index("drawGraph()"), out.index("network.fit"))

    def test_fit_and_disable_physics_present(self):
        out = inject_kg_centering(self._PLAIN_HTML)
        self.assertIn("network.fit({ animation: false })", out)
        self.assertIn("physics: { enabled: false }", out)

    def test_idempotent_no_double_injection(self):
        once = inject_kg_centering(self._PLAIN_HTML)
        twice = inject_kg_centering(once)
        self.assertEqual(once.count("Phase 9 P0: 图谱居中"), 1)
        self.assertEqual(twice, once)

    def test_appends_when_no_script_tag(self):
        out = inject_kg_centering("<html><body></body></html>")
        self.assertIn("Phase 9 P0: 图谱居中", out)

    def test_empty_html_unchanged(self):
        self.assertEqual(inject_kg_centering(""), "")
        self.assertEqual(inject_kg_centering(None), None)


class TestPhase9P0KgLayout(unittest.TestCase):
    """Phase 9 P0 — networkx 确定性居中布局（build_kg_layout）。"""

    def _nodes_edges(self, n=10):
        nodes = [{"id": f"n{i}"} for i in range(n)]
        edges = [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(n - 1)]
        return nodes, edges

    def test_returns_positions_for_all_nodes(self):
        nodes, edges = self._nodes_edges()
        pos = build_kg_layout(nodes, edges)
        self.assertEqual(set(pos.keys()), {n["id"] for n in nodes})
        for v in pos.values():
            self.assertIn("x", v)
            self.assertIn("y", v)

    def test_deterministic_with_seed(self):
        nodes, edges = self._nodes_edges()
        a = build_kg_layout(nodes, edges)
        b = build_kg_layout(nodes, edges)
        self.assertEqual(a, b)

    def test_centered_around_origin(self):
        nodes, edges = self._nodes_edges(20)
        pos = build_kg_layout(nodes, edges)
        xs = [v["x"] for v in pos.values()]
        ys = [v["y"] for v in pos.values()]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        self.assertLess(abs(cx), 1.0)   # 水平居中到原点
        self.assertLess(abs(cy), 1.0)   # 垂直居中到原点

    def test_scaled_within_canvas(self):
        nodes, edges = self._nodes_edges(20)
        pos = build_kg_layout(nodes, edges, canvas_w=560, canvas_h=420)
        xs = [v["x"] for v in pos.values()]
        ys = [v["y"] for v in pos.values()]
        self.assertLessEqual(max(abs(x) for x in xs), 560 / 2)
        self.assertLessEqual(max(abs(y) for y in ys), 420 / 2)

    def test_empty_nodes_returns_empty(self):
        self.assertEqual(build_kg_layout([], []), {})

    def test_single_node_returns_position(self):
        pos = build_kg_layout([{"id": "n1"}], [])
        self.assertEqual(set(pos.keys()), {"n1"})


class TestConversationContext(unittest.TestCase):
    """Phase 9 P2 — build_conversation_context（多轮 Memory 构建）。"""

    def _hist(self, *responses):
        return [{"question": f"Q{i}", "final_response": resp}
                for i, resp in enumerate(responses, 1)]

    def _safe_resp(self, q="二甲双胍用于治疗什么疾病？", ans="二甲双胍用于治疗2型糖尿病。", **ov):
        r = make_response(question=q, answer=ans, safety_level="safe", **ov)
        r["reliability_decision"] = dict(_DECISION, decision="allow", is_final_refusal=False)
        return r

    def test_empty_history(self):
        self.assertEqual(build_conversation_context([]), "")
        self.assertEqual(build_conversation_context(None), "")
        self.assertEqual(build_conversation_context([{}]), "")

    def test_safe_turn_included(self):
        h = self._hist(self._safe_resp())
        ctx = build_conversation_context(h)
        self.assertIn("Q1", ctx)
        self.assertIn("用户问题：", ctx)
        self.assertIn("助手回答：", ctx)
        self.assertIn("二甲双胍用于治疗2型糖尿病。", ctx)

    def test_caution_included(self):
        h = self._hist(make_response(answer="高血压需要综合管理。"))
        self.assertIn("高血压需要综合管理。", build_conversation_context(h))

    def test_revise_is_eligible(self):
        rev = make_response(reliability_decision=dict(_DECISION, decision="revise",
                                                       is_final_refusal=False),
                            answer="高血压需综合管理。")
        h = self._hist(rev)
        self.assertIn("高血压需综合管理。", build_conversation_context(h))

    def test_error_skipped(self):
        err = make_response(status="error", answer="", errors=[{"type": "Boom"}])
        h = self._hist(err, self._safe_resp(q="Q2", ans="A2"))
        ctx = build_conversation_context(h)
        self.assertNotIn("Boom", ctx)
        self.assertIn("A2", ctx)          # 合格轮保留

    def test_blocked_skipped(self):
        blk = make_response(safety_level="blocked",
                            reliability_decision=dict(_DECISION, decision="refuse",
                                                       reason_code="SAFETY_BLOCKED",
                                                       is_final_refusal=True),
                            answer="检测到安全边界。")
        self.assertEqual(build_conversation_context(self._hist(blk)), "")

    def test_refusal_skipped(self):
        ref = make_response(reliability_decision=dict(_DECISION, decision="refuse",
                                                       reason_code="DIAGNOSIS_INTENT",
                                                       is_final_refusal=True),
                            answer="本系统无法诊断。")
        self.assertEqual(build_conversation_context(self._hist(ref)), "")

    def test_three_turn_cap(self):
        h = self._hist(*([self._safe_resp()] * 5))     # 5 个合格轮
        ctx = build_conversation_context(h)
        self.assertEqual(ctx.count("用户问题："), 3)   # 只保留最近 3

    def test_600_char_cap_single_long(self):
        h = self._hist(self._safe_resp(q="Q1", ans="长" * 500))
        ctx = build_conversation_context(h, max_chars=600)
        self.assertLessEqual(len(ctx), 600)

    def test_600_char_cap_multi(self):
        h = self._hist(self._safe_resp(q="Q1", ans="A1" * 200),
                       self._safe_resp(q="Q2", ans="A2" * 200),
                       self._safe_resp(q="Q3", ans="A3" * 200))
        ctx = build_conversation_context(h, max_chars=600)
        self.assertLessEqual(len(ctx), 600)

    def test_mask_sensitive(self):
        h = self._hist(self._safe_resp(ans="我的 API key 是 sk-abc12345XYZ"))
        ctx = build_conversation_context(h)
        self.assertNotIn("sk-abc12345XYZ", ctx)

    def test_no_traceback_or_internal_structure(self):
        h = self._hist(self._safe_resp())
        ctx = build_conversation_context(h)
        for forbidden in ("fused_context", "reliability_decision", "citations",
                          "coverage", "neo4j", "Traceback", "conversation_context"):
            self.assertNotIn(forbidden, ctx)

    def test_clear_chat(self):
        h = self._hist(self._safe_resp())
        fresh = reset_chat_state()
        self.assertEqual(build_conversation_context(fresh["history"]), "")

    def test_session_isolation(self):
        # 两个独立 session 的 history：Memory 互不串线（以答案文本区分会话主题）
        hA = self._hist(self._safe_resp(ans="治疗2型糖尿病"))
        hB = self._hist(self._safe_resp(ans="需综合管理"))
        ctxA = build_conversation_context(hA)
        ctxB = build_conversation_context(hB)
        self.assertIn("治疗2型糖尿病", ctxA)
        self.assertNotIn("治疗2型糖尿病", ctxB)
        self.assertIn("需综合管理", ctxB)
        self.assertNotIn("需综合管理", ctxA)

    def test_agent_passthrough(self):
        calls = []

        class RecAgent:
            async def run(self, question, conversation_context=None):
                calls.append((question, conversation_context))
                return make_response()

        fr = run_agent_sync(RecAgent(), "Q", conversation_context="MEM")
        self.assertEqual(calls, [("Q", "MEM")])
        self.assertNotEqual(fr.get("status"), "error")

    def test_no_memory_compat(self):
        calls = []

        class RecAgent1:
            async def run(self, question):
                calls.append(question)
                return make_response()

        run_agent_sync(RecAgent1(), "Q")                # 无 memory → 不传 conversation_context
        self.assertEqual(calls, ["Q"])

    def test_current_turn_not_included(self):
        h = self._hist(self._safe_resp(q="二甲双胍"))
        ctx = build_conversation_context(h)
        self.assertNotIn("那它有哪些副作用", ctx)       # 当前轮尚未进 history


class TestMemoryFlow(unittest.TestCase):
    """Phase 9 P3 — 多轮 Memory 功能验收（M1-M12 场景，纯函数层）。"""

    def _hist_resp(self, q, ans, decision="allow", safety="safe", refusal=False,
                   error=False):
        rd = dict(_DECISION, decision=decision, is_final_refusal=refusal)
        if refusal:
            rd.update(reason_code="DIAGNOSIS_INTENT")
        r = make_response(question=q, answer=ans, safety_level=safety,
                          reliability_decision=rd)
        if error:
            r["status"] = "error"
            r["answer"] = ""
        return {"question": q, "final_response": r}

    def test_m1_normal_followup_context(self):
        h = [self._hist_resp("二甲双胍用于治疗什么疾病？", "二甲双胍用于治疗2型糖尿病。")]
        ctx = build_conversation_context(h)
        self.assertIn("二甲双胍用于治疗什么疾病？", ctx)
        self.assertIn("二甲双胍用于治疗2型糖尿病。", ctx)

    def test_m2_ellipsis_subject_context(self):
        # "那最常见的是哪些？" 需要依赖前文"高血压常见并发症"
        h = [self._hist_resp("高血压有哪些常见并发症？", "高血压常见并发症包括脑卒中、冠心病等。")]
        ctx = build_conversation_context(h)
        self.assertIn("高血压有哪些常见并发症？", ctx)
        self.assertIn("脑卒中", ctx)

    def test_m3_topic_switch_memory_only_prior(self):
        h = [self._hist_resp("二甲双胍用于治疗什么疾病？", "治疗2型糖尿病。")]
        ctx = build_conversation_context(h)
        self.assertIn("二甲双胍", ctx)              # memory 含 Q1 背景
        self.assertNotIn("高血压", ctx)             # 与当前 Q2 主题无关，不预置

    def test_m4_blocked_not_in_memory(self):
        q1 = self._hist_resp("二甲双胍用于治疗什么疾病？", "治疗2型糖尿病。")
        q2 = self._hist_resp("我不想活了", "检测到安全边界。", decision="refuse",
                             safety="blocked", refusal=True)
        ctx = build_conversation_context([q1, q2])
        self.assertIn("治疗2型糖尿病。", ctx)
        self.assertNotIn("我不想活了", ctx)
        self.assertNotIn("检测到安全边界", ctx)

    def test_m5_blocked_then_recover(self):
        h = [self._hist_resp("二甲双胍用于治疗什么疾病？", "治疗2型糖尿病。"),
             self._hist_resp("我不想活了", "检测到安全边界。", decision="refuse",
                             safety="blocked", refusal=True)]
        ctx = build_conversation_context(h)
        self.assertIn("治疗2型糖尿病。", ctx)       # Q3 可用 Q1 背景
        self.assertNotIn("检测到安全边界", ctx)     # BLOCKED 轮不污染

    def test_m6_diagnosis_refusal_not_in_memory(self):
        q1 = self._hist_resp("二甲双胍用于治疗什么疾病？", "治疗2型糖尿病。")
        q2 = self._hist_resp("你能诊断我是不是糖尿病吗？", "本系统无法诊断。",
                             decision="refuse", refusal=True)
        ctx = build_conversation_context([q1, q2])
        self.assertIn("治疗2型糖尿病。", ctx)
        self.assertNotIn("无法诊断", ctx)

    def test_m7_refusal_then_recover(self):
        h = [self._hist_resp("二甲双胍用于治疗什么疾病？", "治疗2型糖尿病。"),
             self._hist_resp("你能诊断我是不是糖尿病吗？", "本系统无法诊断。",
                             decision="refuse", refusal=True)]
        ctx = build_conversation_context(h)
        self.assertIn("治疗2型糖尿病。", ctx)       # Q3 正常背景可用
        self.assertNotIn("无法诊断", ctx)

    def test_m8_error_then_recover(self):
        h = [self._hist_resp("二甲双胍用于治疗什么疾病？", "治疗2型糖尿病。"),
             self._hist_resp("坏问题", "X", error=True)]
        ctx = build_conversation_context(h)
        self.assertIn("治疗2型糖尿病。", ctx)
        self.assertNotIn("坏问题", ctx)             # error 轮不进入

    def test_m9_clear_chat(self):
        h = [self._hist_resp("二甲双胍用于治疗什么疾病？", "治疗2型糖尿病。"),
             self._hist_resp("高血压有哪些常见并发症？", "脑卒中等。")]
        fresh = reset_chat_state()
        self.assertEqual(fresh["messages"], [])
        self.assertEqual(fresh["history"], [])
        self.assertIsNone(fresh["last_response"])
        self.assertEqual(build_conversation_context(fresh["history"]), "")

    def test_m10_history_limit(self):
        h = [self._hist_resp(f"Q{i}", f"A{i}") for i in range(5)]
        ctx = build_conversation_context(h, max_turns=3, max_chars=600)
        self.assertLessEqual(ctx.count("用户问题："), 3)   # 最多 3 个合格轮
        self.assertLessEqual(len(ctx), 600)

    def test_m11_current_turn_not_in_memory(self):
        prior = [self._hist_resp("二甲双胍用于治疗什么疾病？", "治疗2型糖尿病。")]
        ctx = build_conversation_context(prior)
        self.assertNotIn("那它有哪些常见副作用？", ctx)   # 当前问题未进 history

    def test_m12_masking_multiple_sensitive(self):
        ans = ("密钥 sk-abc12345XYZ, password=secret123, token=abc12345, "
               "服务 http://user:pass@example.com/db")
        h = [self._hist_resp("Q", ans)]
        ctx = build_conversation_context(h)
        for secret in ("sk-abc12345XYZ", "secret123", "abc12345", "user:pass@example.com"):
            self.assertNotIn(secret, ctx)


# ============================================================
# Phase 9 P4: memory_used_hint（「参考前文」提示判定，纯函数）
# ============================================================

class TestMemoryUsedHint(unittest.TestCase):
    """Phase 9 P4 — memory_used_hint（仅当传入了 memory 且为实质回答才提示）。"""

    _HINT = "本次回答已结合前文对话上下文"

    def _allow_response(self, **ov):
        return make_response(reliability_decision=dict(_DECISION, decision="allow",
                                                       is_final_refusal=False), **ov)

    def test_no_memory_no_hint(self):
        self.assertIsNone(memory_used_hint("", self._allow_response()))
        self.assertIsNone(memory_used_hint(None, self._allow_response()))

    def test_memory_normal_answer_hint(self):
        hint = memory_used_hint("用户问题：Q1\n助手回答：A1\n", self._allow_response())
        self.assertIn(self._HINT, hint or "")

    def test_memory_caution_revise_hint(self):
        r = make_response(reliability_decision=dict(_DECISION, decision="revise",
                                                    is_final_refusal=False),
                          answer="高血压需综合管理。")
        self.assertIn(self._HINT, memory_used_hint("MEM", r) or "")

    def test_memory_blocked_no_hint(self):
        r = make_response(safety_level="blocked",
                          reliability_decision=dict(_DECISION, decision="refuse",
                                                     reason_code="SAFETY_BLOCKED",
                                                     is_final_refusal=True))
        self.assertIsNone(memory_used_hint("MEM", r))

    def test_memory_refusal_no_hint(self):
        r = make_response(reliability_decision=dict(_DECISION, decision="refuse",
                                                    reason_code="DIAGNOSIS_INTENT",
                                                    is_final_refusal=True))
        self.assertIsNone(memory_used_hint("MEM", r))

    def test_memory_error_no_hint(self):
        r = make_response(status="error", answer="", errors=[{"type": "Boom"}])
        self.assertIsNone(memory_used_hint("MEM", r))

    def test_memory_none_response_no_hint(self):
        self.assertIsNone(memory_used_hint("MEM", None))

    def test_does_not_mutate_response(self):
        r = self._allow_response()
        before = str(r)
        memory_used_hint("MEM", r)
        self.assertEqual(str(r), before)


# ============================================================
# Phase 9 P4: UI 打磨静态内容（答辩准备）
# ============================================================

class TestPhase9P4UiPolish(unittest.TestCase):
    """Phase 9 P4 — app 层静态内容（示例/关于本系统/首轮提示/hint 使用）。"""

    def test_examples_include_cross_entity(self):
        src = Path(_PROJECT_ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("高血压和脑卒中有什么关系？", src)

    def test_about_system_expander_present(self):
        src = Path(_PROJECT_ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("关于本系统", src)
        self.assertIn("LangGraph Agent", src)

    def test_first_query_spinner_note_present(self):
        src = Path(_PROJECT_ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("首次查询需加载检索内核", src)
        self.assertIn("_has_queried", src)

    def test_memory_hint_used_in_app(self):
        src = Path(_PROJECT_ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("memory_used_hint", src)


if __name__ == "__main__":
    unittest.main()
