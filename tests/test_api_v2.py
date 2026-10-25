"""Tests for the v2 streaming API without a live LLM or Neo4j."""

import asyncio
import json
import unittest

from fastapi.testclient import TestClient

from api.main import app, get_runtime
from api.toolbox import get_toolbox
from api.runtime import AgentRuntime, history_to_entries, sse_event
from api.schemas import ChatRequest, HistoryTurn


def _parse_sse(chunk: str) -> dict:
    line = next(line for line in chunk.splitlines() if line.startswith("data: "))
    return json.loads(line[6:])


class FakeAgent:
    def __init__(self) -> None:
        self.questions: list[str] = []
        self.contexts: list[str | None] = []

    async def astream_state(self, question: str, conversation_context=None):
        self.questions.append(question)
        self.contexts.append(conversation_context)
        yield {
            "safety_check": {
                "safety_level": "safe",
                "mentions": [],
            }
        }
        yield {
            "neo4j_retrieve": {
                "mentions": ["二甲双胍"],
                "neo4j_unavailable": False,
            }
        }
        yield {"build_prompt": {"system_prompt": "evidence"}}
        yield {"generate": {"llm_response": "test answer"}}
        yield {"analyze": {"metrics": {}}}
        yield {"decide": {"reliability_decision": {"decision": "allow"}}}
        yield {
            "finalize": {
                "final_response": {
                    "question": question,
                    "answer": "二甲双胍用于治疗2型糖尿病。",
                    "safety_level": "safe",
                    "mentions": ["二甲双胍"],
                    "llm_generation_count": 1,
                    "reliability_decision": {
                        "decision": "allow",
                        "is_final_refusal": False,
                    },
                    "fused_context": {"evidence_sources": {}},
                    "coverage": {},
                    "errors": [],
                }
            }
        }


class BlockedAgent(FakeAgent):
    async def astream_state(self, question: str, conversation_context=None):
        yield {"safety_check": {"safety_level": "blocked"}}
        yield {
            "decide": {
                "reliability_decision": {
                    "decision": "refuse",
                    "is_final_refusal": True,
                }
            }
        }
        yield {
            "finalize": {
                "final_response": {
                    "question": question,
                    "answer": "安全拦截。",
                    "safety_level": "blocked",
                    "llm_generation_count": 0,
                    "reliability_decision": {
                        "decision": "refuse",
                        "is_final_refusal": True,
                    },
                    "fused_context": {"evidence_sources": {}},
                    "coverage": {},
                    "errors": [],
                }
            }
        }


class TestSerialization(unittest.TestCase):
    def test_sse_event_is_json(self):
        payload = _parse_sse(sse_event({"type": "status", "label": "测试"}))
        self.assertEqual(payload, {"type": "status", "label": "测试"})

    def test_history_mapping(self):
        entries = history_to_entries(
            [
                HistoryTurn(
                    question="前一个问题",
                    answer="前一个回答",
                    safety_level="safe",
                )
            ]
        )
        self.assertEqual(entries[0]["question"], "前一个问题")
        self.assertEqual(
            entries[0]["final_response"]["reliability_decision"][
                "is_final_refusal"
            ],
            False,
        )


class TestAgentRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_stream_emits_status_delta_and_final(self):
        fake = FakeAgent()
        runtime = AgentRuntime(fake, chunk_delay=0)
        chunks = [
            chunk
            async for chunk in runtime.stream(
                ChatRequest(question="二甲双胍治疗什么？")
            )
        ]
        events = [_parse_sse(chunk) for chunk in chunks]
        types = [event["type"] for event in events]
        self.assertIn("status", types)
        self.assertIn("tool", types)
        self.assertIn("answer_delta", types)
        self.assertEqual(types[-1], "final")
        self.assertIn("二甲双胍", "".join(
            event.get("delta", "") for event in events
        ))

    async def test_blocked_workflow_still_completes(self):
        runtime = AgentRuntime(BlockedAgent(), chunk_delay=0)
        events = [
            _parse_sse(chunk)
            async for chunk in runtime.stream(ChatRequest(question="测试"))
        ]
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(events[-1]["response"]["safety_level"], "blocked")


class TestApi(unittest.TestCase):
    def test_health_and_stream_endpoint(self):
        fake = FakeAgent()
        app.dependency_overrides[get_runtime] = lambda: AgentRuntime(
            fake, chunk_delay=0
        )
        try:
            with TestClient(app) as client:
                health = client.get("/api/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json()["version"], "2.0.0")

                response = client.post(
                    "/api/chat/stream",
                    json={"question": "二甲双胍治疗什么？", "history": []},
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn("text/event-stream", response.headers["content-type"])
                self.assertIn('"type":"final"', response.text)
                self.assertEqual(fake.questions[-1], "二甲双胍治疗什么？")
        finally:
            app.dependency_overrides.clear()

    def test_tools_endpoint_and_execution(self):
        class FakeToolbox:
            async def call_tool(self, name, arguments):
                return {
                    "tool": name,
                    "status": "success",
                    "arguments": arguments,
                    "data": {"count": 1},
                    "latency_ms": 1.0,
                }

        app.dependency_overrides[get_toolbox] = lambda: FakeToolbox()
        try:
            with TestClient(app) as client:
                tools = client.get("/api/tools")
                self.assertEqual(tools.status_code, 200)
                names = {item["name"] for item in tools.json()["tools"]}
                self.assertIn("search_medical_documents", names)
                self.assertIn("get_entity_relations", names)

                result = client.post(
                    "/api/tools/execute",
                    json={
                        "tool": "search_graph_entities",
                        "arguments": {"query": "二甲双胍"},
                    },
                )
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()["status"], "success")
        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
