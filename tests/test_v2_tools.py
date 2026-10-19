"""Tests for the controlled v2 medical tool layer."""

import asyncio
import sys
import unittest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_server.medqa_mcp import mcp
from src.tools.medical_tools import (
    MedicalToolbox,
    TOOL_DEFINITIONS,
    ToolExecutionError,
)


class FakeRagEngine:
    def __init__(self):
        self.calls = []

    async def retrieve(self, query, mode="mix", top_k=60, chunk_top_k=20):
        self.calls.append((query, mode, top_k, chunk_top_k))
        return {
            "status": "success",
            "data": {
                "entities": [
                    {
                        "entity_name": "二甲双胍",
                        "entity_type": "concept",
                        "description": "降糖药",
                        "file_path": "metformin.md",
                    }
                ],
                "relationships": [
                    {
                        "src_id": "二甲双胍",
                        "tgt_id": "2型糖尿病",
                        "description": "TREATS",
                    }
                ],
                "chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "content": "二甲双胍用于2型糖尿病。",
                        "file_path": "metformin.md",
                    }
                ],
                "references": [],
            },
        }


class FakeRetriever:
    def search_entities(self, query, entity_type=None, limit=20):
        return {"query": query, "count": 1, "nodes": [{"name": "二甲双胍"}]}

    def get_neighbors(
        self,
        entity,
        relation_type=None,
        direction="both",
        limit=20,
    ):
        return {
            "query": entity,
            "count": 1,
            "triples": [["二甲双胍", "TREATS", "2型糖尿病"]],
        }

    def get_path(self, source, target, max_hops=3):
        return {
            "query": f"{source}->{target}",
            "count": 1,
            "triples": [["高血压", "HAS_COMPLICATION", "脑卒中"]],
        }


class TestMedicalToolbox(unittest.IsolatedAsyncioTestCase):
    def make_toolbox(self):
        return MedicalToolbox(
            rag_engine=FakeRagEngine(),
            graph_retriever=FakeRetriever(),
        )

    async def test_retrieval_tool_is_compact_and_validated(self):
        toolbox = self.make_toolbox()
        result = await toolbox.call_tool(
            "search_medical_documents",
            {
                "query": "二甲双胍治疗什么？",
                "mode": "mix",
                "top_k": 999,
            },
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["counts"]["entities"], 1)
        self.assertEqual(result["data"]["counts"]["relationships"], 1)
        self.assertEqual(toolbox._rag_engine.calls[0][2], 100)

    async def test_graph_tools(self):
        toolbox = self.make_toolbox()
        entity = await toolbox.call_tool(
            "search_graph_entities",
            {"query": "二甲双胍", "limit": 10},
        )
        relations = await toolbox.call_tool(
            "get_entity_relations",
            {"entity": "二甲双胍"},
        )
        path = await toolbox.call_tool(
            "find_entity_path",
            {"source": "高血压", "target": "脑卒中"},
        )
        self.assertEqual(entity["data"]["count"], 1)
        self.assertEqual(relations["data"]["triples"][0][1], "TREATS")
        self.assertEqual(path["data"]["triples"][0][1], "HAS_COMPLICATION")

    async def test_supported_entities(self):
        toolbox = self.make_toolbox()
        result = await toolbox.call_tool(
            "list_supported_entities",
            {"entity_type": "disease", "limit": 3},
        )
        self.assertGreater(result["data"]["count"], 0)
        self.assertTrue(
            all(
                item["entity_type"] == "disease"
                for item in result["data"]["entities"]
            )
        )

    async def test_invalid_tool_and_arguments(self):
        toolbox = self.make_toolbox()
        with self.assertRaises(ToolExecutionError):
            await toolbox.call_tool("unknown_tool", {})
        with self.assertRaises(ToolExecutionError):
            await toolbox.call_tool(
                "search_medical_documents",
                {"query": "test", "cypher": "MATCH (n) RETURN n"},
            )


class TestMcpServer(unittest.IsolatedAsyncioTestCase):
    async def test_tools_are_registered(self):
        tools = await mcp.list_tools()
        names = {tool.name for tool in tools}
        expected = {item["name"] for item in TOOL_DEFINITIONS}
        self.assertEqual(names, expected)

    async def test_stdio_protocol_initialize_and_call(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.medqa_mcp"],
        )

        async def exercise():
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    self.assertIn(
                        "list_supported_entities",
                        {tool.name for tool in tools.tools},
                    )
                    result = await session.call_tool(
                        "list_supported_entities",
                        {"entity_type": "disease", "limit": 3},
                    )
                    self.assertFalse(result.isError)
                    self.assertTrue(result.content)

        await asyncio.wait_for(exercise(), timeout=20)


if __name__ == "__main__":
    unittest.main()
