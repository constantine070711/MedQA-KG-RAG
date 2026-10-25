"""Controlled medical retrieval tools for API and MCP clients."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.config import PROJECT_ROOT

if TYPE_CHECKING:
    from src.neo4j_graph import Neo4jGraphRetriever
    from src.rag_engine import RAGEngine


ALLOWED_MODES = {"naive", "local", "global", "hybrid", "mix"}
DEFAULT_ENTITY_PATH = PROJECT_ROOT / "data" / "entities" / "entities.json"

_DISCLAIMER = (
    "本工具仅返回医疗知识证据，不提供个人诊断、处方或具体用药剂量。"
)

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_medical_documents",
        "description": (
            "使用 LightRAG 在图与向量知识库中执行纯检索，返回实体、关系、"
            "文本片段和来源信息，不调用生成模型。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "maxLength": 500},
                "mode": {
                    "type": "string",
                    "enum": sorted(ALLOWED_MODES),
                    "default": "mix",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 60,
                },
                "chunk_top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 40,
                    "default": 20,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_graph_entities",
        "description": "在 Neo4j 医疗知识图谱中按名称或别名搜索实体。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "maxLength": 200},
                "entity_type": {"type": "string", "maxLength": 50},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 20,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_entity_relations",
        "description": "查询医疗实体的一跳邻居和结构化关系三元组。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "maxLength": 200},
                "relation_type": {"type": "string", "maxLength": 50},
                "direction": {
                    "type": "string",
                    "enum": ["outgoing", "incoming", "both"],
                    "default": "both",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 20,
                },
            },
            "required": ["entity"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_entity_path",
        "description": "在受控跳数内查询两个医疗实体之间的最短图路径。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "maxLength": 200},
                "target": {"type": "string", "maxLength": 200},
                "max_hops": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 6,
                    "default": 3,
                },
            },
            "required": ["source", "target"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_supported_entities",
        "description": "列出当前 curated 医疗知识图谱支持的疾病和药物实体。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_type": {
                    "type": "string",
                    "enum": ["disease", "drug"],
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 50,
                },
            },
            "additionalProperties": False,
        },
    },
]

_TOOL_INDEX = {item["name"]: item for item in TOOL_DEFINITIONS}


class ToolExecutionError(ValueError):
    """Raised for invalid tool names or arguments."""


def _string_arg(
    arguments: dict[str, Any],
    name: str,
    *,
    required: bool = False,
    max_length: int = 500,
) -> str | None:
    value = arguments.get(name)
    if value is None:
        if required:
            raise ToolExecutionError(f"missing required argument: {name}")
        return None
    if not isinstance(value, str):
        raise ToolExecutionError(f"{name} must be a string")
    cleaned = value.strip()
    if required and not cleaned:
        raise ToolExecutionError(f"{name} must not be blank")
    if len(cleaned) > max_length:
        raise ToolExecutionError(f"{name} exceeds {max_length} characters")
    return cleaned or None


def _int_arg(
    arguments: dict[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = arguments.get(name, default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolExecutionError(f"{name} must be an integer") from exc
    return max(minimum, min(parsed, maximum))


def _validate_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    spec = _TOOL_INDEX.get(name)
    if spec is None:
        raise ToolExecutionError(f"unknown tool: {name}")
    if not isinstance(arguments, dict):
        raise ToolExecutionError("arguments must be an object")
    allowed = set((spec.get("inputSchema") or {}).get("properties", {}))
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        raise ToolExecutionError(f"unknown arguments: {', '.join(unknown)}")
    return {key: value for key, value in arguments.items() if value is not None}


def _compact_retrieval(result: dict[str, Any]) -> dict[str, Any]:
    data = result.get("data") if isinstance(result, dict) else {}
    data = data if isinstance(data, dict) else {}
    entities = data.get("entities") if isinstance(data.get("entities"), list) else []
    relationships = (
        data.get("relationships")
        if isinstance(data.get("relationships"), list)
        else []
    )
    chunks = data.get("chunks") if isinstance(data.get("chunks"), list) else []
    references = (
        data.get("references") if isinstance(data.get("references"), list) else []
    )

    return {
        "status": result.get("status", "unknown"),
        "entities": [
            {
                "name": item.get("entity_name") or item.get("name"),
                "type": item.get("entity_type"),
                "description": str(item.get("description") or "")[:300],
                "file_path": item.get("file_path"),
            }
            for item in entities[:20]
            if isinstance(item, dict)
        ],
        "relationships": [
            {
                "source": item.get("src_id") or item.get("source"),
                "target": item.get("tgt_id") or item.get("target"),
                "description": str(item.get("description") or "")[:300],
                "keywords": item.get("keywords"),
            }
            for item in relationships[:20]
            if isinstance(item, dict)
        ],
        "chunks": [
            {
                "chunk_id": item.get("chunk_id") or item.get("id"),
                "content": str(item.get("content") or "")[:500],
                "file_path": item.get("file_path"),
            }
            for item in chunks[:10]
            if isinstance(item, dict)
        ],
        "references": references[:20],
        "counts": {
            "entities": len(entities),
            "relationships": len(relationships),
            "chunks": len(chunks),
        },
    }


class MedicalToolbox:
    """Whitelisted tools backed by the existing retrieval components."""

    def __init__(
        self,
        rag_engine: Any | None = None,
        graph_retriever: Any | None = None,
        entities_path: str | Path | None = None,
    ) -> None:
        self._rag_engine = rag_engine
        self._graph_retriever = graph_retriever
        self._entities_path = Path(entities_path or DEFAULT_ENTITY_PATH)
        self._engine_lock = asyncio.Lock()

    async def _get_rag_engine(self) -> Any:
        async with self._engine_lock:
            if self._rag_engine is None:
                from src.rag_engine import RAGEngine

                self._rag_engine = RAGEngine()
                await self._rag_engine.initialize()
            return self._rag_engine

    def _get_graph_retriever(self) -> Any:
        if self._graph_retriever is None:
            from src.neo4j_graph import Neo4jGraphRetriever

            self._graph_retriever = Neo4jGraphRetriever(
                entities_path=self._entities_path
            )
        return self._graph_retriever

    async def search_medical_documents(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 20,
    ) -> dict[str, Any]:
        query = _string_arg({"query": query}, "query", required=True) or ""
        if mode not in ALLOWED_MODES:
            raise ToolExecutionError(f"invalid mode: {mode}")
        engine = await self._get_rag_engine()
        result = await engine.retrieve(
            query,
            mode=mode,
            top_k=max(1, min(int(top_k), 100)),
            chunk_top_k=max(1, min(int(chunk_top_k), 40)),
        )
        return _compact_retrieval(result or {})

    async def search_graph_entities(
        self,
        query: str,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        query = _string_arg({"query": query}, "query", required=True, max_length=200) or ""
        return self._get_graph_retriever().search_entities(
            query,
            entity_type=entity_type,
            limit=max(1, min(int(limit), 50)),
        )

    async def get_entity_relations(
        self,
        entity: str,
        relation_type: str | None = None,
        direction: str = "both",
        limit: int = 20,
    ) -> dict[str, Any]:
        entity = _string_arg(
            {"entity": entity}, "entity", required=True, max_length=200
        ) or ""
        return self._get_graph_retriever().get_neighbors(
            entity,
            relation_type=relation_type,
            direction=direction,
            limit=max(1, min(int(limit), 50)),
        )

    async def find_entity_path(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
    ) -> dict[str, Any]:
        source = _string_arg(
            {"source": source}, "source", required=True, max_length=200
        ) or ""
        target = _string_arg(
            {"target": target}, "target", required=True, max_length=200
        ) or ""
        return self._get_graph_retriever().get_path(
            source,
            target,
            max_hops=max(1, min(int(max_hops), 6)),
        )

    async def list_supported_entities(
        self,
        entity_type: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        payload = json.loads(self._entities_path.read_text(encoding="utf-8"))
        entities = payload.get("entities") or []
        if entity_type:
            entities = [
                item
                for item in entities
                if isinstance(item, dict)
                and item.get("entity_type") == entity_type
            ]
        limited = entities[: max(1, min(int(limit), 100))]
        return {
            "count": len(entities),
            "entities": [
                {
                    "name": item.get("name"),
                    "entity_type": item.get("entity_type"),
                    "aliases": [
                        alias.get("name")
                        for alias in item.get("aliases") or []
                        if isinstance(alias, dict)
                    ],
                }
                for item in limited
                if isinstance(item, dict)
            ],
        }

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        args = _validate_arguments(name, arguments or {})

        if name == "search_medical_documents":
            query = _string_arg(args, "query", required=True)
            mode = str(args.get("mode") or "mix")
            top_k = _int_arg(args, "top_k", default=60, minimum=1, maximum=100)
            chunk_top_k = _int_arg(
                args, "chunk_top_k", default=20, minimum=1, maximum=40
            )
            data = await self.search_medical_documents(
                query or "",
                mode=mode,
                top_k=top_k,
                chunk_top_k=chunk_top_k,
            )
        elif name == "search_graph_entities":
            query = _string_arg(
                args, "query", required=True, max_length=200
            )
            data = await self.search_graph_entities(
                query or "",
                entity_type=args.get("entity_type"),
                limit=_int_arg(
                    args, "limit", default=20, minimum=1, maximum=50
                ),
            )
        elif name == "get_entity_relations":
            entity = _string_arg(
                args, "entity", required=True, max_length=200
            )
            data = await self.get_entity_relations(
                entity or "",
                relation_type=args.get("relation_type"),
                direction=str(args.get("direction") or "both"),
                limit=_int_arg(
                    args, "limit", default=20, minimum=1, maximum=50
                ),
            )
        elif name == "find_entity_path":
            source = _string_arg(
                args, "source", required=True, max_length=200
            )
            target = _string_arg(
                args, "target", required=True, max_length=200
            )
            data = await self.find_entity_path(
                source or "",
                target or "",
                max_hops=_int_arg(
                    args, "max_hops", default=3, minimum=1, maximum=6
                ),
            )
        elif name == "list_supported_entities":
            data = await self.list_supported_entities(
                entity_type=args.get("entity_type"),
                limit=_int_arg(
                    args, "limit", default=50, minimum=1, maximum=100
                ),
            )
        else:
            raise ToolExecutionError(f"unknown tool: {name}")

        return {
            "tool": name,
            "status": "success",
            "arguments": args,
            "data": data,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "disclaimer": _DISCLAIMER,
        }
