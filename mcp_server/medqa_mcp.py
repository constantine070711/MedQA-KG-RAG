"""MedQA medical knowledge tools exposed through the Model Context Protocol."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from src.tools.medical_tools import MedicalToolbox


mcp = FastMCP(
    "MedQA Medical Knowledge",
    instructions=(
        "Use these tools only for medical knowledge retrieval. "
        "Never use them to diagnose an individual or prescribe treatment."
    ),
)
toolbox = MedicalToolbox()


@mcp.tool()
async def search_medical_documents(
    query: str,
    mode: str = "mix",
    top_k: int = 60,
    chunk_top_k: int = 20,
) -> dict:
    """Search the hybrid LightRAG medical knowledge index without generation."""

    return await toolbox.call_tool(
        "search_medical_documents",
        {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
        },
    )


@mcp.tool()
async def search_graph_entities(
    query: str,
    entity_type: str | None = None,
    limit: int = 20,
) -> dict:
    """Find disease or drug entities in the Neo4j medical knowledge graph."""

    return await toolbox.call_tool(
        "search_graph_entities",
        {
            "query": query,
            "entity_type": entity_type,
            "limit": limit,
        },
    )


@mcp.tool()
async def get_entity_relations(
    entity: str,
    relation_type: str | None = None,
    direction: str = "both",
    limit: int = 20,
) -> dict:
    """Retrieve one-hop medical relations for a graph entity."""

    return await toolbox.call_tool(
        "get_entity_relations",
        {
            "entity": entity,
            "relation_type": relation_type,
            "direction": direction,
            "limit": limit,
        },
    )


@mcp.tool()
async def find_entity_path(
    source: str,
    target: str,
    max_hops: int = 3,
) -> dict:
    """Find a bounded shortest path between two medical entities."""

    return await toolbox.call_tool(
        "find_entity_path",
        {
            "source": source,
            "target": target,
            "max_hops": max_hops,
        },
    )


@mcp.tool()
async def list_supported_entities(
    entity_type: str | None = None,
    limit: int = 50,
) -> dict:
    """List the disease and drug entities covered by the curated graph."""

    return await toolbox.call_tool(
        "list_supported_entities",
        {
            "entity_type": entity_type,
            "limit": limit,
        },
    )


def main() -> None:
    """Run the MCP server over stdio."""

    mcp.run()


if __name__ == "__main__":
    main()
