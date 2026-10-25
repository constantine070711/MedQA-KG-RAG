# MedQA v2 MCP Tools

MedQA exposes its retrieval layer through a real Model Context Protocol server
and through controlled HTTP tool endpoints. Both use the same whitelisted
`MedicalToolbox`; neither accepts user-authored Cypher.

## Tools

| Tool | Backend | Purpose |
|---|---|---|
| `search_medical_documents` | LightRAG | Hybrid document/entity/relation retrieval without an extra LLM call |
| `search_graph_entities` | Neo4j | Find disease or drug entities by name or alias |
| `get_entity_relations` | Neo4j | Retrieve one-hop structured relations |
| `find_entity_path` | Neo4j | Bounded shortest path between two entities |
| `list_supported_entities` | Curated entities JSON | List diseases and drugs available in the graph |

All tools validate argument names, lengths, enums, and numeric bounds. Graph
queries remain parameterized and read-only.

## Run the MCP Server

```powershell
venv\Scripts\activate
python -m mcp_server.medqa_mcp
```

The server communicates over stdio. A typical MCP client configuration is:

```json
{
  "mcpServers": {
    "medqa": {
      "command": "<absolute-path-to-repo>\\venv\\Scripts\\python.exe",
      "args": ["-m", "mcp_server.medqa_mcp"],
      "cwd": "<absolute-path-to-repo>"
    }
  }
}
```

On macOS or Linux, replace the Python executable with
`<absolute-path-to-repo>/venv/bin/python`.

## HTTP Tool Gateway

List tool schemas:

```http
GET /api/tools
```

Execute a tool:

```http
POST /api/tools/execute
Content-Type: application/json

{
  "tool": "find_entity_path",
  "arguments": {
    "source": "高血压",
    "target": "脑卒中",
    "max_hops": 3
  }
}
```

## Browser Trace

The SSE chat endpoint emits `tool` events alongside workflow status and answer
chunks. The evidence inspector displays these calls, their status, latency, and
a short result summary. Tool events are observations of actual retrieval work,
not fabricated model reasoning traces.

## Safety

Tools return medical knowledge evidence only. They do not diagnose a person,
prescribe treatment, or expose concrete dosage instructions. All outputs include
the medical knowledge disclaimer.
