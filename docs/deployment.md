# MedQA v2 Deployment

The v2 stack runs three long-lived services:

- `neo4j`: structured medical knowledge graph.
- `api`: FastAPI, SSE, LangGraph, LightRAG, MCP tools.
- `web`: Next.js standalone evidence workspace.

`neo4j-init` is a one-time profile for schema creation and knowledge indexing.

## Prerequisites

- Docker Desktop with the Linux engine.
- A DeepSeek API key.
- Local embedding model under `models/bge-small-zh-v1.5/`, or network access for
  Hugging Face download during indexing.
- At least several GB of free disk space for PyTorch, the model, and the Neo4j
  volume.

## Configure

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```env
DEEPSEEK_API_KEY=...
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
```

The Compose stack uses these host ports by default:

```text
3000  Web
8000  API
7474  Neo4j Browser
7687  Neo4j Bolt
```

Override `WEB_PORT`, `API_PORT`, `NEO4J_HTTP_PORT`, or `NEO4J_BOLT_PORT` if a
port is occupied.

## First Initialization

The first run needs an LLM-backed LightRAG index and a Neo4j import:

```powershell
docker compose --profile init run --rm neo4j-init
```

This service runs:

```text
neo4j_bootstrap.py
index_docs.py --clear
import_graphml_to_neo4j.py
import_curated_to_neo4j.py
rebuild_curated_relations.py
```

The indexing step can take several minutes and consumes LLM API tokens. It
writes to `lightrag_cache/`. The Neo4j graph is stored in the Docker volume
`medqa-neo4j-data`.

## Start

```powershell
docker compose up -d --build api web
docker compose ps
```

Health checks:

```powershell
Invoke-WebRequest http://localhost:8000/api/health
Invoke-WebRequest http://localhost:3000
```

Open `http://localhost:3000`.

## Everyday Commands

```powershell
# Start existing services
docker compose up -d

# Rebuild after code changes
docker compose up -d --build

# Logs
docker compose logs -f api web

# Stop without deleting data
docker compose down
```

Do not add `-v` unless you intentionally want to delete the Neo4j data volume.

## Update the Index

When tracked medical documents change:

```powershell
docker compose --profile init run --rm neo4j-init
```

The initializer clears and rebuilds the LightRAG index. Review LLM usage and
data licensing before running it against a large corpus.

## Public Deployment

`NEXT_PUBLIC_MEDQA_API_URL` is compiled into the frontend at image-build time.
Set it to a browser-reachable API URL before building:

```powershell
$env:NEXT_PUBLIC_MEDQA_API_URL="https://api.example.com"
docker compose build web
```

Also set `APP_ORIGINS` to the public frontend origin:

```env
APP_ORIGINS=https://demo.example.com
```

Place HTTPS termination in front of the API and web services. Do not expose
Neo4j ports publicly.

## Troubleshooting

### Docker daemon is unavailable

Start Docker Desktop and confirm:

```powershell
docker info
```

### Docker Desktop fails while creating a runtime socket on Windows

Errors mentioning `Docker\run\sailor-ingest.sock`,
`docker-secrets-engine\engine.sock`, or inaccessible reparse points are runtime
socket failures. They do not mean the container data disk is corrupted.

Follow [Windows Docker troubleshooting](windows-docker-troubleshooting.md).
Do not choose **Reset to factory defaults** as the first step because it deletes
containers, images, and volumes.

### Windows port reservation

Inspect reserved ranges:

```powershell
netsh interface ipv4 show excludedportrange protocol=tcp
```

Then override the affected service port in `.env`.

### API starts but graph evidence is unavailable

Confirm the one-time initializer completed and the Neo4j volume exists:

```powershell
docker compose logs neo4j-init
docker compose exec neo4j cypher-shell -u neo4j -p $env:NEO4J_PASSWORD "MATCH (n) RETURN count(n)"
```

The API intentionally degrades to LightRAG when Neo4j is unavailable and will
report that state rather than fabricating graph evidence.

After Docker or its data disk is restarted, restart the FastAPI process before
testing graph retrieval. Existing Neo4j driver connections may have been
invalidated and can log `ConnectionAbortedError(10053)` until the API process is
recreated.

### First query is slow

LightRAG initializes local embedding and retrieval workers on the first query.
The UI shows workflow stages while this work is in progress.

## Security

- Never commit `.env`.
- Never bake the API key into a Docker image.
- Do not publish ports `7474` or `7687` in production.
- Use a dedicated Neo4j password and a public-origin allowlist.
- Keep the demo environment available while it is being evaluated.
