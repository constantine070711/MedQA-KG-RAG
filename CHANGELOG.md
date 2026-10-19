# Changelog

All notable changes to MedQA are documented here.

## [Unreleased] - v2.0.0-dev

### Added

- Next.js 16 + TypeScript evidence workspace in `web/`.
- FastAPI v2 service in `api/` with SSE workflow events.
- Streaming answer events, workflow-stage status, and cancellation.
- Evidence inspector for safety level, reliability, source coverage, citations,
  and Neo4j triples.
- Controlled medical tool layer with HTTP execution and browser tool traces.
- MCP stdio server exposing five read-only retrieval tools.
- Responsive desktop and mobile layouts with a dedicated evidence drawer.
- v2 safe-generation mode that prevents unsolicited dosage details.

### Changed

- `requirements.txt` now includes FastAPI and Uvicorn.
- Public README clone and startup instructions target the real repository.

### Preserved

- v1.0.0 remains available through the matching Git tag.
- The original Streamlit interface remains in `app/` for reference.

## [1.0.0] - 2026-09-01

### Added

- Initial public release of the MedQA KG + RAG system.
- LightRAG hybrid retrieval, Neo4j graph evidence, LangGraph workflow.
- Safety classification, reliability decisions, citations, and monitoring.
- Streamlit interface and knowledge-graph visualization.
