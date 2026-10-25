# Scoring Evidence Matrix

This is an internal self-evaluation based on the published AI WEB track rubric.
It is not an official score.

## AI Technical Depth: 35%

| Criterion | Evidence | Status |
|---|---|---|
| LLM integration quality | DeepSeek API integration, safety prompts, bounded generation, refusal handling | Strong |
| Agent/workflow design | Seven-node LangGraph state machine, deterministic routing, explicit tool traces | Strong; planning is workflow-driven rather than an open-ended planner |
| RAG application | LightRAG vector + graph retrieval, Neo4j evidence fusion, provenance, citations | Strong |
| Multimodal capability | Browser Mandarin voice input/playback and local Chinese/English OCR | Implemented |
| AI safety and controllability | Four-level safety, diagnosis/dosage refusal, uncertainty signals, reliability gate, blocked generation short-circuit | Strong |
| MCP/tool integration | Five read-only tools, real MCP stdio server, HTTP gateway, browser trace | Implemented |

## Innovation: 25%

| Criterion | Evidence | Status |
|---|---|---|
| Topic innovation | Evidence-first medical knowledge assistant with a reliability gate rather than a generic chatbot | Moderate |
| Technical novelty | Dual-source evidence fusion, trust decision layer, MCP tools, local OCR privacy boundary | Moderate to strong |
| Interaction novelty | Evidence inspector, live tool trace, source and graph exploration, voice and image input | Strong |

## Front-end Engineering: 10%

| Criterion | Evidence | Status |
|---|---|---|
| Code quality | Next.js App Router, TypeScript strict mode, component separation, typed SSE contracts | Strong |
| Component architecture | Workspace, workflow status, evidence panel, graph view, voice hooks, OCR dialog | Strong |
| Performance | Static shell, lazy OCR worker, responsive layout, no model load before chat | Good |

## Practicality: 10%

- Real medical knowledge retrieval rather than generated placeholder answers.
- Explicit distinction between general knowledge and personal diagnosis.
- Neo4j degradation is visible and never replaced with fabricated evidence.
- Local OCR avoids uploading source images.

## User Experience: 10%

- Streaming stage status and answer text.
- Evidence available beside the conversation rather than buried in logs.
- Voice and image actions use familiar icon controls with accessible labels.
- Responsive layout with a mobile evidence drawer.
- Remaining risk: first LightRAG query and first OCR model download can be slow.

## Social Value and Commercial Potential: 10%

- Supports medical knowledge accessibility and source verification.
- Can be adapted for patient-education portals, medical libraries, or enterprise
  knowledge management.
- Compliance, medical liability, and authoritative data licensing remain the
  main commercial constraints.

## Remaining Evidence to Produce

- Public deployment URL and uninterrupted review-period availability.
- 5-8 minute video with Mandarin narration.
- 30-page technical report PDF.
- Team, school, group, and advisor information.
- Final submission tag and release notes.
