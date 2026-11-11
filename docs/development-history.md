# MedQA Development History

This timeline is a public, condensed record of work retained in the project
development logs, phase reports, evaluation artifacts, and later Git history.
It is not a retroactive reconstruction of Git commit timestamps.

Early implementation was logged in the team's phase reports before the public
repository was initialized. Public Git history starts with the v1.0.0 release
on 2026-09-01; later v2 work is preserved through branches, pull-request-style
merges, CI runs, and release tags.

## 2026-07-24: Environment and RAG MVP

- Defined the medical knowledge assistant scope and safety boundary.
- Selected LightRAG 1.5.4, DeepSeek API, local bge embeddings, and a
  file-backed vector store.
- Built the first document-to-vector-to-answer loop.
- Recorded the decision to use a deterministic safety layer instead of relying
  only on model instructions.

## 2026-07-25: Modular Core and Knowledge Base

- Refactored the prototype into `src/` modules.
- Added medical entities, relations, safety classification, and document
  metadata.
- Standardized disease and drug Markdown records.
- Added the first validation and regression scripts.

## 2026-07-26: Data Pipeline and Retrieval Evaluation

- Built the Adapter -> Transformer -> Validator -> Writer knowledge pipeline.
- Added metadata provenance and a vector-store abstraction.
- Added a 28-question evaluation set and initial M1-M7 metrics.
- Decided not to introduce ChromaDB after LightRAG 1.5.4 storage changes.

## 2026-07-27 to 2026-08-13: Reliability and Safety Metrics

- Added retrieval entity/chunk metrics, faithfulness judging, citation
  extraction, and citation grounding checks.
- Added deterministic uncertainty signals and reliability rules.
- Introduced `allow / revise / refuse` terminal decisions.
- Added refusal evaluation and adversarial out-of-knowledge-base questions.
- Reached the first stable safety and reliability baseline.

## 2026-08-16: Neo4j Knowledge Graph

- Added a Neo4j Community Docker service and idempotent schema bootstrap.
- Imported the LightRAG graph layer and curated disease/drug layer.
- Implemented parameterized, read-only Cypher retrieval with bounded paths.
- Added evidence fusion across LightRAG and Neo4j.
- Verified structured TREATS and HAS_COMPLICATION evidence.

## 2026-08-17: LangGraph Agent and Streamlit UI

- Added the seven-node `MedQAAgent` LangGraph workflow.
- Enforced zero generation on blocked inputs and one generation on allowed
  inputs.
- Added the Streamlit chat interface, evidence display, multi-turn history, and
  pyvis graph visualization.
- Completed regression coverage for safety, retrieval, graph degradation, and
  UI behavior.

## 2026-08-22: Multi-turn Memory

- Audited checkpointer-based memory and rejected it due to cross-turn state
  contamination risk.
- Implemented a bounded conversation context that cannot enter retrieval,
  evidence, citations, or the reliability gate.
- Verified pronoun resolution while preserving blocked and refusal behavior.

## 2026-08-25 to 2026-08-26: Final v1 Validation and Observability

- Completed a real-environment acceptance run with Neo4j, Streamlit, actual
  LLM calls, browser checks, and the full regression suite.
- Added the v1 knowledge-graph visualization fixes and multi-turn UI polish.
- Added structured monitoring with request IDs, JSONL logs, redaction, metrics,
  and rotation.
- Evaluated CBLUE adaptation and reranker experiments; both remained isolated
  because measured production benefit was insufficient.

## 2026-09-01: Public v1 Release

- Published the Streamlit application, architecture documentation, evaluation
  code, tests, data-source records, and MIT license.
- Recorded the v1 source state as tag `v1.0.0`.
- Preserved the known limitations of the rule-based safety layer and small
  curated graph instead of presenting them as production guarantees.

## 2026-10-19: v2 Application Release

- Added FastAPI SSE streaming and a Next.js 16 + TypeScript evidence workspace.
- Added workflow-stage events, evidence inspection, responsive desktop/mobile
  layouts, and a stricter v2 generation policy for medication details.
- Added five read-only MCP tools and a real stdio protocol test.
- Added browser tool traces, Mandarin voice input, answer playback, and local
  Chinese/English image OCR.
- Added GitHub Actions for API contract tests and frontend lint/build.

## 2026-11-03: Backend AI Vision with Local OCR Fallback

- Added `POST /api/vision/extract` for validated image upload and
  OpenAI-compatible multimodal extraction.
- Added explicit AI vision and local OCR modes in the image dialog, with
  automatic fallback when the vision request fails.
- Added `VISION_*` configuration while reusing DeepSeek credentials by default.
- Added vision endpoint contract tests and included them in v2 CI.
- Updated README, architecture, OCR, and deployment documentation.
- Real-image provider compatibility and recognition quality remain to be tested
  by the user with representative medical images.

## 2026-11-04: Competition Submission Metadata

- Added the public team information section to the README: 西安工业大学,
  知衡医匠, A组.
- Kept team-member assignments out of the README placeholder flow and aligned
  them with the official competition registration materials.
- Did not restore the removed `TEAM.md` file.
- Removed competition-only documents and v1-era screenshots from the current v2
  documentation tree; the v1 tag retains the historical screenshots.
- Kept internal development records local and made `docs/README.md` point only
  to public technical, runtime, and evaluation documentation.

## 2026-11-11: Three-Layer Answer-Safety Policy

- Replaced the former keyword-only refusal behavior with a deterministic
  three-layer policy: question intent, answer-content safety, and final action.
- Safe restricted answers can now be returned with a disclaimer; concrete
  dosage or definitive-diagnosis content is replaced with a fixed safety
  response, and emergency questions use a fixed urgent-care response.
- Removed the legacy `safe_medication_answer` bypass and runtime QID overrides.
- Versioned the evaluation dataset policy as
  `2026-11-09-three-layer-v1`; the six expected refusals are now explicit in the
  dataset.
- Verified 452 tests, local API/Web health, and Neo4j graph access.

## Public Evidence

- Git tag: `v1.0.0`
- Active v2 branch: `v2`
- CI workflow: `.github/workflows/v2-ci.yml`
- Architecture: [ARCHITECTURE.md](../ARCHITECTURE.md)
- Current v2 design: [v2-architecture.md](v2-architecture.md)
- Evaluation outputs: `data/eval/`
