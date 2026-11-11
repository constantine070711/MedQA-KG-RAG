# Changelog

All notable changes to MedQA are documented here.

## [Unreleased] - 2026-11-11

### Added

- Deterministic three-layer refusal and answer-safety analysis covering
  question intent, answer content, and final policy action.
- Fixed safe-response replacement for concrete dosage or definitive-diagnosis
  answers, plus fixed emergency-care responses.
- Versioned evaluation policy `2026-11-09-three-layer-v1` with dataset schema
  version `1.1`.
- Backend AI image recognition at `POST /api/vision/extract` using an
  OpenAI-compatible multimodal model, configured through `VISION_*` variables.
- Automatic browser fallback from AI image recognition to local Tesseract OCR.
- Image signature and 8 MB size validation, editable extraction output, and
  recoverable provider/response errors.
- Vision endpoint contract tests and CI coverage.

### Changed

- Safe restricted dosage and diagnosis answers now use `revise` with a medical
  disclaimer instead of unconditional refusal.
- Concrete dosage, dose frequency, or definitive diagnosis is replaced before
  display; blocked and out-of-knowledge-base cases still refuse.
- Removed legacy `safe_medication_answer` and runtime QID overrides. Expected
  refusal labels are now explicit in the dataset: 9 -> 6, so old M7 results are
  not directly comparable.
- README now records the competition team information for 西安工业大学 /
  知衡医匠 / A组 without restoring `TEAM.md`.
- Public documentation now excludes competition-only materials and v1-era
  screenshots; the v1 tag retains the historical screenshots.
- The image dialog now offers explicit AI vision and local OCR modes, with
  separate privacy copy.
- README, architecture, deployment, and image-recognition documentation now
  describe the hybrid workflow.

### Verified

- Refusal, reliability, Agent, evaluation, evidence-fusion, API, UI, deployment,
  and monitoring suites: 452 tests passed.
- The same dose question now consistently reaches `revise` for safe, unsafe, and
  terse answers; unsafe dosage text is not displayed.
- Backend vision/API/tooling/deployment tests: 25 passed.
- Python compilation, Web ESLint, and Next.js production build passed.
- Real-image DeepSeek multimodal compatibility and recognition quality are not
  yet verified.

## [2.0.0] - 2026-10-25

### Added

- Authentic development timeline covering v1 and v2 implementation.
- Public documentation index and synchronized v2 architecture, capabilities,
  testing counts, limitations, and future-work sections.
- Production API and standalone Next.js container images, three-service Compose
  orchestration, one-time Neo4j/LightRAG initializer, health checks, and
  deployment documentation.
- Next.js 16 + TypeScript evidence workspace in `web/`.
- FastAPI v2 service in `api/` with SSE workflow events.
- Streaming answer events, workflow-stage status, and cancellation.
- Evidence inspector for safety level, reliability, source coverage, citations,
  and Neo4j triples.
- Controlled medical tool layer with HTTP execution and browser tool traces.
- MCP stdio server exposing five read-only retrieval tools.
- Browser-native Mandarin voice input and answer playback with selectable
  Chinese voices, quality-aware automatic ranking, and persisted preferences.
- Browser-local Chinese/English OCR for drug boxes and report images.
- Responsive desktop and mobile layouts with a dedicated evidence drawer.
- Current 1920x1080 v2 workspace and evidence screenshots in the README.
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
