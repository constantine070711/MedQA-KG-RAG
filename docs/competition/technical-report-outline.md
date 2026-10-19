# Technical Report Outline

Target length: 20-30 pages.

## 1. Project Overview

- Problem statement.
- Target users and scenarios.
- Scope and non-goals.
- v1 to v2 evolution.

## 2. Requirements Analysis

- Medical knowledge question answering.
- Evidence traceability.
- Personal diagnosis and dosage boundaries.
- Multimodal input.
- Deployment and competition constraints.

## 3. System Architecture

- Next.js evidence workspace.
- FastAPI and SSE.
- LangGraph workflow.
- LightRAG and Neo4j evidence sources.
- Monitoring and CI.

## 4. AI Technical Solution

- Why deepseek-v4-flash was selected.
- Why LightRAG and Neo4j are both required.
- Evidence fusion strategy.
- Safety prompt and generation constraints.
- Reliability decision and refusal rules.
- MCP tools and tool trace.
- Voice and OCR integration.

## 5. Core Implementation

- API contracts and SSE events.
- State model and graph routing.
- Evidence panel and responsive layout.
- Data ingestion and provenance.
- Local OCR privacy boundary.
- Error and degradation behavior.

## 6. Evaluation and Testing

- 28-question retrieval set.
- M1/M2/M5/M7 and reliability metrics.
- Unit and contract tests.
- Real-environment validation.
- Known failure cases and limitations.

## 7. Deployment

- Environment variables.
- Neo4j startup.
- API and web startup.
- Health checks.
- Public deployment plan.

## 8. Innovation Summary

- Evidence-first interface.
- Reliability gate rather than answer-only chatbot.
- Visible tool execution trace.
- MCP integration.
- Local image OCR and voice interaction with explicit privacy boundaries.

## 9. Limitations and Future Work

- Curated graph vocabulary is small.
- Rules complement but do not replace clinical validation.
- LLM output remains non-deterministic.
- Larger authoritative knowledge bases need retrieval ablation experiments.
- Multi-user authentication and audit storage are future work.
