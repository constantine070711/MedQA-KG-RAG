# Documentation Index

## Start Here

- [README](../README.md): project overview, setup, runtime commands, and
  evaluation summary.
- [Architecture](../ARCHITECTURE.md): inherited v1 core plus v2 presentation and
  tool layers.
- [Development history](development-history.md): implementation timeline from
  v1 through the current v2 application.
- [v2 Architecture](v2-architecture.md): FastAPI, SSE, MCP, voice, and OCR
  details.

## v2 Runtime

- [MCP tools](v2-mcp-tools.md)
- [Voice interaction](v2-voice.md)
- [Image OCR](v2-report-ocr.md)
- [Deployment](deployment.md)
- [Windows Docker troubleshooting](windows-docker-troubleshooting.md)

## Data and Evaluation

- [Data sources](../data/SOURCES.md)
- Evaluation outputs: `data/eval/`
- API contract tests: `tests/test_api_v2.py`
- Tool and MCP tests: `tests/test_v2_tools.py`

## Internal Development Records

`PROJECT_STATUS.md`, `PROJECT_HANDOFF.md`, `PROJECT_LOG.md`, and `DECISIONS.md`
are local development records and are intentionally excluded from the public
repository. Public-facing summaries live under `docs/`.
