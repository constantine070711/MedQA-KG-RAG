# MedQA v2 Architecture

## Purpose

MedQA v2 keeps the verified v1 retrieval, safety, and reliability core while
replacing the presentation layer with an evidence-oriented browser application.

## Runtime Topology

```text
Next.js evidence workspace (web/)
        |
        | POST /api/chat/stream (SSE)
        v
FastAPI v2 gateway (api/)
        |
        +-- Controlled tool gateway (/api/tools)
        |
        v
MedQAAgent LangGraph workflow (src/agent.py)
        |
        +-- SafetyChecker
        +-- Neo4jGraphRetriever
        +-- LightRAG
        +-- Evidence Fusion
        +-- ReliabilityDecision

External MCP clients use the same MedicalToolbox through
`mcp_server/medqa_mcp.py`.

Voice input, answer playback, and image OCR run in the browser. Voice uses the
Web Speech API, including browser-provided Chinese voice discovery, persisted
voice selection, and quality-aware automatic ranking. OCR uses Tesseract.js with
`chi_sim+eng`. Images are not sent to the API unless the user explicitly places
recognized text into a chat request.
```

## API

### `GET /api/health`

Returns service name and API version. It does not initialize LightRAG or call an
LLM, so the web shell can render before the first medical query.

### `GET /api/meta`

Returns example questions and the medical disclaimer.

### `POST /api/chat/stream`

Accepts:

```json
{
  "question": "二甲双胍用于治疗什么疾病？",
  "mode": "mix",
  "history": [
    {
      "question": "前一个问题",
      "answer": "前一个回答",
      "safety_level": "safe",
      "is_final_refusal": false
    }
  ]
}
```

The endpoint emits Server-Sent Events with one of these payloads:

```json
{"type":"status","stage":"safety","label":"安全检查完成","detail":"正在检索结构化医疗知识"}
{"type":"answer_delta","delta":"二甲双胍主要用于"}
{"type":"final","request_id":"...","elapsed_ms":12345.6,"response":{}}
{"type":"error","message":"本次请求暂时无法完成，请稍后重试。"}
```

Heartbeat comments are emitted during long-running generation so reverse
proxies do not close an otherwise healthy stream.

### `GET /api/tools`

Returns the whitelisted tool schemas.

### `POST /api/tools/execute`

Runs one controlled retrieval tool. Tool names, argument names, limits, and
enums are validated before dispatch.

### Browser-local capabilities

- `use-speech.ts`: Mandarin recognition, Chinese voice discovery, persisted
  voice selection, and answer playback.
- `image-ocr-dialog.tsx`: local OCR, editable text, and supported-entity lookup.
- The OCR dialog uses the public `list_supported_entities` tool only for
  vocabulary matching; it does not send the source image.

## Browser Application

The workspace has three functional regions:

- Conversation: questions, streamed answers, medical disclaimer, cancellation.
- Workflow status: safety, retrieval, fusion, generation, and verification.
- Evidence inspector: safety decision, reliability, source coverage, evidence
  cards, graph triples, citations, retrieval counts, and tool traces.

The evidence inspector consumes only the final response returned by the existing
agent. It does not infer citations, fabricate graph relationships, or execute a
second retrieval.

## Safety Boundary

The v2 API enables `enforce_safe_generation_prompt=True`. This keeps v1 behavior
available through the v1 tag while ensuring v2 safe queries use an explicit
medical safety prompt. The prompt prohibits unsolicited dosage values,
administration frequency, and personalized medication plans, even when those
details appear in retrieved context.

## Local Development

Backend:

```bash
venv\Scripts\activate
uvicorn api.main:app --reload --port 8000
```

Frontend:

```bash
cd web
npm install
npm run dev
```

The default frontend API base URL is `http://localhost:8000`. Override it with
`NEXT_PUBLIC_MEDQA_API_URL` when deploying.
