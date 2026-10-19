# 5-8 Minute Demo Script

## 0:00-0:40 Opening

State the problem in one sentence:

> General medical chatbots often answer fluently without showing where a claim
> came from or whether the system had enough evidence.

Introduce MedQA v2 as an evidence-grounded medical knowledge assistant, not a
diagnosis system.

## 0:40-1:30 Architecture

Show the flow:

```text
Next.js evidence workspace
  -> FastAPI SSE
  -> LangGraph safety and evidence workflow
  -> LightRAG + Neo4j
  -> reliability gate and evidence response
```

Highlight that the graph and vector evidence are fused before generation and
that blocked questions skip both retrieval and generation.

## 1:30-2:40 Normal Knowledge Question

Ask:

> 二甲双胍用于治疗什么疾病？

Show:

- streaming workflow stages;
- answer streaming;
- safety and reliability badges;
- LightRAG evidence and citations;
- tool trace with security, graph, retrieval, and reliability steps.

Explain that the v2 prompt prevents unsolicited dosage details.

## 2:40-3:30 Safety Boundary

Ask:

> 二甲双胍一天吃几次，一次吃几片？

Show the dosage refusal and point out that the answer does not provide a
personal medication plan.

## 3:30-4:30 MCP and Tool Evidence

Open the evidence inspector and explain the tool trace.

Then show:

```text
mcp_server/medqa_mcp.py
```

Explain the five read-only tools and that the MCP stdio server is exercised in
automated tests.

## 4:30-5:20 Voice Interaction

- Use the microphone control to dictate a short non-sensitive question.
- Show the recognized text in the composer before sending.
- After an answer, use the playback control.

State that voice recognition may use the browser vendor's service and that the
text-only workflow remains available when speech APIs are unsupported.

## 5:20-6:20 Image OCR

Open the image action and show a drug-box or report image.

- OCR runs locally in the browser.
- Recognized text is editable.
- The system highlights supported disease/drug names.
- Click a supported entity to ask a general knowledge question.

Explicitly state that the system does not interpret personal lab values or
diagnose from the image.

## 6:20-7:10 Engineering Evidence

Show:

- `v1.0.0` tag and v2 branch graph;
- GitHub Actions with API and Web jobs passing;
- test suite and evaluation summary;
- MCP protocol test;
- v1 and v2 architecture separation.

## 7:10-7:50 Close

Summarize:

- hybrid knowledge retrieval;
- visible evidence and tool traces;
- safety and refusal boundaries;
- multimodal input;
- reproducible engineering and CI.

Mention the limitations: curated medical vocabulary, rule-based safety
fallbacks, and non-deterministic LLM output.
