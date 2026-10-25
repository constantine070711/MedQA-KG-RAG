"""FastAPI entry point for the MedQA v2 browser application."""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.runtime import AgentRuntime
from api.schemas import ChatRequest, ToolCallRequest
from api.toolbox import get_toolbox
from src.tools.medical_tools import TOOL_DEFINITIONS, ToolExecutionError


API_VERSION = "2.0.0"
EXAMPLES = [
    "2型糖尿病的常见症状有哪些？",
    "二甲双胍用于治疗什么疾病？",
    "高血压有哪些常见并发症？",
    "高血压和脑卒中有什么关系？",
]

app = FastAPI(
    title="MedQA v2 API",
    version=API_VERSION,
    description="Evidence-grounded medical knowledge assistant API.",
)

_origins = [
    origin.strip()
    for origin in os.getenv(
        "APP_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

_runtime: AgentRuntime | None = None


def get_runtime() -> AgentRuntime:
    global _runtime
    if _runtime is None:
        _runtime = AgentRuntime()
    return _runtime


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": API_VERSION,
        "service": "medqa-v2-api",
    }


@app.get("/api/meta")
async def meta() -> dict:
    return {
        "version": API_VERSION,
        "examples": EXAMPLES,
        "tool_count": len(TOOL_DEFINITIONS),
        "disclaimer": (
            "本系统仅提供医疗知识辅助，不替代医生诊断、处方或专业医疗意见。"
            "如存在紧急情况，请及时寻求专业医疗帮助（紧急情况请拨打 120）。"
        ),
    }


@app.get("/api/tools")
async def tools() -> dict:
    return {
        "version": API_VERSION,
        "tools": TOOL_DEFINITIONS,
    }


@app.post("/api/tools/execute")
async def execute_tool(
    request: ToolCallRequest,
    toolbox=Depends(get_toolbox),
) -> dict:
    try:
        return await toolbox.call_tool(request.tool, request.arguments)
    except ToolExecutionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"tool service unavailable: {type(exc).__name__}",
        ) from exc


@app.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    runtime: AgentRuntime = Depends(get_runtime),
) -> StreamingResponse:
    return StreamingResponse(
        runtime.stream_with_heartbeat(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
