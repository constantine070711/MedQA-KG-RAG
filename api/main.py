"""FastAPI entry point for the MedQA v2 browser application."""

from __future__ import annotations

import logging
import os

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.runtime import AgentRuntime
from api.schemas import ChatRequest, ToolCallRequest, VisionExtractResponse
from api.toolbox import get_toolbox
from src.config import get_config
from src.tools.medical_tools import TOOL_DEFINITIONS, ToolExecutionError
from src.vision_client import (
    VisionClient,
    VisionInvalidImageError,
    VisionNotConfiguredError,
    VisionProviderError,
    VisionResponseError,
    get_vision_client,
)


API_VERSION = "2.0.0"
logger = logging.getLogger(__name__)
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


def detect_image_mime(data: bytes) -> str | None:
    """Detect supported image formats from their file signatures."""

    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


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


@app.post("/api/vision/extract", response_model=VisionExtractResponse)
async def extract_image_text(
    file: UploadFile = File(...),
    client: VisionClient = Depends(get_vision_client),
) -> VisionExtractResponse:
    """Extract editable text from an image through the multimodal backend."""

    max_bytes = max(1, get_config().vision_max_image_mb) * 1024 * 1024
    image_bytes = await file.read(max_bytes + 1)
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail={"code": "empty_image", "message": "图片内容为空。"},
        )
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "image_too_large",
                "message": f"图片不能超过 {get_config().vision_max_image_mb} MB。",
            },
        )

    mime_type = detect_image_mime(image_bytes)
    if mime_type is None:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "unsupported_image",
                "message": "仅支持 JPG、PNG 或 WebP 图片。",
            },
        )

    try:
        extraction = await client.extract_text(image_bytes, mime_type)
    except VisionNotConfiguredError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "vision_not_configured",
                "message": "AI 识图服务尚未配置，请使用本地识别。",
            },
        ) from exc
    except VisionInvalidImageError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_image", "message": str(exc)},
        ) from exc
    except VisionProviderError as exc:
        logger.warning("vision provider unavailable: %s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "vision_provider_unavailable",
                "message": "AI 识图暂时不可用，请切换本地识别。",
            },
        ) from exc
    except VisionResponseError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "vision_invalid_response",
                "message": "AI 识图未返回有效文字，请切换本地识别。",
            },
        ) from exc

    return VisionExtractResponse(
        text=extraction.text,
        provider=extraction.provider,
        model=extraction.model,
        latency_ms=extraction.latency_ms,
        warnings=["AI 识图结果需要人工校对。"],
    )


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
