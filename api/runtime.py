"""Streaming adapter around the existing MedQA LangGraph agent."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from app.ui_helpers import build_conversation_context
from src.monitoring import log_run, new_request_id

from api.schemas import ChatRequest, HistoryTurn


_STAGE_LABELS = {
    "safety_check": ("safety", "安全检查完成"),
    "neo4j_retrieve": ("retrieval", "知识图谱检索完成"),
    "build_prompt": ("fusion", "证据上下文已构建"),
    "generate": ("generation", "回答生成完成"),
    "analyze": ("analysis", "可靠性与引用分析完成"),
    "decide": ("decision", "安全与可靠性判定完成"),
    "finalize": ("finalize", "回答与证据已封装"),
}


def sse_event(payload: dict[str, Any]) -> str:
    """Encode one JSON payload as a Server-Sent Event."""

    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {data}\n\n"


def history_to_entries(history: list[HistoryTurn]) -> list[dict[str, Any]]:
    """Map compact browser history to the shape consumed by v1 memory helpers."""

    entries: list[dict[str, Any]] = []
    for turn in history:
        entries.append(
            {
                "question": turn.question,
                "final_response": {
                    "answer": turn.answer,
                    "safety_level": turn.safety_level or "safe",
                    "reliability_decision": {
                        "is_final_refusal": turn.is_final_refusal,
                    },
                },
            }
        )
    return entries


def _stage_event(node: str, update: dict[str, Any]) -> dict[str, Any]:
    stage, label = _STAGE_LABELS.get(node, (node, "工作流处理中"))
    detail = ""

    if node == "safety_check":
        level = str(update.get("safety_level") or "unknown")
        if level == "blocked":
            detail = "检测到安全边界，正在准备安全响应"
        else:
            detail = "正在检索结构化医疗知识"
    elif node == "neo4j_retrieve":
        mentions = update.get("mentions") or []
        if update.get("neo4j_unavailable"):
            detail = "图谱服务暂不可用，将使用已配置的 RAG 证据"
        elif mentions:
            detail = f"图谱命中实体：{'、'.join(str(m) for m in mentions[:4])}"
        else:
            detail = "未命中结构化实体，继续执行混合检索"
    elif node == "build_prompt":
        detail = "正在融合安全指令、图谱证据与检索上下文"
    elif node == "generate":
        detail = "回答已生成，正在校验引用与可靠性"
    elif node == "analyze":
        detail = "正在执行确定性安全与引用信号分析"
    elif node == "decide":
        detail = "正在确认最终回答策略"
    elif node == "finalize":
        detail = "正在整理可追溯证据"

    return {
        "type": "status",
        "stage": stage,
        "label": label,
        "detail": detail,
    }


class AgentRuntime:
    """Serialize access to LightRAG and stream LangGraph node updates."""

    def __init__(
        self,
        agent: Any | None = None,
        *,
        chunk_size: int = 28,
        chunk_delay: float = 0.006,
        heartbeat_seconds: float = 10.0,
    ) -> None:
        if agent is None:
            from src.agent import MedQAAgent

            agent = MedQAAgent(enforce_safe_generation_prompt=True)
        self._agent = agent
        self._lock = asyncio.Lock()
        self._chunk_size = max(1, chunk_size)
        self._chunk_delay = max(0.0, chunk_delay)
        self._heartbeat_seconds = max(1.0, heartbeat_seconds)

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Yield SSE status, answer delta, and final response events."""

        request_id = new_request_id()
        started = time.perf_counter()
        context = build_conversation_context(history_to_entries(request.history))

        yield sse_event(
            {
                "type": "status",
                "stage": "queued",
                "label": "请求已接收",
                "detail": "正在检查安全边界",
                "request_id": request_id,
            }
        )

        final_response: dict[str, Any] | None = None
        try:
            async with self._lock:
                async for update in self._agent.astream_state(
                    request.question,
                    conversation_context=context or None,
                ):
                    if not isinstance(update, dict) or not update:
                        continue
                    node = next(iter(update))
                    node_update = update.get(node) or {}
                    if not isinstance(node_update, dict):
                        node_update = {}

                    yield sse_event(_stage_event(node, node_update))

                    if node == "finalize":
                        candidate = node_update.get("final_response")
                        if isinstance(candidate, dict):
                            final_response = candidate

            if final_response is None:
                raise RuntimeError("Agent workflow finished without final_response")

            answer = str(final_response.get("answer") or "")
            for offset in range(0, len(answer), self._chunk_size):
                yield sse_event(
                    {
                        "type": "answer_delta",
                        "delta": answer[offset : offset + self._chunk_size],
                    }
                )
                if self._chunk_delay:
                    await asyncio.sleep(self._chunk_delay)

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            try:
                log_run(
                    request_id,
                    request.question,
                    final_response,
                    elapsed_ms,
                    conversation_context=context or None,
                )
            except Exception:
                pass

            yield sse_event(
                {
                    "type": "final",
                    "request_id": request_id,
                    "elapsed_ms": round(elapsed_ms, 1),
                    "response": final_response,
                }
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield sse_event(
                {
                    "type": "error",
                    "request_id": request_id,
                    "message": "本次请求暂时无法完成，请稍后重试。",
                    "error_type": type(exc).__name__,
                }
            )

    async def stream_with_heartbeat(self, request: ChatRequest) -> AsyncIterator[str]:
        """Proxy runtime events while keeping long generations alive through SSE."""

        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def produce() -> None:
            try:
                async for chunk in self.stream(request):
                    await queue.put(("chunk", chunk))
            except Exception as exc:
                await queue.put(("error", exc))
            finally:
                await queue.put(("done", None))

        task = asyncio.create_task(produce())
        try:
            while True:
                try:
                    kind, value = await asyncio.wait_for(
                        queue.get(), timeout=self._heartbeat_seconds
                    )
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                if kind == "chunk":
                    yield str(value)
                elif kind == "error":
                    yield sse_event(
                        {
                            "type": "error",
                            "message": "流式服务暂时不可用，请稍后重试。",
                            "error_type": type(value).__name__,
                        }
                    )
                else:
                    break
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
