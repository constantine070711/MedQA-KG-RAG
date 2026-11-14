"""Per-client rate limiting for the public MedQA demo API.

The API container runs a single uvicorn worker (see ``api/Dockerfile``), so
process-local counters are enough for a demo deployment. If the API is ever
scaled to multiple workers or replicas, move these counters to Redis or apply
the limits at the reverse proxy instead.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Mapping

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


PATH_BUCKETS = {
    "/api/chat/stream": "chat",
    "/api/vision/extract": "vision",
    "/api/tools/execute": "tool",
}
EXEMPT_PATHS = {"/api/health"}
DEFAULT_LIMITS = {"chat": 20, "vision": 10, "tool": 40, "general": 120}


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def limits_from_env() -> dict[str, int]:
    """Read per-bucket request limits from the environment."""

    return {
        "chat": _env_int("RATE_LIMIT_CHAT_PER_MINUTE", DEFAULT_LIMITS["chat"]),
        "vision": _env_int("RATE_LIMIT_VISION_PER_MINUTE", DEFAULT_LIMITS["vision"]),
        "tool": _env_int("RATE_LIMIT_TOOL_PER_MINUTE", DEFAULT_LIMITS["tool"]),
        "general": _env_int("RATE_LIMIT_GENERAL_PER_MINUTE", DEFAULT_LIMITS["general"]),
    }


@dataclass
class _Window:
    started: float
    count: int


class RateLimiter:
    """Fixed-window counters keyed by client address and route bucket."""

    def __init__(
        self,
        limits: Mapping[str, int],
        window_seconds: float = 60.0,
    ) -> None:
        self.limits = dict(limits)
        self.window_seconds = window_seconds
        self._windows: dict[tuple[str, str], _Window] = {}
        self._lock = threading.Lock()

    def hit(
        self,
        client: str,
        bucket: str,
        now: float | None = None,
    ) -> tuple[bool, int]:
        """Record one request and return ``(allowed, retry_after_seconds)``."""

        limit = self.limits.get(bucket)
        if limit is None:
            return True, 0

        current = time.monotonic() if now is None else now
        key = (client, bucket)
        with self._lock:
            window = self._windows.get(key)
            if window is None or current - window.started >= self.window_seconds:
                self._windows[key] = _Window(current, 1)
                self._prune(current)
                return True, 0
            if window.count >= limit:
                remaining = self.window_seconds - (current - window.started)
                return False, max(1, int(remaining + 0.999))
            window.count += 1
            return True, 0

    def _prune(self, now: float) -> None:
        if len(self._windows) < 5000:
            return
        cutoff = now - self.window_seconds
        stale = [key for key, window in self._windows.items() if window.started < cutoff]
        for key in stale:
            del self._windows[key]


class RateLimitMiddleware:
    """Pure ASGI middleware so streaming responses are never buffered."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        limits: Mapping[str, int] | None = None,
        trust_proxy_headers: bool | None = None,
        window_seconds: float = 60.0,
    ) -> None:
        self.app = app
        self.trust_proxy_headers = (
            _env_flag("TRUST_PROXY_HEADERS")
            if trust_proxy_headers is None
            else trust_proxy_headers
        )
        self.limiter = RateLimiter(limits or limits_from_env(), window_seconds)

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http" or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        bucket = PATH_BUCKETS.get(path, "general")
        allowed, retry_after = self.limiter.hit(self._client_ip(scope), bucket)
        if not allowed:
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": {
                        "code": "rate_limited",
                        "message": "请求过于频繁，请稍后再试。",
                        "retry_after_seconds": retry_after,
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

    def _client_ip(self, scope: Scope) -> str:
        if self.trust_proxy_headers:
            headers = Headers(scope=scope)
            forwarded = headers.get("x-forwarded-for")
            if forwarded:
                return forwarded.split(",", 1)[0].strip()
            real_ip = headers.get("x-real-ip")
            if real_ip:
                return real_ip.strip()

        client = scope.get("client")
        return client[0] if client else "unknown"
