"""Tests for the public-demo rate limiter and ASGI middleware."""

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.ratelimit import RateLimiter, RateLimitMiddleware


class TestRateLimiter(unittest.TestCase):
    def test_allows_limit_then_blocks(self):
        limiter = RateLimiter({"chat": 2}, window_seconds=60)

        self.assertEqual(limiter.hit("1.2.3.4", "chat", now=0.0), (True, 0))
        self.assertEqual(limiter.hit("1.2.3.4", "chat", now=1.0), (True, 0))
        allowed, retry_after = limiter.hit("1.2.3.4", "chat", now=2.0)

        self.assertFalse(allowed)
        self.assertEqual(retry_after, 58)

    def test_window_resets(self):
        limiter = RateLimiter({"chat": 1}, window_seconds=60)

        self.assertEqual(limiter.hit("1.2.3.4", "chat", now=0.0), (True, 0))
        self.assertFalse(limiter.hit("1.2.3.4", "chat", now=30.0)[0])
        self.assertEqual(limiter.hit("1.2.3.4", "chat", now=60.0), (True, 0))

    def test_clients_and_buckets_are_independent(self):
        limiter = RateLimiter({"chat": 1, "vision": 1}, window_seconds=60)

        self.assertTrue(limiter.hit("a", "chat", now=0.0)[0])
        self.assertTrue(limiter.hit("b", "chat", now=0.0)[0])
        self.assertTrue(limiter.hit("a", "vision", now=0.0)[0])
        self.assertFalse(limiter.hit("a", "chat", now=0.0)[0])

    def test_unknown_bucket_is_not_limited(self):
        limiter = RateLimiter({"chat": 1}, window_seconds=60)

        self.assertEqual(limiter.hit("a", "unknown", now=0.0), (True, 0))


def _demo_app(**middleware_kwargs) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, **middleware_kwargs)

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/chat/stream")
    async def chat():
        return {"ok": True}

    return app


class TestRateLimitMiddleware(unittest.TestCase):
    def test_returns_429_with_retry_after(self):
        client = TestClient(
            _demo_app(
                limits={"chat": 1, "general": 10},
                trust_proxy_headers=True,
            )
        )

        first = client.post("/api/chat/stream")
        second = client.post("/api/chat/stream")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.json()["detail"]["code"], "rate_limited")
        self.assertGreater(int(second.headers["Retry-After"]), 0)

    def test_forwarded_clients_get_separate_quotas(self):
        client = TestClient(
            _demo_app(
                limits={"chat": 1, "general": 10},
                trust_proxy_headers=True,
            )
        )

        self.assertEqual(
            client.post(
                "/api/chat/stream",
                headers={"X-Forwarded-For": "203.0.113.1"},
            ).status_code,
            200,
        )
        self.assertEqual(
            client.post(
                "/api/chat/stream",
                headers={"X-Forwarded-For": "203.0.113.2"},
            ).status_code,
            200,
        )
        self.assertEqual(
            client.post(
                "/api/chat/stream",
                headers={"X-Forwarded-For": "203.0.113.1"},
            ).status_code,
            429,
        )

    def test_health_endpoint_is_exempt(self):
        client = TestClient(
            _demo_app(limits={"general": 1}, trust_proxy_headers=True)
        )

        for _ in range(5):
            self.assertEqual(client.get("/api/health").status_code, 200)

    def test_forwarded_header_is_ignored_without_proxy_trust(self):
        client = TestClient(
            _demo_app(
                limits={"chat": 1, "general": 10},
                trust_proxy_headers=False,
            )
        )

        first = client.post(
            "/api/chat/stream",
            headers={"X-Forwarded-For": "203.0.113.1"},
        )
        second = client.post(
            "/api/chat/stream",
            headers={"X-Forwarded-For": "203.0.113.2"},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)


if __name__ == "__main__":
    unittest.main()
