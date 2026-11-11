"""Contract tests for the backend image-understanding endpoint."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from api.main import app, detect_image_mime, get_vision_client
from src.vision_client import (
    VisionExtraction,
    VisionProviderError,
)


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"test-image"


class SuccessVisionClient:
    async def extract_text(
        self,
        image_bytes: bytes,
        mime_type: str,
    ) -> VisionExtraction:
        if image_bytes != PNG_BYTES or mime_type != "image/png":
            raise AssertionError("unexpected vision input")
        return VisionExtraction(
            text="二甲双胍 500 mg",
            provider="deepseek",
            model="vision-test-model",
            latency_ms=12.5,
        )


class FailingVisionClient:
    async def extract_text(
        self,
        image_bytes: bytes,
        mime_type: str,
    ) -> VisionExtraction:
        raise VisionProviderError("provider unavailable")


class TestImageSignature(unittest.TestCase):
    def test_detects_supported_images(self):
        self.assertEqual(detect_image_mime(PNG_BYTES), "image/png")
        self.assertEqual(detect_image_mime(b"\xff\xd8\xffjpeg"), "image/jpeg")
        self.assertEqual(
            detect_image_mime(b"RIFF" + b"0000" + b"WEBP"),
            "image/webp",
        )

    def test_rejects_unknown_image(self):
        self.assertIsNone(detect_image_mime(b"not-an-image"))


class TestVisionEndpoint(unittest.TestCase):
    def test_extracts_text_with_injected_vision_client(self):
        app.dependency_overrides[get_vision_client] = lambda: SuccessVisionClient()
        try:
            with TestClient(app) as client:
                response = client.post(
                    "/api/vision/extract",
                    files={"file": ("drug.png", PNG_BYTES, "image/png")},
                )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["text"], "二甲双胍 500 mg")
            self.assertEqual(payload["provider"], "deepseek")
            self.assertEqual(payload["model"], "vision-test-model")
            self.assertTrue(payload["warnings"])
        finally:
            app.dependency_overrides.clear()

    def test_rejects_non_image_payload(self):
        app.dependency_overrides[get_vision_client] = lambda: SuccessVisionClient()
        try:
            with TestClient(app) as client:
                response = client.post(
                    "/api/vision/extract",
                    files={"file": ("notes.txt", b"plain text", "text/plain")},
                )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["detail"]["code"], "unsupported_image")
        finally:
            app.dependency_overrides.clear()

    def test_provider_failure_is_recoverable(self):
        app.dependency_overrides[get_vision_client] = lambda: FailingVisionClient()
        try:
            with TestClient(app) as client:
                response = client.post(
                    "/api/vision/extract",
                    files={"file": ("drug.png", PNG_BYTES, "image/png")},
                )
            self.assertEqual(response.status_code, 503)
            self.assertEqual(
                response.json()["detail"]["code"],
                "vision_provider_unavailable",
            )
        finally:
            app.dependency_overrides.clear()


if __name__ == "__main__":
    unittest.main()
