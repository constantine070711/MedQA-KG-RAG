"""Backend-only image understanding client for MedQA.

The client accepts an image from the API layer, sends it to an
OpenAI-compatible multimodal model, and returns editable text. Images and
recognized text are not persisted by this module.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

from src.config import Config, get_config


if TYPE_CHECKING:
    from openai import AsyncOpenAI


logger = logging.getLogger(__name__)

VISION_PROMPT = (
    "你是医学图片文字提取器。请只提取图片中清晰可见的文字，"
    "保持药品名、数值、单位和表格顺序。看不清的内容标记为 [无法辨认]，"
    "不要猜测，不要补全，不要进行诊断或解释。"
    "请直接输出纯文本，不要使用 Markdown 代码块。"
)


class VisionError(Exception):
    """Base class for image-understanding failures."""


class VisionNotConfiguredError(VisionError):
    """Raised when the vision API is disabled or missing credentials."""


class VisionInvalidImageError(VisionError):
    """Raised when the supplied image bytes or MIME type are invalid."""


class VisionProviderError(VisionError):
    """Raised when the upstream multimodal provider request fails."""


class VisionResponseError(VisionError):
    """Raised when the provider returns an empty or malformed response."""


@dataclass(frozen=True)
class VisionExtraction:
    """Text extracted by a multimodal model."""

    text: str
    provider: str
    model: str
    latency_ms: float


class VisionClient:
    """Small provider abstraction for image-to-text extraction."""

    provider = "deepseek"

    def __init__(
        self,
        config: Optional[Config] = None,
        client: Optional["AsyncOpenAI"] = None,
    ) -> None:
        self.config = config or get_config()
        self._client = client

    @property
    def configured(self) -> bool:
        """Return whether the feature has enough configuration to run."""

        api_key = self.config.vision_api_key.strip()
        return bool(
            self.config.vision_enabled
            and api_key
            and "your_" not in api_key
        )

    def _get_client(self) -> "AsyncOpenAI":
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise VisionNotConfiguredError(
                    "OpenAI SDK is not installed"
                ) from exc
            self._client = AsyncOpenAI(
                api_key=self.config.vision_api_key,
                base_url=self.config.vision_base_url,
                timeout=float(self.config.vision_timeout_seconds),
            )
        return self._client

    async def extract_text(
        self,
        image_bytes: bytes,
        mime_type: str,
    ) -> VisionExtraction:
        """Extract visible text from one image using a multimodal model."""

        if not self.configured:
            raise VisionNotConfiguredError("vision API is not configured")
        if not image_bytes:
            raise VisionInvalidImageError("image is empty")
        if not mime_type.startswith("image/"):
            raise VisionInvalidImageError("unsupported image MIME type")

        encoded = base64.b64encode(image_bytes).decode("ascii")
        image_url = f"data:{mime_type};base64,{encoded}"
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ]

        client = self._get_client()
        started = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=self.config.vision_model,
                messages=messages,
                temperature=0,
                max_tokens=2000,
            )
        except Exception as exc:
            logger.warning(
                "vision provider request failed: %s",
                type(exc).__name__,
            )
            raise VisionProviderError(
                "multimodal provider request failed"
            ) from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise VisionResponseError(
                "multimodal provider returned no choices"
            ) from exc

        text = self._content_to_text(content).strip()
        if not text:
            raise VisionResponseError("multimodal provider returned empty text")

        return VisionExtraction(
            text=text,
            provider=self.provider,
            model=str(getattr(response, "model", None) or self.config.vision_model),
            latency_ms=round((time.perf_counter() - started) * 1000.0, 1),
        )

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(parts)
        return ""


@lru_cache(maxsize=1)
def get_vision_client() -> VisionClient:
    """Return the process-wide multimodal client."""

    return VisionClient()
