"""
LLM API 调用模块 (DeepSeek)

职责:
    - 封装 DeepSeek API 调用（OpenAI 兼容接口）
    - 提供 LightRAG 兼容的 llm_model_func 回调
    - 统一管理 token 统计和错误处理

DeepSeek 优势:
    - 中文能力强，医疗领域表现好
    - 价格极低（约 ¥1/百万 token）
    - OpenAI 兼容接口，无需额外适配

用法:
    from src.llm_client import LLMClient
    llm = LLMClient()
    result = await llm.llm_model_func("你好", system_prompt="你是助手")
"""

import logging
from typing import Optional

from lightrag.llm.openai import openai_complete_if_cache

from .config import get_config

logger = logging.getLogger(__name__)


class LLMClient:
    """
    DeepSeek API 客户端封装

    Attributes:
        model: 模型名称
        api_key: API Key
        base_url: API 端点
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        初始化 LLM 客户端。

        Args:
            model: 模型名称，默认从配置读取
            api_key: API Key，默认从配置读取
            base_url: API 端点，默认从配置读取
        """
        cfg = get_config()

        self.model = model or cfg.deepseek_model
        self.api_key = api_key or cfg.deepseek_api_key
        self.base_url = base_url or cfg.deepseek_base_url

    # ----------------------------------------------------------
    # LightRAG 兼容接口
    # ----------------------------------------------------------

    async def llm_model_func(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: list = None,
        **kwargs,
    ) -> str:
        """
        LightRAG 要求的 LLM 回调函数。

        通过 OpenAI 兼容接口调用 DeepSeek API。

        Args:
            prompt: 用户 prompt
            system_prompt: 系统 prompt（可选）
            history_messages: 历史消息列表（可选）
            **kwargs: 传递给 openai_complete_if_cache 的额外参数

        Returns:
            LLM 生成的文本回复

        Note:
            - LightRAG 内置了 LLM 缓存机制，相同 prompt 不会重复调用 API
            - 缓存存储在 KV Store 中 (kv_store_*.json)
        """
        if history_messages is None:
            history_messages = []

        return await openai_complete_if_cache(
            model=self.model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs,
        )

    # ----------------------------------------------------------
    # 扩展接口（后续 Agent/UI 使用）
    # ----------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        标准 Chat Completion 接口（非 LightRAG 路径）。

        Args:
            messages: [{"role": "user", "content": "..."}, ...]
            temperature: 生成温度
            max_tokens: 最大输出 token

        Returns:
            生成的文本
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    # ----------------------------------------------------------
    # 信息查询
    # ----------------------------------------------------------

    def get_info(self) -> dict:
        """返回 LLM 客户端信息"""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key and "your-deepseek" not in self.api_key),
        }


# ============================================================
# 模块级便捷工厂
# ============================================================

_llm_client_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取全局 LLMClient 单例"""
    global _llm_client_instance

    if _llm_client_instance is None:
        _llm_client_instance = LLMClient()

    return _llm_client_instance
