"""
医疗知识图谱RAG问答系统 — src 包

模块结构:
    config.py       — 统一配置管理
    embedder.py     — Embedding 模型封装 (bge-small-zh-v1.5)
    llm_client.py   — DeepSeek API 封装 (OpenAI 兼容)
    data_loader.py  — 文档加载/清洗/分块 (Phase 3: 多格式 + metadata)
    rag_engine.py   — LightRAG 初始化和查询封装
    metadata.py     — 文档元数据管理 (Phase 3)
    medical_core/   — 医疗领域核心 (实体/关系/安全规则)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any


_EXPORTS = {
    "Config": ".config",
    "get_config": ".config",
    "Embedder": ".embedder",
    "LLMClient": ".llm_client",
    "DataLoader": ".data_loader",
    "Document": ".data_loader",
    "RAGEngine": ".rag_engine",
    "DocumentMetadata": ".metadata",
    "MetadataRegistry": ".metadata",
    "parse_simple_yaml": ".metadata",
}

__all__ = [
    "Config",
    "get_config",
    "Embedder",
    "LLMClient",
    "DataLoader",
    "Document",
    "RAGEngine",
    "DocumentMetadata",
    "MetadataRegistry",
    "parse_simple_yaml",
]


if TYPE_CHECKING:
    from .config import Config, get_config
    from .data_loader import DataLoader, Document
    from .embedder import Embedder
    from .llm_client import LLMClient
    from .metadata import DocumentMetadata, MetadataRegistry, parse_simple_yaml
    from .rag_engine import RAGEngine


def __getattr__(name: str) -> Any:
    """Lazily import public classes so lightweight modules avoid model imports."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
