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

from .config import Config, get_config
from .embedder import Embedder
from .llm_client import LLMClient
from .data_loader import DataLoader, Document
from .rag_engine import RAGEngine
from .metadata import DocumentMetadata, MetadataRegistry, parse_simple_yaml

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
