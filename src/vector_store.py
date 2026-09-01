"""
向量存储抽象接口模块

职责:
    - 定义向量存储的抽象接口（VectorStore ABC）
    - 为未来切换向量数据库后端提供统一 API
    - 解除上层代码与具体向量数据库实现的耦合

当前状态 (2026-07-26):
    - 主力: LightRAG 默认 NanoVectorDB (JSON 文件存储)
    - 规模: 380 实体 + 422 关系, 512d 向量
    - 检索延迟: <100ms (当前规模下完全足够)

为什么不集成 ChromaDB:
    - LightRAG 1.5.4 已正式弃用 ChromaDB 支持 (代码移至 kg/deprecated/)
    - 验证层 (verify_storage_implementation) 拒绝 ChromaVectorDBStorage
    - 模块解析层 (STORAGES 映射) 指向不存在的 chroma_impl.py
    - 已弃用代码存在语法错误 (重复 except 块)
    - 强行集成需要 monkey-patch LightRAG 内部实现，违反稳定性原则
    - NanoVectorDB 在 10,000 文档以下完全够用，当前仅 14 个文档

LightRAG 1.5.4 官方支持的向量存储后端:
    - NanoVectorDBStorage  (默认, JSON 文件, 零配置)     ← 当前使用
    - MilvusVectorDBStorage (分布式, Docker 部署)         ← 大规模场景
    - PGVectorStorage       (PostgreSQL 扩展, SQL 生态)   ← 中等规模
    - FaissVectorDBStorage  (Facebook, 高性能 ANN)        ← 单机高性能
    - QdrantVectorDBStorage (Rust 实现, Docker 部署)      ← 生产环境
    - MongoVectorDBStorage  (MongoDB 扩展)                ← 文档数据库场景
    - OpenSearchVectorDBStorage (Elasticsearch 生态)      ← 全文+向量混合

未来切换向量数据库的触发条件 (需同时满足):
    1. 知识文件数 > 500
    2. 实体数 > 10,000
    3. NanoVectorDB 检索延迟 > 3s
    4. 需要多集合/多租户管理
    5. 需要分布式部署

切换目标优先级 (按 LightRAG 官方支持程度):
    1. FaissVectorDBStorage  — pip install, 无需 Docker, 性能好
    2. PGVectorStorage       — 如果项目已有 PostgreSQL
    3. QdrantVectorDBStorage — 如果需要生产级分布式

用法:
    from src.vector_store import VectorStore, get_current_store_info

    info = get_current_store_info()
    print(f"当前向量存储: {info['backend']}, {info['entity_count']} 实体")
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# 抽象接口
# ============================================================

class VectorStore(ABC):
    """
    向量存储抽象基类

    定义向量数据库的标准操作接口。当前由 LightRAG 的 NanoVectorDB
    提供具体实现。未来如需切换后端，只需实现此接口并注入到
    RAGEngine 即可。
    """

    @abstractmethod
    async def upsert(self, vectors: dict[str, dict]) -> None:
        """
        插入或更新向量。

        Args:
            vectors: {id: {"content": str, "embedding": list[float], ...}}
        """
        ...

    @abstractmethod
    async def query(self, query: str, top_k: int) -> list[dict]:
        """
        向量相似度检索。

        Args:
            query: 查询文本
            top_k: 返回 top-k 个结果

        Returns:
            [{"id": str, "content": str, "distance": float, ...}, ...]
        """
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """删除指定向量"""
        ...

    @abstractmethod
    async def drop(self) -> dict:
        """清空所有向量数据"""
        ...


# ============================================================
# 当前存储信息
# ============================================================

@dataclass
class VectorStoreInfo:
    """当前向量存储状态信息"""
    backend: str = "NanoVectorDB"
    backend_version: str = "LightRAG 1.5.4 built-in"
    description: str = "JSON 文件存储，零配置，适合 MVP/小规模场景"
    official_supported: bool = True
    storage_path: str = "lightrag_cache/"
    estimated_max_documents: int = 10_000
    estimated_max_entities: int = 100_000
    current_entity_count: int = 0
    current_relation_count: int = 0
    notes: list[str] = field(default_factory=lambda: [
        "当前规模 (380 实体) 远低于上限 (100,000)，无需切换后端",
        "LightRAG 1.5.4 官方支持的替代方案: Faiss/Qdrant/Milvus/PGVector",
        "切换后端只需修改 LightRAG 构造参数 vector_storage='...'",
    ])


def get_current_store_info(
    entity_count: int = 0,
    relation_count: int = 0,
) -> dict:
    """
    获取当前向量存储状态摘要。

    Args:
        entity_count: 当前实体数
        relation_count: 当前关系数

    Returns:
        存储状态字典
    """
    info = VectorStoreInfo(
        current_entity_count=entity_count,
        current_relation_count=relation_count,
    )
    return {
        "backend": info.backend,
        "description": info.description,
        "official_supported": info.official_supported,
        "estimated_max_entities": info.estimated_max_entities,
        "current_entity_count": info.current_entity_count,
        "utilization_pct": round(
            info.current_entity_count / info.estimated_max_entities * 100, 2
        ) if info.estimated_max_entities else 0,
        "recommendation": (
            "保持 NanoVectorDB"
            if info.current_entity_count < info.estimated_max_entities * 0.5
            else "考虑评估替代后端"
        ),
        "alternative_backends": [
            "FaissVectorDBStorage",
            "QdrantVectorDBStorage",
            "PGVectorStorage",
            "MilvusVectorDBStorage",
        ],
    }
