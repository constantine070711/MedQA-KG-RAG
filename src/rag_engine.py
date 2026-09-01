"""
LightRAG 检索引擎封装模块

职责:
    - 封装 LightRAG 初始化、索引、查询流程
    - 提供同步/异步统一接口
    - 管理缓存生命周期
    - 为未来 LangGraph Agent 和 Streamlit UI 提供简洁 API

检索模式:
    - naive:  纯向量检索
    - local:  实体邻居检索（1-hop 子图）
    - global: 全局关系检索
    - hybrid: local + global
    - mix:    全混合（推荐默认）

用法:
    from src.rag_engine import RAGEngine
    engine = RAGEngine()
    await engine.initialize()

    # 索引文档
    await engine.insert_document("糖尿病是一种慢性代谢性疾病...")

    # 查询
    answer = await engine.query("糖尿病有哪些症状?")
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from .config import get_config
from .embedder import get_embedder
from .llm_client import get_llm_client

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    LightRAG 检索引擎封装

    统一管理:
        - LightRAG 实例生命周期
        - 文档索引（支持增量更新）
        - 混合检索查询（5 种模式）
        - 缓存管理

    Attributes:
        working_dir: LightRAG 缓存目录
        rag: LightRAG 实例
        embedder: Embedder 实例
        llm_client: LLMClient 实例
    """

    def __init__(
        self,
        working_dir: Optional[str] = None,
        embedder=None,
        llm_client=None,
    ):
        """
        初始化 RAG 引擎。

        Args:
            working_dir: LightRAG 工作目录，默认从配置读取
            embedder: Embedder 实例，默认使用全局单例
            llm_client: LLMClient 实例，默认使用全局单例
        """
        cfg = get_config()

        self.working_dir = working_dir or str(cfg.cache_dir)
        self.embedder = embedder or get_embedder()
        self.llm_client = llm_client or get_llm_client()

        self._rag: Optional[LightRAG] = None
        self._initialized: bool = False

    # ----------------------------------------------------------
    # 初始化
    # ----------------------------------------------------------

    async def initialize(self) -> "RAGEngine":
        """
        初始化 LightRAG 实例和底层存储。

        Returns:
            self，支持链式调用
        """
        if self._initialized:
            logger.info("RAGEngine 已初始化，跳过")
            return self

        cfg = get_config()

        logger.info(f"初始化 LightRAG (working_dir={self.working_dir})")

        self._rag = LightRAG(
            working_dir=self.working_dir,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedder.dim,
                max_token_size=cfg.embedding_max_token_size,
                func=self.embedder.embedding_func,
            ),
            llm_model_func=self.llm_client.llm_model_func,
        )

        await self._rag.initialize_storages()

        self._initialized = True
        logger.info("LightRAG 初始化完成")

        return self

    @property
    def rag(self) -> LightRAG:
        """获取底层 LightRAG 实例（需先调用 initialize()）"""
        if self._rag is None:
            raise RuntimeError(
                "RAGEngine 未初始化，请先调用 await engine.initialize()"
            )
        return self._rag

    # ----------------------------------------------------------
    # 文档索引
    # ----------------------------------------------------------

    async def insert_document(self, text: str) -> None:
        """
        索引单个文档（增量更新）。

        LightRAG 会自动:
            1. 文本分块 (token_size=1200, overlap=100)
            2. 实体抽取 (LLM)
            3. 关系抽取 (LLM)
            4. 向量化 (Embedding)
            5. 存入图库 + 向量库

        Args:
            text: 文档文本内容
        """
        await self.rag.ainsert(text)
        logger.info(f"文档已索引 ({len(text)} 字符)")

    async def insert_documents(self, texts: list[str]) -> None:
        """
        批量索引多个文档。

        Args:
            texts: 文档文本列表
        """
        for i, text in enumerate(texts):
            logger.info(f"索引文档 [{i+1}/{len(texts)}]...")
            await self.insert_document(text)

        logger.info(f"批量索引完成: {len(texts)} 个文档")

    async def insert_file(self, file_path: str) -> None:
        """
        索引单个文件。

        Args:
            file_path: 文件路径
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        text = path.read_text(encoding="utf-8")
        await self.insert_document(text)
        logger.info(f"文件已索引: {path.name}")

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------

    async def query(
        self,
        question: str,
        mode: str = "mix",
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        执行 RAG 查询（检索 + 生成）。

        Args:
            question: 用户问题
            mode: 检索模式 —
                "mix"    全混合 (推荐, 默认)
                "hybrid" local + global
                "local"  实体邻居检索
                "global" 全局关系检索
                "naive"  纯向量检索
            system_prompt: 自定义系统 prompt（可选）

        Returns:
            LLM 生成的回答（包含来源引用）
        """
        # 构建查询参数
        param = QueryParam(mode=mode)

        # 如果需要自定义 system_prompt，通过 kwargs 传递
        # LightRAG 的 aquery 支持 system_prompt 参数
        if system_prompt:
            result = await self.rag.aquery(
                question,
                param=param,
                system_prompt=system_prompt,
            )
        else:
            result = await self.rag.aquery(question, param=param)

        return result

    async def query_with_medical_prompt(self, question: str) -> str:
        """
        使用医疗安全 prompt 的查询。

        自动附加医疗免责声明和安全指令。

        Args:
            question: 用户问题

        Returns:
            带安全边界的回答
        """
        medical_prompt = (
            "你是一个医疗知识辅助助手。\n"
            "重要规则:\n"
            "1. 你的回答不能替代医生诊断和治疗建议\n"
            "2. 如果问题涉及个人病情诊断，请建议用户咨询专业医生\n"
            "3. 对于不确定的医学问题，请明确说明不确定性\n"
            "4. 回答应基于提供的医学知识库，并注明来源\n"
            "5. 遇到药物剂量问题时，必须强调遵医嘱"
        )
        return await self.query(question, mode="mix", system_prompt=medical_prompt)

    async def retrieve(
        self,
        question: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 20,
    ) -> dict:
        """
        执行纯检索（不调用 LLM 生成回答），返回结构化检索上下文。

        使用 LightRAG.aquery_data() — 复用与 aquery() 相同的检索逻辑，
        但在 LLM 生成之前停止，返回最终处理后的 entities / relationships / chunks。

        Args:
            question: 查询问题
            mode: 检索模式 — "mix" / "hybrid" / "local" / "global" / "naive"
            top_k: 实体/关系检索数量
            chunk_top_k: 文本块检索数量

        Returns:
            LightRAG aquery_data() 的完整返回:
            {
                "status": "success" | "failure",
                "message": str,
                "data": {
                    "entities": [{entity_name, entity_type, description,
                                  source_id, file_path, reference_id, ...}],
                    "relationships": [{src_id, tgt_id, description, keywords,
                                       weight, source_id, file_path, reference_id, ...}],
                    "chunks": [{content, file_path, chunk_id, reference_id, ...}],
                    "references": [{reference_id, file_path, ...}],
                },
                "metadata": {
                    "query_mode": str,
                    "keywords": {"high_level": [...], "low_level": [...]},
                    "processing_info": {  # 可选，查询结果为空时可能不存在
                        "total_entities_found", "total_relations_found",
                        "entities_after_truncation", "relations_after_truncation",
                        "merged_chunks_count", "final_chunks_count",
                    },
                },
            }

        Note:
            - 不修改 LightRAG 源码
            - 不调用 LLM，不会产生 API 费用
            - naive 模式下 entities / relationships 为空数组（正常行为）
            - failure 时 data 为空 dict {}
        """
        param = QueryParam(
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
        )
        return await self.rag.aquery_data(question, param)

    async def query_with_context(
        self,
        question: str,
        mode: str = "mix",
        system_prompt: str = None,
        top_k: int = 60,
        chunk_top_k: int = 20,
    ) -> dict:
        """
        执行查询并同时返回结构化检索上下文和 LLM 生成回答。

        使用 LightRAG.aquery_llm() — 一次调用同时获取检索结果和生成结果，
        避免重复检索。

        Args:
            question: 查询问题
            mode: 检索模式 — "mix" / "hybrid" / "local" / "global" / "naive"
            system_prompt: 自定义系统 prompt（可选）
            top_k: 实体/关系检索数量
            chunk_top_k: 文本块检索数量

        Returns:
            {
                "status": "success" | "failure",
                "data": {
                    "entities": [...],
                    "relationships": [...],
                    "chunks": [...],
                    "references": [...],
                },
                "metadata": {...},
                "llm_response": str,  # LLM 生成的回答文本
            }

        Note:
            - 不修改现有 query() 方法
            - data 来自 aquery_llm() 原始返回的 data 字段
            - metadata 来自原始返回的 metadata 字段
            - failure 时 data 为 {}，llm_response 为 ""
        """
        param = QueryParam(
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
        )
        result = await self.rag.aquery_llm(question, param, system_prompt)

        # 安全提取 llm_response content（streaming 模式下为 None）
        llm_resp = result.get("llm_response", {}) or {}
        llm_content = llm_resp.get("content") or ""

        return {
            "status": result.get("status", "unknown"),
            "data": result.get("data", {}),
            "metadata": result.get("metadata", {}),
            "llm_response": llm_content,
        }

    # ----------------------------------------------------------
    # 缓存管理
    # ----------------------------------------------------------

    def clear_cache(self) -> None:
        """
        清除 LightRAG 缓存（删除所有索引数据）。

        Warning:
            此操作不可逆！清除后需要重新索引所有文档。
        """
        import shutil

        cache_path = Path(self.working_dir)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            logger.info(f"缓存已清除: {self.working_dir}")
            self._initialized = False
            self._rag = None

    def get_cache_info(self) -> dict:
        """返回缓存目录信息"""
        cache_path = Path(self.working_dir)
        if not cache_path.exists():
            return {"exists": False, "files": [], "total_size_mb": 0}

        files = list(cache_path.glob("*"))
        total_size = sum(
            f.stat().st_size for f in files if f.is_file()
        )

        return {
            "exists": True,
            "working_dir": str(cache_path),
            "file_count": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": [f.name for f in sorted(files)],
        }

    # ----------------------------------------------------------
    # 状态检查
    # ----------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """是否已完成初始化"""
        return self._initialized

    def get_info(self) -> dict:
        """返回引擎状态信息"""
        return {
            "initialized": self._initialized,
            "working_dir": self.working_dir,
            "embedder": self.embedder.get_info(),
            "llm": self.llm_client.get_info(),
            "cache": self.get_cache_info(),
        }


# ============================================================
# 模块级便捷工厂
# ============================================================

_engine_instance: Optional[RAGEngine] = None


async def get_rag_engine(force_new: bool = False) -> RAGEngine:
    """
    获取全局 RAGEngine 单例（已初始化）。

    Args:
        force_new: 是否强制创建新实例

    Returns:
        已初始化的 RAGEngine 实例
    """
    global _engine_instance

    if _engine_instance is not None and not force_new:
        if not _engine_instance.is_initialized:
            await _engine_instance.initialize()
        return _engine_instance

    _engine_instance = RAGEngine()
    await _engine_instance.initialize()
    return _engine_instance
