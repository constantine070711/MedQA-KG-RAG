"""
Embedding 模型封装模块

职责:
    - 加载本地/在线 Embedding 模型 (bge-small-zh-v1.5)
    - 提供符合 LightRAG 要求的 embedding_func 回调
    - 统一管理模型生命周期

LightRAG 要求:
    - embedding_func 必须返回 numpy array（有 .size 属性）
    - 不能返回 list（会导致 AttributeError）

用法:
    from src.embedder import Embedder
    embedder = Embedder()
    await embedder.embedding_func(["文本1", "文本2"])
"""

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from .config import get_config

logger = logging.getLogger(__name__)


class Embedder:
    """
    Embedding 模型封装类

    支持:
        - 本地预下载模型优先加载
        - 回退到 HuggingFace 在线加载
        - 自动 normalize（余弦相似度）

    Attributes:
        model_name: HuggingFace 模型 ID
        model_path: 实际加载路径（本地或远程）
        dim: 向量维度
        model: SentenceTransformer 实例
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        初始化 Embedding 模型。

        Args:
            model_name: 可选覆盖默认模型名称
        """
        cfg = get_config()

        self.model_name = model_name or cfg.embedding_model_name
        self.model_path = cfg.embedding_local_path
        self.dim = cfg.embedding_dim

        self._model: Optional[SentenceTransformer] = None

    # ----------------------------------------------------------
    # 模型加载
    # ----------------------------------------------------------

    def load(self) -> "Embedder":
        """
        加载 Embedding 模型到内存。

        Returns:
            self，支持链式调用
        """
        if self._model is not None:
            logger.info("Embedding 模型已在内存中，跳过加载")
            return self

        is_local = self.model_path != self.model_name
        source = "本地预下载" if is_local else "HuggingFace 在线"
        logger.info(f"加载 Embedding 模型 ({source}): {self.model_path}")

        self._model = SentenceTransformer(self.model_path)
        logger.info(f"Embedding 模型加载成功 (维度: {self.dim})")

        return self

    @property
    def model(self) -> SentenceTransformer:
        """获取底层 SentenceTransformer 模型（自动懒加载）"""
        if self._model is None:
            self.load()
        return self._model

    # ----------------------------------------------------------
    # LightRAG 兼容接口
    # ----------------------------------------------------------

    async def embedding_func(self, texts: list[str]):
        """
        LightRAG 要求的 embedding 回调函数。

        将文本列表转换为归一化的向量数组。

        Args:
            texts: 待编码的文本列表

        Returns:
            numpy.ndarray，形状 (len(texts), dim)，
            已 L2 归一化，可直接用于余弦相似度计算。

        Note:
            必须返回 numpy array（有 .size 属性），
            LightRAG 源码依赖此属性。不要调用 .tolist()。
        """
        if self._model is None:
            self.load()

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings  # numpy array，不要 .tolist()

    # ----------------------------------------------------------
    # 信息查询
    # ----------------------------------------------------------

    def get_info(self) -> dict:
        """返回模型信息"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "dim": self.dim,
            "is_local": self.model_path != self.model_name,
            "is_loaded": self._model is not None,
        }


# ============================================================
# 模块级便捷工厂
# ============================================================

_embedder_instance: Optional[Embedder] = None


def get_embedder(reload: bool = False) -> Embedder:
    """
    获取全局 Embedder 单例。

    Args:
        reload: 是否强制重建实例

    Returns:
        Embedder 实例（已加载模型）
    """
    global _embedder_instance

    if _embedder_instance is not None and not reload:
        return _embedder_instance

    _embedder_instance = Embedder()
    _embedder_instance.load()
    return _embedder_instance
