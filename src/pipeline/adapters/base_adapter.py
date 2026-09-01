"""
数据源适配器抽象基类

职责:
    - 定义所有数据源适配器的统一接口
    - 每种外部数据格式对应一个 Adapter 子类
    - Adapter 负责解析原始数据 → 生成 RawEntity dict
    - 不负责清洗/标准化/去重（这些由 Transformer 处理）

子类化:
    class CBLUEAdapter(DataSourceAdapter):
        def ingest(self, source_path: str) -> list[dict]:
            ...

    class CustomAdapter(DataSourceAdapter):
        def ingest(self, source_path: str) -> list[dict]:
            ...

用法:
    adapter = MarkdownAdapter(data_dir="data/raw")
    raw_entities = adapter.ingest()
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DataSourceAdapter(ABC):
    """
    数据源适配器抽象基类。

    每个子类实现一种数据格式/来源的解析逻辑。
    输出的是 RawEntity (dict)，由后续 Transformer 统一处理。

    Attributes:
        name: 适配器名称 (如 "markdown", "cblue-cmeee")
        description: 适配器描述
        supported_formats: 支持的输入格式列表
    """

    name: str = "base"
    description: str = "Abstract data source adapter"
    supported_formats: list[str] = []

    def __init__(self, source_path: Optional[str] = None):
        self.source_path = Path(source_path) if source_path else None

    @abstractmethod
    def ingest(self, source_path: Optional[str] = None) -> list[dict]:
        """
        从数据源读取并解析原始数据。

        Args:
            source_path: 数据源路径（文件或目录），None 使用构造函数传入的路径

        Returns:
            RawEntity dict 列表，每个 dict 至少包含:
                {
                    "name": str,           # 实体名称
                    "entity_type": str,    # 实体类型
                    "aliases": [str],      # 别名
                    "standard_code": str,  # 标准编码
                    "code_system": str,    # 编码系统 (ICD-10/ATC)
                    "category": str,       # 分类
                    "description": str,    # 描述
                    "raw_text": str,       # 原始文本（用于索引）
                    "source_name": str,    # 数据来源名称
                    "source_url": str,     # 数据来源 URL
                    "license": str,        # 许可证
                    "relations": [dict],   # 关联关系
                    "properties": dict,    # 扩展属性
                }
        """
        ...

    def get_info(self) -> dict:
        """返回适配器信息"""
        return {
            "name": self.name,
            "description": self.description,
            "supported_formats": self.supported_formats,
            "source_path": str(self.source_path) if self.source_path else None,
        }
