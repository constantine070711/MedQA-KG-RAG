"""
Markdown 知识文件适配器

将 data/raw/ 下的 .md 文件（YAML frontmatter + Markdown 正文）
解析为 RawEntity dict，供后续 Transformer 处理。

复用现有 DataLoader 的 frontmatter 解析能力，不重复实现 YAML 解析。

用法:
    adapter = MarkdownAdapter(data_dir="data/raw")
    raw_entities = adapter.ingest()  # 返回 14 个 RawEntity dict
"""

import logging
from pathlib import Path
from typing import Optional
from datetime import date

from .base_adapter import DataSourceAdapter

logger = logging.getLogger(__name__)


class MarkdownAdapter(DataSourceAdapter):
    """
    现有 data/raw/ 目录的 Markdown 知识文件适配器。

    复用 src.data_loader.DataLoader 的 frontmatter 解析逻辑，
    将 Document 对象转换为 Pipeline 标准的 RawEntity dict。

    Input:  data/raw/diseases/*.md, data/raw/drugs/*.md
    Output: list[RawEntity dict] — 每个实体一条
    """

    name: str = "markdown"
    description: str = "Parse data/raw/ .md files with YAML frontmatter"
    supported_formats: list[str] = [".md"]

    # entity_type 映射：目录名 → 实体类型
    ENTITY_TYPE_MAP = {
        "diseases": "disease",
        "drugs": "drug",
        "symptoms": "symptom",
        "guidelines": "guideline",
    }

    # 编码系统映射
    CODE_SYSTEM_MAP = {
        "disease": "ICD-10",
        "drug": "ATC",
    }

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(source_path=data_dir)
        self._data_dir = Path(data_dir) if data_dir else None

    def ingest(self, source_path: Optional[str] = None) -> list[dict]:
        """
        扫描 data/raw/ 目录，将每个 .md 文件转换为 RawEntity dict。

        Args:
            source_path: 数据目录路径，None 使用构造函数传入的 data_dir

        Returns:
            RawEntity dict 列表
        """
        from src.data_loader import DataLoader

        data_dir = source_path or (str(self._data_dir) if self._data_dir else None)
        if not data_dir:
            raise ValueError("data_dir is required for MarkdownAdapter")

        loader = DataLoader(data_dir=data_dir)
        documents = loader.load_all()

        raw_entities = []
        for doc in documents:
            raw = self._document_to_raw_entity(doc)
            if raw:
                raw_entities.append(raw)

        logger.info(
            f"MarkdownAdapter: 解析 {len(documents)} 个文档 → {len(raw_entities)} 个原始实体"
        )
        return raw_entities

    def _document_to_raw_entity(self, doc) -> Optional[dict]:
        """
        将 DataLoader.Document 转换为 RawEntity dict。

        从 YAML frontmatter 和 Markdown 正文中提取结构化信息，
        同时保留原始文本供 LightRAG 索引使用。

        Args:
            doc: DataLoader.Document 对象

        Returns:
            RawEntity dict，转换失败返回 None
        """
        meta = doc.metadata

        # 确定编码系统和编码值
        entity_type = meta.entity_type or self._infer_type_from_path(doc.path)
        code_system = self.CODE_SYSTEM_MAP.get(entity_type, "")
        standard_code = meta.standard_code or ""

        # 提取别名
        aliases = list(meta.aliases) if meta.aliases else []

        # 从正文中提取第一段作为描述
        description = self._extract_description(doc.plain_text)

        # 从正文中提取关系（基于关键词匹配）
        relations = self._extract_relations(doc.plain_text, meta.name, entity_type)

        # 构建来源名称
        source_name = f"Manual Curation — {entity_type.title()} Knowledge"
        if meta.category:
            source_name += f" ({meta.category})"

        raw = {
            "name": meta.name or doc.entity_name,
            "entity_type": entity_type,
            "aliases": aliases,
            "standard_code": standard_code,
            "code_system": code_system,
            "category": meta.category or "",
            "description": description,
            "raw_text": doc.content,
            "source_name": source_name,
            "source_url": "",
            "license": meta.license or "Public Domain",
            "relations": relations,
            "properties": {
                "filename": meta.filename,
                "created_date": meta.created_date or str(date.today()),
                "updated_date": meta.updated_date or str(date.today()),
                "char_count": meta.char_count,
                "drug_type": getattr(meta, "drug_type", ""),
            },
        }
        return raw

    @staticmethod
    def _infer_type_from_path(filepath: str) -> str:
        """从文件路径推断实体类型"""
        path = Path(filepath)
        parent = path.parent.name
        type_map = {
            "diseases": "disease",
            "drugs": "drug",
            "symptoms": "symptom",
            "guidelines": "guideline",
        }
        return type_map.get(parent, "unknown")

    @staticmethod
    def _extract_description(text: str, max_len: int = 200) -> str:
        """
        从 Markdown 正文提取第一段作为简要描述。

        跳过标题行（# 开头），取第一个有意义的段落。
        """
        lines = text.strip().split("\n")
        content_lines = []
        for line in lines:
            stripped = line.strip()
            # 跳过标题、空行、列表项
            if not stripped or stripped.startswith("#"):
                continue
            content_lines.append(stripped)

        desc = " ".join(content_lines)
        if len(desc) > max_len:
            desc = desc[:max_len] + "..."
        return desc

    @staticmethod
    def _extract_relations(
        text: str, entity_name: str, entity_type: str
    ) -> list[dict]:
        """
        从正文中提取潜在关系（基于关键词）。

        这是一个轻量级启发式方法。完整的医疗关系抽取
        应由 CBLUE CMeIE-V2 适配器提供。

        Args:
            text: 正文文本
            entity_name: 实体名称
            entity_type: 实体类型

        Returns:
            relations 列表 [{"subject": str, "relation": str, "object": str}]
        """
        relations = []

        if entity_type == "disease":
            # 尝试从"症状"章节提取
            if "症状" in text:
                relations.append({
                    "subject": entity_name,
                    "relation": "HAS_SYMPTOM",
                    "object": "见正文症状章节",
                })
            if "并发症" in text:
                relations.append({
                    "subject": entity_name,
                    "relation": "HAS_COMPLICATION",
                    "object": "见正文并发症章节",
                })
            if "治疗" in text:
                relations.append({
                    "subject": entity_name,
                    "relation": "TREATED_BY",
                    "object": "见正文治疗方法章节",
                })

        elif entity_type == "drug":
            if "适应症" in text:
                relations.append({
                    "subject": entity_name,
                    "relation": "TREATS",
                    "object": "见正文适应症章节",
                })
            if "不良反应" in text:
                relations.append({
                    "subject": entity_name,
                    "relation": "HAS_ADVERSE_REACTION",
                    "object": "见正文不良反应章节",
                })
            if "禁忌" in text:
                relations.append({
                    "subject": entity_name,
                    "relation": "CONTRAINDICATED_FOR",
                    "object": "见正文禁忌症章节",
                })

        return relations
