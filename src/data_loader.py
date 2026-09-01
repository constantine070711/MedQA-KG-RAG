"""
数据加载和预处理模块 (Phase 3 增强版)

职责:
    - 扫描 data/raw/{diseases,drugs,symptoms,guidelines}/ 目录
    - 支持多格式: .txt / .md / .json / .pdf (预留)
    - 解析 YAML frontmatter 提取元数据
    - 记录来源/类型/实体类别/时间戳
    - 为 LightRAG 索引提供纯文本输出

支持的格式:
    - .md  (Markdown + YAML frontmatter) ← 当前主力格式
    - .txt (纯文本, 向后兼容)
    - .json (结构化 JSON, 预留)
    - .pdf (需要 pymupdf/pdfplumber, 预留)

用法:
    from src.data_loader import DataLoader
    loader = DataLoader()
    documents = loader.load_all()
    for doc in loader.documents:
        print(f"{doc.name}: {doc.metadata}")
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field

from .config import get_config
from .metadata import DocumentMetadata, MetadataRegistry, parse_simple_yaml

logger = logging.getLogger(__name__)


# ============================================================
# 文档数据结构
# ============================================================

@dataclass
class Document:
    """
    统一文档数据结构

    Attributes:
        name: 文件名
        path: 绝对路径
        content: 原始文本内容（用于 LightRAG 索引）
        metadata: 文档元数据
        plain_text: 去 frontmatter 的纯文本正文
        format: 文件格式 (md/txt/json/pdf)
    """
    name: str
    path: str
    content: str              # 完整原始文本
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    plain_text: str = ""      # 去 frontmatter 的纯净正文
    format: str = "md"

    def __post_init__(self):
        if not self.plain_text:
            self.plain_text = self.content

    @property
    def char_count(self) -> int:
        """正文字符数"""
        return len(self.plain_text)

    @property
    def entity_type(self) -> str:
        """实体类别"""
        return self.metadata.entity_type

    @property
    def entity_name(self) -> str:
        """实体名称"""
        return self.metadata.name or self.name


# ============================================================
# DataLoader (Phase 3 增强版)
# ============================================================

class DataLoader:
    """
    医疗文档加载器 (Phase 3)

    扫描 data/raw/ 子目录:
        - diseases/    — 疾病知识
        - drugs/       — 药物知识
        - symptoms/    — 症状知识 (预留)
        - guidelines/  — 临床指南 (预留)

    支持格式:
        - .md  ← 推荐（YAML frontmatter + Markdown 正文）
        - .txt ← 向后兼容
        - .json ← 结构化数据
        - .pdf ← 预留

    Attributes:
        data_dir: data/raw/ 根目录
        documents: 已加载的文档列表
        metadata_registry: 元数据注册表
    """

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".json", ".pdf"}

    MIN_DOC_LENGTH = 50  # 最小文档长度

    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化数据加载器。

        Args:
            data_dir: data/raw/ 目录路径，默认从配置读取
        """
        cfg = get_config()
        self.data_dir = Path(data_dir) if data_dir else cfg.data_dir
        self.documents: list[Document] = []
        self.metadata_registry = MetadataRegistry()

    # ----------------------------------------------------------
    # 文件发现
    # ----------------------------------------------------------

    def discover_files(
        self,
        subdir: Optional[str] = None,
        extensions: Optional[set[str]] = None,
    ) -> list[Path]:
        """
        发现数据目录下的文档文件。

        支持递归扫描子目录（diseases/drugs/symptoms/guidelines）。

        Args:
            subdir: 限定子目录（如 "diseases"），None 表示全部
            extensions: 限定扩展名，默认全部支持格式

        Returns:
            匹配的文件路径列表（按路径排序）
        """
        exts = extensions or self.SUPPORTED_EXTENSIONS

        if subdir:
            search_dir = self.data_dir / subdir
        else:
            search_dir = self.data_dir

        if not search_dir.exists():
            logger.warning(f"目录不存在: {search_dir}")
            return []

        files = []
        for ext in exts:
            files.extend(search_dir.rglob(f"*{ext}"))

        files.sort()
        logger.info(
            f"发现 {len(files)} 个文档文件"
            + (f" (subdir={subdir})" if subdir else "")
        )
        return files

    def discover_by_entity_type(self) -> dict[str, list[Path]]:
        """
        按实体类别分组发现文件。

        Returns:
            {"disease": [Path, ...], "drug": [Path, ...], ...}
        """
        result = {}
        for entity_type in ["diseases", "drugs", "symptoms", "guidelines"]:
            subdir = self.data_dir / entity_type
            if subdir.exists():
                files = sorted(subdir.rglob("*.md")) + sorted(subdir.rglob("*.txt"))
                if files:
                    result[entity_type] = files
        return result

    # ----------------------------------------------------------
    # 格式解析器
    # ----------------------------------------------------------

    def _parse_markdown(self, file_path: Path) -> Optional[Document]:
        """
        解析 Markdown 文件（支持 YAML frontmatter）。

        YAML frontmatter 格式:
            ---
            source: "..."
            entity_type: "disease"
            name: "..."
            icd10_code: "..."
            ...
            ---
            # 标题
            正文...
        """
        content = file_path.read_text(encoding="utf-8")

        # 解析 YAML frontmatter
        metadata = DocumentMetadata(filename=file_path.name)
        plain_text = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                yaml_text = parts[1].strip()
                plain_text = parts[2].strip()

                # 复用独立 YAML 解析函数
                parsed = parse_simple_yaml(yaml_text)

                entity_type = parsed.get("entity_type", "")

                metadata = DocumentMetadata(
                    filename=file_path.name,
                    filepath=str(file_path.resolve()),
                    source=parsed.get("source", ""),
                    entity_type=entity_type,
                    name=parsed.get("name", file_path.stem),
                    aliases=parsed.get("aliases", []),
                    standard_code=(
                        parsed.get("icd10_code", "")
                        or parsed.get("atc_code", "")
                    ),
                    category=parsed.get("category", ""),
                    drug_type=parsed.get("drug_type", ""),
                    created_date=parsed.get("created_date", ""),
                    updated_date=parsed.get("updated_date", ""),
                    license=parsed.get("license", ""),
                    char_count=len(plain_text),
                )

                # 注册到全局注册表
                self.metadata_registry.add(metadata)

        return Document(
            name=file_path.name,
            path=str(file_path.resolve()),
            content=content,
            metadata=metadata,
            plain_text=plain_text,
            format="md",
        )

    def _parse_text(self, file_path: Path) -> Optional[Document]:
        """
        解析纯文本文件（向后兼容旧格式）。

        支持从注释行提取元数据:
            # @source: ...
            # @license: ...
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding="gbk")
            except Exception as e:
                logger.error(f"无法读取文件 {file_path.name}: {e}")
                return None

        # 提取简单元数据
        source = ""
        license_info = ""
        for line in content.split("\n")[:10]:
            line = line.strip()
            if line.startswith("# @source:"):
                source = line.replace("# @source:", "").strip()
            elif line.startswith("# @license:"):
                license_info = line.replace("# @license:", "").strip()

        metadata = DocumentMetadata(
            filename=file_path.name,
            filepath=str(file_path.resolve()),
            source=source,
            entity_type=self._infer_entity_type(file_path),
            license=license_info,
            char_count=len(content.strip()),
        )
        self.metadata_registry.add(metadata)

        return Document(
            name=file_path.name,
            path=str(file_path.resolve()),
            content=content,
            metadata=metadata,
            plain_text=content,
            format="txt",
        )

    def _parse_json(self, file_path: Path) -> Optional[Document]:
        """
        解析结构化 JSON 文件（预留格式）。

        期望格式:
            {
              "name": "...",
              "entity_type": "disease",
              "source": "...",
              "content": "...",
              "metadata": { ... }
            }
        """
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"JSON 解析失败: {file_path.name} — {e}")
            return None

        content = data.get("content", "") or json.dumps(data, ensure_ascii=False)

        meta = data.get("metadata", {})
        metadata = DocumentMetadata(
            filename=file_path.name,
            filepath=str(file_path.resolve()),
            source=data.get("source", ""),
            entity_type=data.get("entity_type", ""),
            name=data.get("name", ""),
            aliases=data.get("aliases", []),
            standard_code=data.get("icd10_code", "") or data.get("atc_code", ""),
            category=meta.get("category", ""),
            license=data.get("license", ""),
            char_count=len(content),
        )
        self.metadata_registry.add(metadata)

        return Document(
            name=file_path.name,
            path=str(file_path.resolve()),
            content=content,
            metadata=metadata,
            plain_text=content,
            format="json",
        )

    def _parse_pdf(self, file_path: Path) -> Optional[Document]:
        """
        解析 PDF 文件（预留）。

        后续可选依赖:
            - pymupdf (fitz)
            - pdfplumber
            - docling (LightRAG 内置)
        """
        logger.warning(
            f"PDF 解析尚未实现，跳过: {file_path.name}"
        )
        # TODO Phase 4+: 集成 pymupdf/pdfplumber 提取文本
        return None

    # ----------------------------------------------------------
    # 文件加载
    # ----------------------------------------------------------

    def load_file(self, file_path: Union[str, Path]) -> Optional[Document]:
        """
        加载单个文档文件（自动检测格式）。

        Args:
            file_path: 文件路径

        Returns:
            Document 对象，加载失败返回 None
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"不支持的文件格式: {ext} ({file_path.name})")
            return None

        # 按格式分发
        parsers = {
            ".md": self._parse_markdown,
            ".txt": self._parse_text,
            ".json": self._parse_json,
            ".pdf": self._parse_pdf,
        }

        parser = parsers.get(ext)
        if parser is None:
            return None

        doc = parser(file_path)
        if doc and doc.char_count >= self.MIN_DOC_LENGTH:
            logger.info(
                f"已加载: [{doc.entity_type or '?'}] {doc.entity_name} "
                f"({doc.char_count} 字符, {doc.format})"
            )
            return doc
        elif doc:
            logger.warning(f"文档过短，跳过: {file_path.name} ({doc.char_count} 字符)")

        return None

    def load_all(
        self,
        entity_types: Optional[list[str]] = None,
    ) -> list[Document]:
        """
        加载所有文档。

        Args:
            entity_types: 限定实体类别 ["diseases", "drugs"]，None 表示全部

        Returns:
            Document 列表
        """
        self.documents = []
        self.metadata_registry = MetadataRegistry()

        if entity_types:
            subdirs = entity_types
        else:
            subdirs = ["diseases", "drugs", "symptoms", "guidelines"]

        for subdir in subdirs:
            files = self.discover_files(subdir=subdir)
            for file_path in files:
                doc = self.load_file(file_path)
                if doc is not None:
                    self.documents.append(doc)

        total_chars = sum(d.char_count for d in self.documents)
        logger.info(
            f"共加载 {len(self.documents)} 个文档, "
            f"{total_chars} 字符, "
            f"{len(self.metadata_registry)} 条元数据"
        )

        return self.documents

    def load_for_lightrag(self) -> str:
        """
        加载所有文档并拼接为 LightRAG 索引用的单个文本。

        每个文档的 plain_text 用双换行分隔。
        元数据通过 YAML frontmatter 已自动包含在 content 中。

        Returns:
            所有文档内容的拼接文本
        """
        docs = self.load_all()
        texts = []

        for doc in docs:
            # 对于 md 文件，保留完整的 content（含 frontmatter）
            # LightRAG 可以从中提取更丰富的语义
            if doc.format == "md":
                texts.append(doc.content)
            else:
                texts.append(doc.plain_text)

        return "\n\n".join(texts)

    def load_for_lightrag_separate(self) -> list[str]:
        """
        逐个文档加载，用于增量索引。

        Returns:
            每个文档完整内容的列表
        """
        docs = self.load_all()
        return [doc.content if doc.format == "md" else doc.plain_text for doc in docs]

    # ----------------------------------------------------------
    # 工具函数
    # ----------------------------------------------------------

    def _infer_entity_type(self, file_path: Path) -> str:
        """从文件路径推断实体类型"""
        parent = file_path.parent.name
        type_map = {
            "diseases": "disease",
            "drugs": "drug",
            "symptoms": "symptom",
            "guidelines": "guideline",
        }
        return type_map.get(parent, "")

    def get_stats(self) -> dict:
        """返回数据加载统计"""
        if not self.documents:
            self.load_all()

        by_type = {}
        for doc in self.documents:
            t = doc.entity_type or "unknown"
            by_type[t] = by_type.get(t, 0) + 1

        by_format = {}
        for doc in self.documents:
            fmt = doc.format
            by_format[fmt] = by_format.get(fmt, 0) + 1

        return {
            "data_dir": str(self.data_dir),
            "total_documents": len(self.documents),
            "total_characters": sum(d.char_count for d in self.documents),
            "by_entity_type": by_type,
            "by_format": by_format,
            "metadata_records": len(self.metadata_registry),
            "metadata_summary": self.metadata_registry.summary(),
            "documents": [
                {
                    "name": d.name,
                    "entity_type": d.entity_type,
                    "entity_name": d.entity_name,
                    "char_count": d.char_count,
                    "format": d.format,
                    "source": d.metadata.source,
                    "license": d.metadata.license,
                }
                for d in self.documents
            ],
        }

    def build_metadata_index(self, output_path: Optional[str] = None) -> MetadataRegistry:
        """
        构建并导出元数据索引。

        Args:
            output_path: JSON 输出路径

        Returns:
            MetadataRegistry 实例
        """
        if not self.documents:
            self.load_all()

        if output_path:
            self.metadata_registry.to_json(output_path)

        return self.metadata_registry
