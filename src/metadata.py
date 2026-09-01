"""
文档元数据管理模块

职责:
    - 定义文档元数据标准结构
    - 管理元数据注册表和索引
    - 支持元数据导入/导出（JSON）
    - 为数据溯源和论文实验记录提供基础

元数据字段:
    - source: 数据来源
    - entity_type: 实体类别 (disease/drug/symptom/guideline)
    - name: 实体名称
    - icd10_code / atc_code: 标准编码（可选）
    - category: 分类
    - created_date / updated_date: 时间戳
    - license: 许可证信息
    - aliases: 别名列表

用法:
    from src.metadata import DocumentMetadata, MetadataRegistry

    registry = MetadataRegistry()
    registry.load_from_directory("data/raw")
    print(registry.summary())
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import date

logger = logging.getLogger(__name__)


# ============================================================
# 简易 YAML 解析 (独立函数，供 metadata 和 data_loader 共用)
# ============================================================

def parse_simple_yaml(text: str) -> dict:
    """
    简易 YAML 解析器（仅支持顶层标量和单层列表）。

    避免引入 PyYAML 依赖，满足 frontmatter 解析需求。
    支持格式:
        key: value
        key: "quoted value"
        key:
          - item1
          - item2
    """
    result = {}
    current_list_key = None
    current_list = []

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if ":" in stripped and not stripped.startswith("- "):
            # 处理之前的列表
            if current_list_key:
                result[current_list_key] = current_list
                current_list_key = None
                current_list = []

            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value:
                result[key] = value
            else:
                # 可能是列表的开始，记录 key
                current_list_key = key

        elif stripped.startswith("- "):
            item = stripped[2:].strip().strip('"').strip("'")
            if current_list_key:
                current_list.append(item)

    # 处理最后可能遗留的列表
    if current_list_key:
        result[current_list_key] = current_list

    return result


# ============================================================
# 文档元数据
# ============================================================

@dataclass
class DocumentMetadata:
    """
    单个知识片段的元数据

    对应 data/raw/ 下每个 .md 文件的 YAML frontmatter。

    Attributes:
        filename: 文件名
        filepath: 文件绝对路径
        source: 数据来源
        entity_type: 实体类别
        name: 实体名称（中文）
        aliases: 别名列表
        standard_code: 标准编码 (ICD-10 / ATC)
        category: 疾病/药物分类
        drug_type: 药物类型（仅 drug 类）
        created_date: 创建日期
        updated_date: 更新日期
        license: 许可证信息
        char_count: 文件字符数
    """
    filename: str
    filepath: str = ""
    source: str = ""
    entity_type: str = ""       # disease | drug | symptom | guideline
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    standard_code: str = ""     # ICD-10 for disease, ATC for drug
    category: str = ""
    drug_type: str = ""         # 仅 drug 类型使用
    created_date: str = ""
    updated_date: str = ""
    license: str = ""
    char_count: int = 0

    # 扩展字段（预留）
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典（用于 JSON 序列化）"""
        d = asdict(self)
        # 移除空值以保持简洁
        return {k: v for k, v in d.items() if v not in ("", [], {}, 0, None)}

    def to_neo4j_properties(self) -> dict:
        """
        转换为 Neo4j 节点属性字典。

        用于后续 Cypher CREATE 语句的 SET 子句。
        """
        props = {
            "name": self.name,
            "entity_type": self.entity_type,
            "source": self.source,
            "aliases": self.aliases,
        }
        if self.standard_code:
            props["standard_code"] = self.standard_code
        if self.category:
            props["category"] = self.category
        if self.drug_type:
            props["drug_type"] = self.drug_type
        if self.license:
            props["license"] = self.license
        return props


# ============================================================
# 元数据注册表
# ============================================================

class MetadataRegistry:
    """
    元数据注册表

    管理所有知识片段的元数据，支持:
        - 从 data/raw/ 目录扫描加载
        - 按实体类型/名称/编码查询
        - 导出 JSON 索引
        - 生成统计摘要
    """

    def __init__(self):
        self._records: list[DocumentMetadata] = []

    # ----------------------------------------------------------
    # 加载
    # ----------------------------------------------------------

    def add(self, metadata: DocumentMetadata) -> None:
        """添加一条元数据记录"""
        self._records.append(metadata)

    def load_from_directory(self, data_dir: str) -> int:
        """
        扫描 data/raw/ 目录，从 .md 文件的 YAML frontmatter 加载元数据。

        Args:
            data_dir: data/raw/ 目录路径

        Returns:
            加载的记录数
        """
        data_path = Path(data_dir)

        if not data_path.exists():
            logger.warning(f"目录不存在: {data_dir}")
            return 0

        count = 0
        for md_file in sorted(data_path.rglob("*.md")):
            try:
                metadata = self._parse_frontmatter(md_file)
                if metadata:
                    self.add(metadata)
                    count += 1
            except Exception as e:
                logger.warning(f"解析元数据失败: {md_file.name} — {e}")

        logger.info(f"从 {data_dir} 加载了 {count} 条元数据记录")
        return count

    def _parse_frontmatter(self, file_path: Path) -> Optional[DocumentMetadata]:
        """
        解析 Markdown 文件的 YAML frontmatter。

        期望格式:
            ---
            source: "..."
            entity_type: "disease"
            name: "2型糖尿病"
            icd10_code: "E11"
            ...
            ---
            # 标题
            正文内容...
        """
        content = file_path.read_text(encoding="utf-8")

        if not content.startswith("---"):
            return None

        # 提取 YAML frontmatter
        parts = content.split("---", 2)
        if len(parts) < 3:
            return None

        yaml_text = parts[1].strip()
        body = parts[2]

        # 简易 YAML 解析（避免引入 PyYAML 依赖）
        frontmatter = parse_simple_yaml(yaml_text)

        # 构建元数据
        entity_type = frontmatter.get("entity_type", "")
        standard_code = (
            frontmatter.get("icd10_code", "")
            or frontmatter.get("atc_code", "")
        )

        return DocumentMetadata(
            filename=file_path.name,
            filepath=str(file_path.resolve()),
            source=frontmatter.get("source", ""),
            entity_type=entity_type,
            name=frontmatter.get("name", file_path.stem),
            aliases=frontmatter.get("aliases", []),
            standard_code=standard_code,
            category=frontmatter.get("category", ""),
            drug_type=frontmatter.get("drug_type", ""),
            created_date=frontmatter.get("created_date", ""),
            updated_date=frontmatter.get("updated_date", ""),
            license=frontmatter.get("license", ""),
            char_count=len(body.strip()),
        )

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------

    def get_by_type(self, entity_type: str) -> list[DocumentMetadata]:
        """按实体类型过滤"""
        return [r for r in self._records if r.entity_type == entity_type]

    def get_by_name(self, name: str) -> Optional[DocumentMetadata]:
        """按名称精确查找"""
        for r in self._records:
            if r.name == name or name in r.aliases:
                return r
        return None

    def get_by_code(self, code: str) -> Optional[DocumentMetadata]:
        """按标准编码查找"""
        for r in self._records:
            if r.standard_code == code:
                return r
        return None

    def get_all(self) -> list[DocumentMetadata]:
        """获取所有记录"""
        return list(self._records)

    # ----------------------------------------------------------
    # 导出
    # ----------------------------------------------------------

    def to_json(self, filepath: str, pretty: bool = True) -> None:
        """
        导出元数据索引为 JSON 文件。

        Args:
            filepath: 输出路径
            pretty: 是否美化格式
        """
        data = {
            "generated_at": str(date.today()),
            "total_records": len(self._records),
            "records": [r.to_dict() for r in self._records],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(data, f, ensure_ascii=False, indent=2)
            else:
                json.dump(data, f, ensure_ascii=False)

        logger.info(f"元数据索引已导出: {filepath} ({len(self._records)} 条记录)")

    def to_neo4j_cypher(self, filepath: str) -> None:
        """
        导出为 Neo4j Cypher 语句文件。

        生成的 Cypher 使用 MERGE 避免重复创建节点。

        Args:
            filepath: 输出路径 (.cypher 文件)
        """
        lines = [
            "// Neo4j 知识图谱节点导入脚本",
            f"// 自动生成于: {date.today()}",
            f"// 节点数: {len(self._records)}",
            "",
        ]

        for r in self._records:
            label = r.entity_type.upper() if r.entity_type else "Entity"
            props = r.to_neo4j_properties()

            # 构建属性字符串
            props_str = ", ".join(
                f"{k}: '{str(v).replace(chr(39), chr(39)+chr(39))}'"
                if isinstance(v, str)
                else f"{k}: {json.dumps(v, ensure_ascii=False)}"
                for k, v in props.items()
            )

            lines.append(f"MERGE (n:{label} {{{props_str}}})")
            lines.append(f"  SET n.updated_at = date()")
            lines.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Neo4j Cypher 已导出: {filepath}")

    # ----------------------------------------------------------
    # 统计
    # ----------------------------------------------------------

    def summary(self) -> dict:
        """返回元数据摘要统计"""
        by_type = {}
        for r in self._records:
            t = r.entity_type or "unknown"
            by_type[t] = by_type.get(t, 0) + 1

        by_source = {}
        for r in self._records:
            s = r.source or "unknown"
            by_source[s] = by_source.get(s, 0) + 1

        total_chars = sum(r.char_count for r in self._records)

        return {
            "total_records": len(self._records),
            "total_characters": total_chars,
            "by_entity_type": by_type,
            "by_source": by_source,
            "records": [
                {
                    "filename": r.filename,
                    "name": r.name,
                    "entity_type": r.entity_type,
                    "standard_code": r.standard_code,
                    "char_count": r.char_count,
                }
                for r in self._records
            ],
        }

    def __len__(self) -> int:
        return len(self._records)


# ============================================================
# 模块级便捷函数
# ============================================================

def build_metadata_index(
    data_dir: str,
    output_path: Optional[str] = None,
) -> MetadataRegistry:
    """
    从 data/raw/ 目录构建元数据索引。

    Args:
        data_dir: data/raw/ 目录路径
        output_path: JSON 索引输出路径（可选）

    Returns:
        填充好的 MetadataRegistry
    """
    registry = MetadataRegistry()
    registry.load_from_directory(data_dir)

    if output_path:
        registry.to_json(output_path)

    return registry
