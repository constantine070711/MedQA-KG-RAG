"""
统一实体数据模型 — Pipeline 中间表示 (IR)

职责:
    - 定义 Pipeline 各阶段之间的统一数据结构
    - 所有 Adapter 输出原始实体 → Transformer 标准化为 UnifiedEntity → Writer 写入
    - 完整保留数据溯源链 (provenance)
    - 记录质量检查结果

设计原则:
    - UnifiedEntity 是 Pipeline 专用的中间表示
    - 不替代 DocumentMetadata (文件级元数据)
    - 不替代 MedicalEntity (领域建模)
    - 作为"数据源 → 内部存储"之间的标准化桥梁

用法:
    from src.pipeline.unified_entity import UnifiedEntity, ProvenanceEntry, QualityReport
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ============================================================
# 溯源信息
# ============================================================

@dataclass
class ProvenanceEntry:
    """
    数据来源溯源链中的单条记录。

    每个 UnifiedEntity 可以有多条溯源信息（多个数据源贡献），
    每条记录独立保留 source / license / adapter 信息。

    Attributes:
        source_name: 数据来源名称 (如 "CBLUE CMeEE-V2" / "Manual Curation")
        source_url: 数据来源 URL
        license: 许可证 (如 "CC BY-NC-SA 4.0" / "Public Domain")
        adapter: 使用的适配器名称
        original_id: 原始数据中的 ID
        ingest_date: 导入日期
        notes: 备注
    """
    source_name: str = ""
    source_url: str = ""
    license: str = ""
    adapter: str = ""
    original_id: str = ""
    ingest_date: str = ""
    notes: str = ""


# ============================================================
# 质量报告
# ============================================================

class QualityLevel(Enum):
    """质量等级"""
    PASS = "pass"        # 通过所有检查
    WARN = "warn"        # 有警告但可继续
    FAIL = "fail"        # 未通过，需修复


@dataclass
class QualityReport:
    """
    单个实体的数据质量检查报告。

    Attributes:
        level: 综合质量等级
        errors: 错误列表 (必须修复)
        warnings: 警告列表 (建议修复)
        checks: 已执行的检查项及其结果
    """
    level: QualityLevel = QualityLevel.PASS
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """是否通过质量检查（允许有警告）"""
        return self.level != QualityLevel.FAIL

    @property
    def has_issues(self) -> bool:
        """是否有任何问题"""
        return len(self.errors) > 0 or len(self.warnings) > 0


# ============================================================
# 别名条目
# ============================================================

@dataclass
class AliasEntry:
    """
    结构化别名。

    Attributes:
        name: 别名文本
        language: 语言 (zh/en/la)
        source: 别名来源
    """
    name: str
    language: str = "zh"
    source: str = ""


# ============================================================
# 标准编码条目
# ============================================================

@dataclass
class CodeEntry:
    """
    标准化编码（ICD-10 / ATC / SNOMED CT）。

    Attributes:
        system: 编码系统 (ICD-10 / ATC / SNOMED_CT)
        code: 编码值
        description: 编码描述
    """
    system: str = ""
    code: str = ""
    description: str = ""


# ============================================================
# 统一实体 (核心)
# ============================================================

@dataclass
class UnifiedEntity:
    """
    统一实体 — Pipeline 的核心中间表示。

    所有数据源（.md / JSON / CSV / API）通过 Adapter 转换为
    此格式，经 Transformer 标准化后，由 Writer 写入目标存储。

    Attributes:
        entity_id: 全局唯一标识 (格式: "source:type:name" 如 "manual:disease:2型糖尿病")
        name: 规范化实体名称 (中文主名)
        entity_type: 实体类型 (disease/drug/symptom/examination/department/guideline)
        aliases: 标准化别名列表
        standard_codes: 标准编码列表 (ICD-10/ATC)
        category: 分类
        description: 简要描述
        properties: 扩展属性
        relations: 关联的实体名列表 [{subject, relation, object}]
        provenance: 溯源信息列表
        quality: 质量报告
        raw_text: 原始文本 (用于 LightRAG 索引)
    """
    entity_id: str = ""
    name: str = ""
    entity_type: str = ""  # disease/drug/symptom/examination/department/guideline

    aliases: list[AliasEntry] = field(default_factory=list)
    standard_codes: list[CodeEntry] = field(default_factory=list)

    category: str = ""
    description: str = ""
    properties: dict = field(default_factory=dict)

    relations: list[dict] = field(default_factory=list)
    # relations: [{"subject": str, "relation": str, "object": str}, ...]

    provenance: list[ProvenanceEntry] = field(default_factory=list)
    quality: QualityReport = field(default_factory=QualityReport)

    raw_text: str = ""  # 原始文本（用于 LightRAG 索引）

    @property
    def all_names(self) -> list[str]:
        """所有名称（主名 + 所有别名）"""
        alias_names = [a.name for a in self.aliases]
        return [self.name] + alias_names

    @property
    def primary_code(self) -> str:
        """主编码值"""
        if self.standard_codes:
            return self.standard_codes[0].code
        return ""

    @property
    def is_valid(self) -> bool:
        """快捷访问质量检查结果"""
        return self.quality.is_valid

    def to_dict(self) -> dict:
        """转换为可序列化字典"""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "aliases": [{"name": a.name, "language": a.language} for a in self.aliases],
            "standard_codes": [
                {"system": c.system, "code": c.code} for c in self.standard_codes
            ],
            "category": self.category,
            "description": self.description,
            "properties": self.properties,
            "relations": self.relations,
            "provenance": [
                {
                    "source_name": p.source_name,
                    "license": p.license,
                    "adapter": p.adapter,
                }
                for p in self.provenance
            ],
            "quality": {
                "level": self.quality.level.value,
                "errors": self.quality.errors,
                "warnings": self.quality.warnings,
            },
        }


# ============================================================
# Pipeline 运行报告
# ============================================================

@dataclass
class PipelineReport:
    """
    单次 Pipeline 运行的完整报告。

    Attributes:
        source_name: 数据源名称
        adapter_name: 使用的适配器
        total_input: 输入实体数
        total_valid: 通过校验的实体数
        total_invalid: 未通过校验的实体数
        total_with_warnings: 有警告的实体数
        total_duplicates_merged: 已合并的重复实体数
        total_relations: 关系总数
        by_entity_type: 按类型的实体分布
        quality_issues: 质量问题汇总
        duration_seconds: 耗时（秒）
    """
    source_name: str = ""
    adapter_name: str = ""

    total_input: int = 0
    total_valid: int = 0
    total_invalid: int = 0
    total_with_warnings: int = 0
    total_duplicates_merged: int = 0
    total_relations: int = 0

    by_entity_type: dict = field(default_factory=dict)
    quality_issues: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def summary(self) -> str:
        """可读的摘要报告"""
        lines = [
            f"Pipeline Report: {self.source_name} (adapter={self.adapter_name})",
            f"  输入: {self.total_input} | 有效: {self.total_valid} | 无效: {self.total_invalid}",
            f"  警告: {self.total_with_warnings} | 重复合并: {self.total_duplicates_merged}",
            f"  关系: {self.total_relations}",
            f"  类型分布: {self.by_entity_type}",
            f"  耗时: {self.duration_seconds:.1f}s",
        ]
        if self.quality_issues:
            lines.append(f"  质量问题 ({len(self.quality_issues)}):")
            for issue in self.quality_issues[:10]:
                lines.append(f"    - {issue}")
        return "\n".join(lines)
