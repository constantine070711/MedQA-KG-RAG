"""
医疗知识数据管道 (Medical Knowledge Data Pipeline)

架构:
    Adapter → Transformer → Validator → Writer

组件:
    adapters/      — 数据源适配器 (Markdown / CBLUE / ...)
    transformers/  — 数据转换器 (Cleaner / Normalizer / Deduplicator / Validator)
    writers/       — 数据写入器 (EntityWriter / MetadataWriter)
    orchestrator   — Pipeline 编排器
    unified_entity — 统一实体数据模型
"""

from .unified_entity import (
    UnifiedEntity,
    ProvenanceEntry,
    QualityReport,
    QualityLevel,
    AliasEntry,
    CodeEntry,
    PipelineReport,
)
