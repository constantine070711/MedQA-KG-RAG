"""
Pipeline 编排器

职责:
    - 编排完整的数据导入流程: Adapter → Transformer → Validator → Writer
    - 记录每一步的统计信息
    - 生成 Pipeline 运行报告

流程:
    1. INGEST:    Adapter 读取原始数据 → List[RawEntity dict]
    2. CLEAN:     TextCleaner 清洗文本
    3. NORMALIZE: EntityNormalizer 标准化 → List[UnifiedEntity]
    4. DEDUP:     EntityDeduplicator 去重
    5. VALIDATE:  DataValidator 质量校验
    6. WRITE:     EntityWriter → EntityRegistry + entities.json
    7. WRITE:     MetadataWriter → metadata/index.json

用法:
    from src.pipeline.orchestrator import PipelineOrchestrator
    orch = PipelineOrchestrator()
    report = orch.run(source="markdown")
    print(report.summary())
"""

import time
import logging
from pathlib import Path
from typing import Optional

from .adapters import MarkdownAdapter
from .transformers import (
    TextCleaner,
    EntityNormalizer,
    EntityDeduplicator,
    DataValidator,
)
from .writers import EntityWriter, MetadataWriter
from .unified_entity import PipelineReport

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Pipeline 编排器。

    管理从数据源到存储的完整流程。
    每个阶段独立运行，有明确的输入/输出类型。
    """

    def __init__(self, project_root: Optional[str] = None):
        """
        Args:
            project_root: 项目根目录路径
        """
        if project_root:
            self.project_root = Path(project_root)
        else:
            # 自动检测
            from src.config import PROJECT_ROOT
            self.project_root = PROJECT_ROOT

    def run(
        self,
        source: str = "markdown",
        source_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> PipelineReport:
        """
        运行完整 Pipeline。

        Args:
            source: 数据源类型 ("markdown")
            source_path: 数据源路径
            output_dir: 输出目录

        Returns:
            PipelineReport
        """
        start_time = time.time()
        report = PipelineReport(source_name=source, adapter_name=source)

        logger.info(f"Pipeline 启动: source={source}")

        # ---- Stage 1: INGEST ----
        logger.info("[Stage 1/6] INGEST — 加载原始数据...")
        raw_entities = self._ingest(source, source_path)
        report.total_input = len(raw_entities)
        logger.info(f"  加载 {len(raw_entities)} 个原始实体")

        if not raw_entities:
            logger.warning("无数据可处理，Pipeline 终止")
            report.duration_seconds = time.time() - start_time
            return report

        # ---- Stage 2: CLEAN ----
        logger.info("[Stage 2/6] CLEAN — 清洗文本...")
        cleaner = TextCleaner()
        raw_entities = [cleaner.clean_entity(re) for re in raw_entities]
        logger.info(f"  清洗完成: {len(raw_entities)} 个实体")

        # ---- Stage 3: NORMALIZE ----
        logger.info("[Stage 3/6] NORMALIZE — 标准化实体...")
        normalizer = EntityNormalizer(source_prefix=source)
        unified_entities = [
            normalizer.normalize(re) for re in raw_entities
        ]
        logger.info(f"  标准化完成: {len(unified_entities)} 个 UnifiedEntity")

        # ---- Stage 4: DEDUP ----
        logger.info("[Stage 4/6] DEDUP — 去重...")
        deduplicator = EntityDeduplicator()
        unified_entities, dedup_report = deduplicator.deduplicate(unified_entities)
        report.total_duplicates_merged = dedup_report.duplicates_merged
        for pair in dedup_report.potential_duplicates:
            report.quality_issues.append(f"潜在重复: '{pair[0]}' ↔ '{pair[1]}'")
        logger.info(f"  {dedup_report.summary()}")

        # ---- Stage 5: VALIDATE ----
        logger.info("[Stage 5/6] VALIDATE — 质量校验...")
        validator = DataValidator()
        unified_entities = validator.validate(unified_entities)
        valid_summary = validator.get_summary()
        report.total_valid = valid_summary["pass"] + valid_summary["warn"]
        report.total_invalid = valid_summary["fail"]
        report.total_with_warnings = valid_summary["warn"]
        for issue in valid_summary["issues"]:
            report.quality_issues.append(issue)
        logger.info(f"  校验: PASS={valid_summary['pass']} WARN={valid_summary['warn']} FAIL={valid_summary['fail']}")

        if report.total_invalid > 0:
            logger.warning(f"  {report.total_invalid} 个实体未通过校验!")

        # 只保留有效实体写入
        valid_entities = validator.get_valid_entities(unified_entities)

        # ---- Stage 6: WRITE ----
        logger.info("[Stage 6/6] WRITE — 写入存储...")

        # 6a: EntityWriter → EntityRegistry + entities.json
        entity_writer = EntityWriter()
        entity_output_dir = output_dir or str(
            self.project_root / "data" / "entities"
        )
        entity_writer.write(valid_entities, output_dir=entity_output_dir)

        # 填充 EntityRegistry（打通数据断层）
        registry_count = entity_writer.fill_entity_registry(valid_entities)
        logger.info(f"  EntityRegistry: {registry_count} 个领域实体")

        # 6b: MetadataWriter → metadata/pipeline_index.json
        metadata_writer = MetadataWriter()
        metadata_output = str(
            self.project_root / "data" / "metadata" / "pipeline_index.json"
        )
        metadata_writer.write(valid_entities, output_path=metadata_output)

        # ---- 汇总 ----
        report.total_relations = entity_writer.relations_extracted
        report.by_entity_type = self._count_by_type(valid_entities)
        report.duration_seconds = round(time.time() - start_time, 1)

        logger.info(f"Pipeline 完成: {report.summary()}")
        return report

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _ingest(self, source: str, source_path: Optional[str]) -> list[dict]:
        """
        根据数据源类型选择适配器并执行 Ingestion。

        Args:
            source: 数据源类型
            source_path: 数据源路径

        Returns:
            RawEntity dict 列表
        """
        if source == "markdown":
            data_dir = source_path or str(self.project_root / "data" / "raw")
            adapter = MarkdownAdapter(data_dir=data_dir)
            return adapter.ingest()

        # 未来扩展:
        # elif source == "cblue_cmeee":
        #     adapter = CBLUEAdapter(data_dir=source_path, subset="cmeee_v2")
        #     return adapter.ingest()
        # elif source == "custom_json":
        #     ...

        else:
            raise ValueError(f"不支持的数据源类型: {source}。可用: markdown")

    @staticmethod
    def _count_by_type(entities: list) -> dict:
        """按实体类型统计"""
        counts = {}
        for e in entities:
            t = e.entity_type or "unknown"
            counts[t] = counts.get(t, 0) + 1
        return counts
