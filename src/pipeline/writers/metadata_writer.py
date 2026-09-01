"""
元数据写入器

职责:
    - 将 UnifiedEntity 的元数据信息写入 MetadataRegistry
    - 导出 metadata index JSON
    - 为数据溯源提供持久化记录

注意:
    MetadataWriter 写的是文件级元数据（每实体一条记录）。
    EntityWriter 写的是领域实体（填充 EntityRegistry）。

    两者互补：Metadata 侧重"这个知识片段从哪里来"，
    Entity 侧重"这个医学实体是什么"。

用法:
    from src.pipeline.writers import MetadataWriter
    writer = MetadataWriter()
    writer.write(entities, output_path="data/metadata/index.json")
"""

import json
import logging
from pathlib import Path
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


class MetadataWriter:
    """
    元数据写入器。

    将 UnifiedEntity 列表的关键元数据写入 JSON 索引文件。
    每一条记录对应一个知识片段的文件级元数据。
    """

    def write(
        self,
        entities: list,
        output_path: Optional[str] = None,
    ) -> dict:
        """
        将 UnifiedEntity 列表的元数据写入 JSON。

        Args:
            entities: UnifiedEntity 列表
            output_path: JSON 输出路径（可选）

        Returns:
            写入的 metadata index dict
        """
        records = []
        for e in entities:
            record = {
                "name": e.name,
                "entity_type": e.entity_type,
                "entity_id": e.entity_id,
                "aliases": [a.name for a in e.aliases],
                "standard_codes": [
                    {"system": c.system, "code": c.code}
                    for c in e.standard_codes
                ],
                "category": e.category,
                "description": e.description,
                "source_name": (
                    e.provenance[0].source_name if e.provenance else ""
                ),
                "license": (
                    e.provenance[0].license if e.provenance else ""
                ),
                "char_count": len(e.raw_text) if e.raw_text else 0,
                "quality_level": e.quality.level.value,
            }
            records.append(record)

        index = {
            "generated_at": str(date.today()),
            "pipeline_version": "Phase 4",
            "total_records": len(records),
            "by_entity_type": self._count_by_type(entities),
            "records": records,
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
            logger.info(
                f"元数据已写入: {output_path} ({len(records)} 条记录)"
            )

        return index

    @staticmethod
    def _count_by_type(entities: list) -> dict:
        """按实体类型统计"""
        counts = {}
        for e in entities:
            t = e.entity_type or "unknown"
            counts[t] = counts.get(t, 0) + 1
        return counts
