"""
实体写入器

职责:
    - 将 UnifiedEntity 填充到 EntityRegistry（打通 Phase 2 遗留的数据断层）
    - 导出结构化实体数据到 data/entities/entities.json
    - 为 Neo4j 导入和领域查询提供实体数据

这是 Phase 4 最关键的 Writer —— EntityRegistry 自 Phase 2 定义以来
从未被实际数据填充过。此模块解决了这个架构断层。

输出:
    data/entities/entities.json  — 标准化的实体和关系数据
        {
            "generated_at": "2026-07-26",
            "entities": [ ... ],
            "relations": [ ... ],
            "total_entities": N,
            "total_relations": M
        }

用法:
    from src.pipeline.writers import EntityWriter
    writer = EntityWriter()
    writer.write(entities, output_dir="data/entities")
"""

import json
import logging
from pathlib import Path
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


class EntityWriter:
    """
    实体写入器。

    将 UnifiedEntity 列表写入 EntityRegistry 和 JSON 文件。
    同时提取关系信息，为 Neo4j 导入提供数据基础。
    """

    def __init__(self):
        self._entities_written: int = 0
        self._relations_extracted: int = 0

    def write(
        self,
        entities: list,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        将 UnifiedEntity 列表写入存储。

        Args:
            entities: UnifiedEntity 列表（已通过校验）
            output_dir: 输出目录（默认为 data/entities/）

        Returns:
            写入摘要 dict
        """
        # 序列化实体
        entity_records = []
        for e in entities:
            record = e.to_dict()
            # 补充 pipeline 元数据
            record["pipeline_processed_at"] = str(date.today())
            record["pipeline_status"] = e.quality.level.value
            entity_records.append(record)

        # 提取所有关系（去重）
        relations = self._extract_all_relations(entities)

        # 构建输出数据
        output = {
            "generated_at": str(date.today()),
            "pipeline_version": "Phase 4",
            "total_entities": len(entity_records),
            "total_relations": len(relations),
            "by_entity_type": self._count_by_type(entities),
            "entities": entity_records,
            "relations": relations,
        }

        # 写入 JSON
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            json_path = output_path / "entities.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            logger.info(
                f"实体数据已写入: {json_path} "
                f"({len(entity_records)} 实体, {len(relations)} 关系)"
            )

            # 同时写入关系 CSV（方便 Neo4j 导入）
            csv_path = output_path / "relations.csv"
            self._write_relations_csv(relations, csv_path)

        self._entities_written = len(entity_records)
        self._relations_extracted = len(relations)

        return output

    def fill_entity_registry(self, entities: list) -> int:
        """
        将实体填充到 medical_core EntityRegistry。

        打通 Phase 2 以来 EntityRegistry 一直为空的数据断层。

        Args:
            entities: UnifiedEntity 列表

        Returns:
            填充的实体数
        """
        from src.medical_core.entities import EntityRegistry, EntityType
        from src.medical_core.entities import (
            Disease, Symptom, Drug, Examination, Department
        )

        registry = EntityRegistry()

        ET = EntityType

        # entity_type → 构造函数的映射
        entity_factory = {
            "disease": lambda e: Disease(
                name=e.name,
                entity_type=ET.DISEASE,
                aliases=[a.name for a in e.aliases],
                description=e.description,
                source=e.provenance[0].source_name if e.provenance else "",
                icd10_code=e.primary_code if e.primary_code else "",
                category=e.category,
            ),
            "drug": lambda e: Drug(
                name=e.name,
                entity_type=ET.DRUG,
                aliases=[a.name for a in e.aliases],
                description=e.description,
                source=e.provenance[0].source_name if e.provenance else "",
                atc_code=e.primary_code if e.primary_code else "",
                drug_type=e.properties.get("drug_type", ""),
            ),
            "symptom": lambda e: Symptom(
                name=e.name,
                entity_type=ET.SYMPTOM,
                aliases=[a.name for a in e.aliases],
                description=e.description,
                source=e.provenance[0].source_name if e.provenance else "",
            ),
            "examination": lambda e: Examination(
                name=e.name,
                entity_type=ET.EXAMINATION,
                aliases=[a.name for a in e.aliases],
                description=e.description,
                source=e.provenance[0].source_name if e.provenance else "",
            ),
            "department": lambda e: Department(
                name=e.name,
                entity_type=ET.DEPARTMENT,
                aliases=[a.name for a in e.aliases],
                description=e.description,
                source=e.provenance[0].source_name if e.provenance else "",
            ),
        }

        count = 0
        for e in entities:
            factory = entity_factory.get(e.entity_type)
            if factory:
                try:
                    entity = factory(e)
                    registry.register(entity)
                    count += 1
                except Exception as ex:
                    logger.warning(f"注册实体失败: {e.name} — {ex}")
            else:
                logger.debug(f"跳过未知实体类型: {e.entity_type} — {e.name}")

        logger.info(f"EntityRegistry 已填充: {count} 个领域实体 (共 {len(registry)} 个索引)")
        return count

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    @staticmethod
    def _extract_all_relations(entities: list) -> list[dict]:
        """
        从所有实体中提取关系列表（去重）。

        Args:
            entities: UnifiedEntity 列表

        Returns:
            去重后的关系列表
        """
        seen = set()
        unique_relations = []

        for entity in entities:
            for rel in entity.relations:
                # 构建去重键
                subj = rel.get("subject", "")
                pred = rel.get("relation", "")
                obj = rel.get("object", "")
                key = (subj, pred, obj)

                if key not in seen:
                    seen.add(key)
                    unique_relations.append({
                        "subject": subj,
                        "relation": pred,
                        "object": obj,
                    })

        return unique_relations

    @staticmethod
    def _write_relations_csv(relations: list[dict], csv_path: Path) -> None:
        """
        导出关系为 CSV（简化 Neo4j 导入）。

        CSV 格式: subject,relation,object
        """
        lines = ["subject,relation,object"]
        for rel in relations:
            subj = rel.get("subject", "").replace(",", "，")
            pred = rel.get("relation", "")
            obj = rel.get("object", "").replace(",", "，")
            lines.append(f"{subj},{pred},{obj}")

        csv_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"关系 CSV 已导出: {csv_path} ({len(relations)} 条)")

    @staticmethod
    def _count_by_type(entities: list) -> dict:
        """按实体类型统计"""
        counts = {}
        for e in entities:
            t = e.entity_type or "unknown"
            counts[t] = counts.get(t, 0) + 1
        return counts

    @property
    def entities_written(self) -> int:
        return self._entities_written

    @property
    def relations_extracted(self) -> int:
        return self._relations_extracted
