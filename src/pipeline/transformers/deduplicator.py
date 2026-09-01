"""
实体去重器

职责:
    - 检测并标记重复实体
    - 基于确定性规则（名称 + 编码），不引入 ML 复杂度
    - 合并重复实体的别名和溯源信息

去重策略（按优先级）:
    1. 精确 entity_id 匹配 → 直接合并
    2. 同类型 + 同名 → 合并别名和 provenance
    3. 同类型 + 同标准编码 → 合并（可能更名但编码相同）
    4. 别名交叉匹配 → 标记为潜在重复（warn 级别，不自动合并）

用法:
    from src.pipeline.transformers import EntityDeduplicator
    dedup = EntityDeduplicator()
    merged, report = dedup.deduplicate(entities)
"""

import logging
from collections import defaultdict

from src.pipeline.unified_entity import UnifiedEntity

logger = logging.getLogger(__name__)


class DedupReport:
    """去重统计报告"""

    def __init__(self):
        self.total_input: int = 0
        self.total_output: int = 0
        self.duplicates_merged: int = 0
        self.potential_duplicates: list[tuple[str, str]] = []  # [(name_a, name_b), ...]

    def summary(self) -> str:
        return (
            f"去重: {self.total_input} → {self.total_output} "
            f"(合并 {self.duplicates_merged} 组重复, "
            f"{len(self.potential_duplicates)} 组潜在重复)"
        )


class EntityDeduplicator:
    """
    实体去重器。

    基于确定性规则（entity_id / 名称 / 编码）进行去重。
    对于无法确定的情况标记为"潜在重复"（warn），不自动合并。
    """

    def __init__(self, auto_merge: bool = True):
        """
        Args:
            auto_merge: 是否自动合并确认的重复实体
        """
        self.auto_merge = auto_merge

    def deduplicate(self, entities: list[UnifiedEntity]) -> tuple[list[UnifiedEntity], DedupReport]:
        """
        对实体列表进行去重。

        Args:
            entities: UnifiedEntity 列表

        Returns:
            (去重后的实体列表, 去重报告)
        """
        report = DedupReport()
        report.total_input = len(entities)

        if not entities:
            return [], report

        # 阶段 1: 按 entity_id 分组（精确匹配）
        by_id = defaultdict(list)
        for e in entities:
            by_id[e.entity_id].append(e)

        merged = []
        for entity_id, group in by_id.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # 相同 entity_id 的实体 → 合并
                merged_entity = self._merge_entities(group)
                merged.append(merged_entity)
                report.duplicates_merged += len(group) - 1
                logger.debug(f"合并重复 (entity_id): {entity_id} ({len(group)} 个实例)")

        # 阶段 2: 按 (entity_type, name) 检测跨 source_prefix 的重复
        merged, cross_source_report = self._dedup_by_name_and_type(merged)
        report.duplicates_merged += cross_source_report

        # 阶段 3: 别名交叉匹配（仅标记，不自动合并）
        potential = self._detect_alias_overlaps(merged)
        report.potential_duplicates = potential

        report.total_output = len(merged)

        if report.duplicates_merged > 0:
            logger.info(report.summary())
        if potential:
            logger.warning(
                f"发现 {len(potential)} 组潜在重复实体（别名交叉），"
                f"未自动合并，请人工审查"
            )

        return merged, report

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _merge_entities(self, entities: list[UnifiedEntity]) -> UnifiedEntity:
        """
        合并一组重复实体。

        保留第一个实体为主体，合并其他实体的别名和来源信息。

        Args:
            entities: 重复实体列表（至少 2 个）

        Returns:
            合并后的 UnifiedEntity
        """
        if len(entities) == 1:
            return entities[0]

        base = entities[0]

        # 合并别名
        existing_aliases = {(a.name.lower(), a.language) for a in base.aliases}
        for e in entities[1:]:
            for alias in e.aliases:
                key = (alias.name.lower(), alias.language)
                if key not in existing_aliases:
                    base.aliases.append(alias)
                    existing_aliases.add(key)

        # 合并 provenance
        existing_sources = {p.source_name for p in base.provenance}
        for e in entities[1:]:
            for prov in e.provenance:
                if prov.source_name not in existing_sources:
                    base.provenance.append(prov)
                    existing_sources.add(prov.source_name)

        # 合并 properties
        for e in entities[1:]:
            for k, v in e.properties.items():
                if k not in base.properties:
                    base.properties[k] = v

        return base

    def _dedup_by_name_and_type(
        self, entities: list[UnifiedEntity]
    ) -> tuple[list[UnifiedEntity], int]:
        """
        按 (entity_type, name) 检测重复。

        例如: "manual:disease:2型糖尿病" 和 "cblue:disease:2型糖尿病"
        它们的 entity_id 不同，但类型和名称相同。

        Returns:
            (去重后的实体列表, 合并的重复数量)
        """
        by_key = defaultdict(list)
        for e in entities:
            key = (e.entity_type, e.name)
            by_key[key].append(e)

        merged_list = []
        merged_count = 0

        for key, group in by_key.items():
            if len(group) == 1:
                merged_list.append(group[0])
            else:
                merged_entity = self._merge_entities(group)
                merged_list.append(merged_entity)
                merged_count += len(group) - 1
                logger.debug(f"合并重复 (name+type): {key} ({len(group)} 个实例)")

        return merged_list, merged_count

    def _detect_alias_overlaps(
        self, entities: list[UnifiedEntity]
    ) -> list[tuple[str, str]]:
        """
        检测别名交叉匹配（潜在重复）。

        例如: 实体 A 的别名包含 "T2DM"，实体 B 的别名也包含 "T2DM"
        但它们的名称不同。可能是同一实体的不同名称。

        仅标记警告，不自动合并。

        Returns:
            [(entity_a_name, entity_b_name), ...]
        """
        # 构建别名索引: alias_lower → list[entity_name]
        alias_index = defaultdict(list)
        for e in entities:
            for alias in e.aliases:
                alias_index[alias.name.lower()].append(e.name)

        potential = []
        seen_pairs = set()

        for alias, names in alias_index.items():
            if len(names) >= 2:
                for i, name_a in enumerate(names):
                    for name_b in names[i + 1:]:
                        pair = tuple(sorted([name_a, name_b]))
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            potential.append(pair)
                            logger.warning(
                                f"潜在重复: '{name_a}' ↔ '{name_b}' "
                                f"(共享别名: '{alias}')"
                            )

        return potential
