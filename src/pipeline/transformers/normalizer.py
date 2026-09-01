"""
实体标准化器

职责:
    - 实体类型映射（Adapter 输出 → 标准 EntityType）
    - 实体名称规范化
    - 别名标准化（去重、排序）
    - 标准编码规范化为 CodeEntry 格式
    - 生成全局唯一的 entity_id

规则:
    1. entity_type 必须是 src.medical_core.entities.EntityType 中定义的有效值
    2. entity_id 格式: "source_prefix:entity_type:normalized_name"
    3. 别名去重且排除与主名相同的项
    4. 标准编码从 str → [CodeEntry] 转换

用法:
    from src.pipeline.transformers import EntityNormalizer
    normalizer = EntityNormalizer()
    unified = normalizer.normalize(raw_entity)
"""

import logging
import re
from typing import Optional

from src.medical_core.entities import EntityType
from src.pipeline.unified_entity import (
    UnifiedEntity, AliasEntry, CodeEntry, ProvenanceEntry
)

logger = logging.getLogger(__name__)


class EntityNormalizer:
    """
    实体标准化器。

    将 Adapter 输出的 RawEntity dict 转换为标准化的 UnifiedEntity。
    """

    # entity_type 字符串 → EntityType 枚举映射
    ENTITY_TYPE_MAP = {
        # 标准名称
        "disease": EntityType.DISEASE,
        "symptom": EntityType.SYMPTOM,
        "drug": EntityType.DRUG,
        "examination": EntityType.EXAMINATION,
        "department": EntityType.DEPARTMENT,
        # 预留类型
        "anatomy": EntityType.ANATOMY,
        "procedure": EntityType.PROCEDURE,
        "biomarker": EntityType.BIOMARKER,
        "guideline": EntityType.GUIDELINE,
        "complication": EntityType.COMPLICATION,
        "population": EntityType.POPULATION,
        # 复数形式（来自目录名）
        "diseases": EntityType.DISEASE,
        "drugs": EntityType.DRUG,
        "symptoms": EntityType.SYMPTOM,
        "guidelines": EntityType.GUIDELINE,
    }

    # 编码系统标准化名称
    CODE_SYSTEM_MAP = {
        "ICD-10": "ICD-10",
        "ICD10": "ICD-10",
        "icd-10": "ICD-10",
        "icd10": "ICD-10",
        "ATC": "ATC",
        "atc": "ATC",
        "SNOMED CT": "SNOMED_CT",
        "SNOMED_CT": "SNOMED_CT",
        "snomed": "SNOMED_CT",
    }

    def __init__(self, source_prefix: str = "manual"):
        """
        Args:
            source_prefix: entity_id 中使用的来源前缀
        """
        self.source_prefix = source_prefix

    def normalize(self, raw_entity: dict) -> UnifiedEntity:
        """
        将 RawEntity dict 标准化为 UnifiedEntity。

        Args:
            raw_entity: Adapter 输出的原始实体 dict

        Returns:
            标准化的 UnifiedEntity
        """
        # 1. 实体类型映射
        entity_type_raw = raw_entity.get("entity_type", "")
        entity_type = self._normalize_entity_type(entity_type_raw)

        # 2. 实体名称规范化
        name = self._normalize_name(raw_entity.get("name", ""))

        # 3. 别名标准化
        aliases = self._normalize_aliases(
            raw_entity.get("aliases", []), name
        )

        # 4. 标准编码规范化
        standard_codes = self._normalize_codes(
            raw_entity.get("standard_code", ""),
            raw_entity.get("code_system", ""),
            entity_type_raw,
        )

        # 5. 生成全局唯一 entity_id
        entity_id = self._generate_entity_id(name, entity_type_raw)

        # 6. 构建 Provenance
        provenance = self._build_provenance(raw_entity)

        # 7. 构建 UnifiedEntity
        return UnifiedEntity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type_raw,
            aliases=aliases,
            standard_codes=standard_codes,
            category=raw_entity.get("category", ""),
            description=raw_entity.get("description", ""),
            properties=raw_entity.get("properties", {}),
            relations=raw_entity.get("relations", []),
            provenance=provenance,
            raw_text=raw_entity.get("raw_text", ""),
        )

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _normalize_entity_type(self, raw_type: str) -> str:
        """
        标准化实体类型字符串。

        Args:
            raw_type: 原始类型字符串

        Returns:
            标准化的实体类型（小写英文）
        """
        normalized = raw_type.strip().lower()
        if normalized in ["diseases"]:
            normalized = "disease"
        elif normalized in ["drugs"]:
            normalized = "drug"
        elif normalized in ["symptoms"]:
            normalized = "symptom"
        elif normalized in ["guidelines"]:
            normalized = "guideline"

        if normalized not in self.ENTITY_TYPE_MAP:
            logger.warning(f"未知实体类型: '{raw_type}'，保留原值")
            return normalized

        return normalized

    @staticmethod
    def _normalize_name(name: str) -> str:
        """规范化实体名称"""
        if not name:
            return ""
        # 去除前后空白
        name = name.strip()
        # 移除多余的内部空白
        name = re.sub(r"\s+", " ", name)
        return name

    @staticmethod
    def _normalize_aliases(raw_aliases: list[str], primary_name: str) -> list[AliasEntry]:
        """
        规范化别名列表。

        - 去重（忽略大小写和空白差异）
        - 排除与主名相同的别名
        - 推断别名的语言（zh/en/la）

        Args:
            raw_aliases: 原始别名列表
            primary_name: 主名称

        Returns:
            标准化 AliasEntry 列表
        """
        seen = {primary_name.lower().strip()}
        result = []

        for alias in raw_aliases:
            if not alias or not isinstance(alias, str):
                continue
            alias_clean = alias.strip()
            alias_lower = alias_clean.lower()
            if alias_lower in seen:
                continue
            seen.add(alias_lower)

            # 推断语言
            lang = EntityNormalizer._infer_language(alias_clean)

            result.append(AliasEntry(name=alias_clean, language=lang))

        return result

    @staticmethod
    def _infer_language(text: str) -> str:
        """推断文本语言"""
        # 包含中文字符 → zh
        if re.search(r"[一-鿿]", text):
            return "zh"
        # 全为 ASCII → en
        if all(ord(c) < 128 for c in text):
            return "en"
        return "zh"

    def _normalize_codes(
        self, code_value: str, code_system: str, entity_type: str
    ) -> list[CodeEntry]:
        """
        标准化编码为 CodeEntry 列表。

        Args:
            code_value: 编码值（如 "E11" 或 "A10BA02"）
            code_system: 编码系统（如 "ICD-10" 或 "ATC"）
            entity_type: 实体类型（用于推断默认编码系统）

        Returns:
            CodeEntry 列表
        """
        if not code_value:
            return []

        # 标准化编码系统
        system = self.CODE_SYSTEM_MAP.get(code_system, code_system)

        # 如果编码系统未知，根据实体类型推断
        if not system:
            if entity_type in ("disease", "diseases"):
                system = "ICD-10"
            elif entity_type in ("drug", "drugs"):
                system = "ATC"

        if not system:
            return []

        return [CodeEntry(system=system, code=code_value.strip())]

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """
        生成全局唯一 entity_id。

        格式: "source_prefix:entity_type:normalized_name"
        示例: "manual:disease:2型糖尿病"
        """
        clean_name = name.strip().lower().replace(" ", "_")
        return f"{self.source_prefix}:{entity_type}:{clean_name}"

    def _build_provenance(self, raw_entity: dict) -> list[ProvenanceEntry]:
        """从 RawEntity 构建溯源信息"""
        return [ProvenanceEntry(
            source_name=raw_entity.get("source_name", "Manual Curation"),
            source_url=raw_entity.get("source_url", ""),
            license=raw_entity.get("license", "Public Domain"),
            adapter=self.source_prefix,
            original_id=raw_entity.get("name", ""),
        )]
