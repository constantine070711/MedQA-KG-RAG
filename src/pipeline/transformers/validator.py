"""
数据质量校验器

职责:
    - 对 UnifiedEntity 进行质量检查
    - 必填字段检查
    - entity_type 合法性检查
    - source / license 完整性检查
    - 空内容检测
    - 异常数据统计

质量等级:
    PASS  — 通过所有检查
    WARN  — 有警告但可继续
    FAIL  — 未通过（缺少必填字段或类型不合法）

用法:
    from src.pipeline.transformers import DataValidator
    validator = DataValidator()
    validated = validator.validate(entities)
    report = validator.get_summary()
"""

import logging
from typing import Optional

from src.pipeline.unified_entity import (
    UnifiedEntity, QualityReport, QualityLevel
)

logger = logging.getLogger(__name__)


class DataValidator:
    """
    数据质量校验器。

    对 UnifiedEntity 列表逐项检查，将质量报告写入每个实体的 quality 字段。
    不修改实体数据内容（仅标记）。
    """

    # 合法的 entity_type 值
    VALID_ENTITY_TYPES = {
        "disease", "symptom", "drug", "examination", "department",
        "anatomy", "procedure", "biomarker", "guideline",
        "complication", "population",
    }

    def __init__(self):
        self._total_checked: int = 0
        self._total_pass: int = 0
        self._total_warn: int = 0
        self._total_fail: int = 0
        self._all_issues: list[str] = []

    def validate(self, entities: list[UnifiedEntity]) -> list[UnifiedEntity]:
        """
        对所有实体执行质量检查。

        将检查结果写入每个实体的 quality 字段。

        Args:
            entities: UnifiedEntity 列表

        Returns:
            带质量报告的 UnifiedEntity 列表（就地修改）
        """
        self._reset()

        for entity in entities:
            report = self._check_entity(entity)
            entity.quality = report

            self._total_checked += 1
            if report.level == QualityLevel.PASS:
                self._total_pass += 1
            elif report.level == QualityLevel.WARN:
                self._total_warn += 1
            else:
                self._total_fail += 1

            # 收集问题
            for err in report.errors:
                self._all_issues.append(f"[FAIL] {entity.name}: {err}")
            for warn in report.warnings:
                self._all_issues.append(f"[WARN] {entity.name}: {warn}")

        logger.info(
            f"数据校验完成: {self._total_checked} 实体 "
            f"(PASS={self._total_pass} WARN={self._total_warn} FAIL={self._total_fail})"
        )

        if self._total_fail > 0:
            logger.warning(f"{self._total_fail} 个实体未通过校验!")

        return entities

    def get_summary(self) -> dict:
        """返回质量校验摘要"""
        return {
            "total_checked": self._total_checked,
            "pass": self._total_pass,
            "warn": self._total_warn,
            "fail": self._total_fail,
            "pass_rate": round(
                self._total_pass / max(self._total_checked, 1) * 100, 1
            ),
            "issues": self._all_issues[:20],  # 最多 20 条
            "total_issues": len(self._all_issues),
        }

    def get_valid_entities(
        self, entities: list[UnifiedEntity]
    ) -> list[UnifiedEntity]:
        """
        获取通过校验的实体（允许有警告）。

        Args:
            entities: UnifiedEntity 列表

        Returns:
            通过校验的实体
        """
        return [e for e in entities if e.is_valid]

    def get_invalid_entities(
        self, entities: list[UnifiedEntity]
    ) -> list[UnifiedEntity]:
        """
        获取未通过校验的实体。

        Args:
            entities: UnifiedEntity 列表

        Returns:
            未通过校验的实体
        """
        return [e for e in entities if not e.is_valid]

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _reset(self):
        """重置统计计数器"""
        self._total_checked = 0
        self._total_pass = 0
        self._total_warn = 0
        self._total_fail = 0
        self._all_issues = []

    def _check_entity(self, entity: UnifiedEntity) -> QualityReport:
        """
        对单个实体执行全部质量检查。

        Args:
            entity: UnifiedEntity

        Returns:
            QualityReport
        """
        errors = []
        warnings = []
        checks = {}

        # ---- 必填字段检查 ----
        checks["name_not_empty"] = self._check_name(entity, errors)
        checks["entity_type_valid"] = self._check_entity_type(entity, errors)
        checks["raw_text_not_empty"] = self._check_raw_text(entity, warnings)

        # ---- 来源和许可证检查 ----
        checks["has_source"] = self._check_source(entity, warnings)
        checks["has_license"] = self._check_license(entity, warnings)

        # ---- 编码检查 ----
        checks["code_valid"] = self._check_codes(entity, warnings)

        # ---- 别名检查 ----
        checks["aliases_not_duplicate_name"] = self._check_aliases(entity, warnings)

        # ---- 确定质量等级 ----
        if errors:
            level = QualityLevel.FAIL
        elif warnings:
            level = QualityLevel.WARN
        else:
            level = QualityLevel.PASS

        return QualityReport(
            level=level,
            errors=errors,
            warnings=warnings,
            checks=checks,
        )

    def _check_name(self, entity: UnifiedEntity, errors: list) -> bool:
        """检查实体名称是否为空"""
        if not entity.name or not entity.name.strip():
            errors.append("实体名称为空")
            return False
        return True

    def _check_entity_type(self, entity: UnifiedEntity, errors: list) -> bool:
        """检查 entity_type 是否合法"""
        if not entity.entity_type:
            errors.append("entity_type 为空")
            return False
        if entity.entity_type not in self.VALID_ENTITY_TYPES:
            errors.append(f"非法的 entity_type: '{entity.entity_type}'")
            return False
        return True

    def _check_raw_text(self, entity: UnifiedEntity, warnings: list) -> bool:
        """检查原始文本是否为空或过短"""
        if not entity.raw_text or not entity.raw_text.strip():
            warnings.append("raw_text 为空，无法索引")
            return False
        if len(entity.raw_text.strip()) < 50:
            warnings.append(f"raw_text 过短 ({len(entity.raw_text)} 字符)")
            return False
        return True

    def _check_source(self, entity: UnifiedEntity, warnings: list) -> bool:
        """检查是否有数据来源信息"""
        if not entity.provenance:
            warnings.append("缺少 provenance 信息")
            return False
        for prov in entity.provenance:
            if not prov.source_name:
                warnings.append("provenance.source_name 为空")
                return False
        return True

    def _check_license(self, entity: UnifiedEntity, warnings: list) -> bool:
        """检查是否有许可证信息"""
        if not entity.provenance:
            return False
        for prov in entity.provenance:
            if not prov.license:
                warnings.append("provenance.license 为空")
                return False
        return True

    def _check_codes(self, entity: UnifiedEntity, warnings: list) -> bool:
        """检查标准编码"""
        if not entity.standard_codes:
            warnings.append("缺少标准编码 (ICD-10/ATC)")
            return False
        return True

    def _check_aliases(self, entity: UnifiedEntity, warnings: list) -> bool:
        """检查别名列表"""
        if not entity.aliases:
            warnings.append("没有别名（可能导致跨语种检索困难）")
            return False
        return True
