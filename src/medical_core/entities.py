"""
医疗实体定义模块

定义医疗知识图谱中的核心实体类型。

层级结构:
    MedicalEntity (抽象基类)
    ├── Disease      — 疾病 (含 ICD-10 预留)
    ├── Symptom      — 症状
    ├── Drug         — 药品 (含 ATC 分类预留)
    ├── Examination  — 检查/检验
    ├── Department   — 科室
    └── (后续扩展)
         ├── Anatomy       — 解剖部位
         ├── Procedure     — 手术/操作
         ├── Biomarker     — 生物标志物
         └── Guideline     — 临床指南

参考标准:
    - ICD-10: 国际疾病分类
    - ATC: 解剖治疗化学分类系统
    - SNOMED CT: 医学术语标准 (预留)

用法:
    from src.medical_core.entities import Disease, Symptom, Drug

    d = Disease(
        name="2型糖尿病",
        icd10_code="E11",
        symptoms=["多饮", "多尿", "多食", "体重下降"],
    )
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 实体类型枚举
# ============================================================

class EntityType(Enum):
    """医疗实体类型枚举 — 对应 Neo4j 节点标签"""

    DISEASE = "Disease"
    SYMPTOM = "Symptom"
    DRUG = "Drug"
    EXAMINATION = "Examination"
    DEPARTMENT = "Department"

    # 预留扩展
    ANATOMY = "Anatomy"          # 解剖部位
    PROCEDURE = "Procedure"       # 手术/操作
    BIOMARKER = "Biomarker"       # 生物标志物
    GUIDELINE = "Guideline"       # 临床指南
    COMPLICATION = "Complication"  # 并发症
    POPULATION = "Population"     # 人群 (孕妇/儿童/老人等)

    def to_neo4j_label(self) -> str:
        """转换为 Neo4j 节点标签"""
        return self.value


# ============================================================
# 抽象基类
# ============================================================

@dataclass
class MedicalEntity:
    """
    医疗实体抽象基类

    所有医疗实体共享的属性。

    Attributes:
        name: 实体名称（规范化中文名）
        aliases: 别名/同义词列表
        entity_type: 实体类型
        description: 简要描述
        source: 数据来源
    """
    name: str
    entity_type: EntityType
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    source: str = ""

    @property
    def primary_name(self) -> str:
        """主名称"""
        return self.name

    @property
    def all_names(self) -> list[str]:
        """所有名称 (主名 + 别名)"""
        return [self.name] + self.aliases


# ============================================================
# 具体实体类
# ============================================================

@dataclass
class Disease(MedicalEntity):
    """
    疾病实体

    Attributes:
        icd10_code: ICD-10 编码
        category: 疾病分类 (如 "内分泌与代谢疾病")
        symptoms: 关联症状列表
        complications: 并发症列表
        risk_factors: 危险因素列表
    """
    icd10_code: str = ""
    category: str = ""
    symptoms: list[str] = field(default_factory=list)
    complications: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.entity_type:
            object.__setattr__(self, 'entity_type', EntityType.DISEASE)


@dataclass
class Symptom(MedicalEntity):
    """
    症状实体

    Attributes:
        related_diseases: 关联疾病
        body_site: 发生部位
        severity: 严重程度 (轻/中/重)
        duration: 持续时间特征 (急性/慢性/间歇性)
    """
    related_diseases: list[str] = field(default_factory=list)
    body_site: str = ""
    severity: str = ""   # 轻/中/重
    duration: str = ""   # 急性/慢性/间歇性

    def __post_init__(self):
        if not self.entity_type:
            object.__setattr__(self, 'entity_type', EntityType.SYMPTOM)


@dataclass
class Drug(MedicalEntity):
    """
    药品实体

    Attributes:
        atc_code: ATC 分类编码
        drug_type: 药物分类 (如 "双胍类口服降糖药")
        indications: 适应症
        contraindications: 禁忌症
        adverse_reactions: 不良反应
        dosage: 常用剂量说明
        mechanism: 作用机制
    """
    atc_code: str = ""
    drug_type: str = ""
    indications: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    adverse_reactions: list[str] = field(default_factory=list)
    dosage: str = ""
    mechanism: str = ""

    def __post_init__(self):
        if not self.entity_type:
            object.__setattr__(self, 'entity_type', EntityType.DRUG)


@dataclass
class Examination(MedicalEntity):
    """
    检查/检验实体

    Attributes:
        exam_type: 检查类型 (实验室检查/影像学检查/体格检查)
        normal_range: 正常值范围
        related_diseases: 关联疾病
    """
    exam_type: str = ""
    normal_range: str = ""
    related_diseases: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.entity_type:
            object.__setattr__(self, 'entity_type', EntityType.EXAMINATION)


@dataclass
class Department(MedicalEntity):
    """
    科室实体

    Attributes:
        parent_department: 上级科室
        scope: 诊疗范围
    """
    parent_department: str = ""
    scope: str = ""

    def __post_init__(self):
        if not self.entity_type:
            object.__setattr__(self, 'entity_type', EntityType.DEPARTMENT)


# ============================================================
# 实体注册表 (预留 — Phase 2+)
# ============================================================

class EntityRegistry:
    """
    实体注册表

    用于管理知识图谱中的实体，支持:
        - 按名称/别名查找
        - 按类型过滤
        - 按 ICD-10/ATC 编码检索
        - Neo4j 导入适配

    当前为预留接口，后续阶段实现完整的 CRUD。
    """

    def __init__(self):
        self._entities: dict[str, MedicalEntity] = {}

    def register(self, entity: MedicalEntity) -> None:
        """注册实体"""
        self._entities[entity.name] = entity
        for alias in entity.aliases:
            self._entities[alias] = entity

    def get(self, name: str) -> Optional[MedicalEntity]:
        """按名称查找实体"""
        return self._entities.get(name)

    def get_by_type(self, entity_type: EntityType) -> list[MedicalEntity]:
        """按类型过滤实体"""
        seen = set()
        result = []
        for e in self._entities.values():
            if e.entity_type == entity_type and e.name not in seen:
                result.append(e)
                seen.add(e.name)
        return result

    def __len__(self) -> int:
        return len(set(e.name for e in self._entities.values()))
