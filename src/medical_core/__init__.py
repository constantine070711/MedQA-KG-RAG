"""
医疗领域核心模块 (medical_core)

为未来医疗知识图谱和论文扩展预留的结构化定义。

子模块:
    entities.py  — 医疗实体定义 (疾病/症状/药物/检查/科室等)
    relations.py — 医疗关系定义 (诊断/治疗/并发症/禁忌等)
    safety.py   — 医疗安全规则 (拒答策略/免责声明/内容审核)

设计原则:
    - 所有实体和关系定义都使用 Python dataclass，便于类型检查和序列化
    - 预留 Neo4j 节点标签和关系类型映射
    - 安全规则可独立于检索引擎运行（输入/输出过滤）
    - 为未来 ICD-10/SNOMED CT 等标准本体映射预留接口

用法:
    from src.medical_core.entities import Disease, Symptom, Drug
    from src.medical_core.relations import RelationType
    from src.medical_core.safety import SafetyChecker
"""

from .entities import (
    EntityType,
    Disease,
    Symptom,
    Drug,
    Examination,
    Department,
    MedicalEntity,
)
from .relations import (
    RelationType,
    MedicalRelation,
)
from .safety import (
    SafetyChecker,
    SafetyLevel,
    MEDICAL_DISCLAIMER,
)

__all__ = [
    # 实体
    "EntityType",
    "Disease",
    "Symptom",
    "Drug",
    "Examination",
    "Department",
    "MedicalEntity",
    # 关系
    "RelationType",
    "MedicalRelation",
    # 安全
    "SafetyChecker",
    "SafetyLevel",
    "MEDICAL_DISCLAIMER",
]
