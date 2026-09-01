"""
医疗关系定义模块

定义医疗知识图谱中的关系类型和结构。

关系分类:
    - 疾病相关:   并发症/鉴别诊断/危险因素
    - 疾病-症状:  临床表现
    - 疾病-药物:  药物治疗/禁忌用药
    - 疾病-检查:  诊断检查/筛查
    - 药物-药物:  联合用药/配伍禁忌
    - 科室-疾病:  就诊科室

参考:
    - UMLS Semantic Network (关系类型设计)
    - CMeKG 关系体系 (中文医学知识图谱)

用法:
    from src.medical_core.relations import RelationType, MedicalRelation

    rel = MedicalRelation(
        subject="二甲双胍",
        relation=RelationType.TREATS,
        object="2型糖尿病",
    )
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 关系类型枚举
# ============================================================

class RelationType(Enum):
    """医疗关系类型枚举 — 对应 Neo4j 关系类型"""

    # --- 疾病相关 ---
    HAS_SYMPTOM = "HAS_SYMPTOM"                # 疾病 → 症状
    HAS_COMPLICATION = "HAS_COMPLICATION"       # 疾病 → 并发症
    HAS_RISK_FACTOR = "HAS_RISK_FACTOR"        # 疾病 ← 危险因素
    DIFFERENTIAL_DIAGNOSIS = "DIFFERENTIAL_DIAGNOSIS"  # 疾病 ↔ 疾病 (鉴别)

    # --- 药物相关 ---
    TREATS = "TREATS"                          # 药物 → 疾病 (治疗)
    CONTRAINDICATED_FOR = "CONTRAINDICATED_FOR"  # 药物 → 疾病/人群 (禁忌)
    HAS_ADVERSE_REACTION = "HAS_ADVERSE_REACTION"  # 药物 → 不良反应
    DRUG_INTERACTION = "DRUG_INTERACTION"      # 药物 ↔ 药物 (相互作用)
    ADMINISTERED_WITH = "ADMINISTERED_WITH"    # 药物 → 药物 (联合用药)

    # --- 检查相关 ---
    DIAGNOSES = "DIAGNOSES"                    # 检查 → 疾病 (诊断)
    SCREENS_FOR = "SCREENS_FOR"                # 检查 → 疾病 (筛查)
    MONITORS = "MONITORS"                      # 检查 → 疾病 (监测)

    # --- 科室相关 ---
    TREATS_DISEASE = "TREATS_DISEASE"          # 科室 → 疾病 (治疗)

    # --- 解剖相关 (预留) ---
    LOCATED_IN = "LOCATED_IN"                  # 症状/疾病 → 解剖部位
    AFFECTS = "AFFECTS"                        # 疾病 → 解剖部位

    # --- 人群相关 (预留) ---
    HIGH_RISK_GROUP = "HIGH_RISK_GROUP"        # 疾病 ← 人群 (高危)
    SPECIAL_PRECAUTION = "SPECIAL_PRECAUTION"  # 药物 → 人群 (特殊注意)

    def to_neo4j_rel_type(self) -> str:
        """转换为 Neo4j 关系类型"""
        return self.value

    @property
    def inverse(self) -> Optional["RelationType"]:
        """获取反向关系类型"""
        inverses = {
            RelationType.HAS_SYMPTOM: None,  # 症状被多种疾病共有，不设反向
            RelationType.HAS_COMPLICATION: None,
            RelationType.TREATS: None,
            RelationType.DIAGNOSES: None,
            RelationType.CONTRAINDICATED_FOR: None,
            RelationType.DRUG_INTERACTION: RelationType.DRUG_INTERACTION,
        }
        return inverses.get(self)


# ============================================================
# 关系数据类
# ============================================================

@dataclass
class MedicalRelation:
    """
    医疗关系三元组

    Attributes:
        subject: 主体实体名称
        relation: 关系类型
        object: 客体实体名称
        confidence: 置信度 (0.0 ~ 1.0)
        evidence: 证据来源
        properties: 附加属性 (如：剂量、频率、条件等)
    """
    subject: str
    relation: RelationType
    object: str
    confidence: float = 1.0
    evidence: str = ""
    properties: dict = field(default_factory=dict)

    def to_triple(self) -> tuple[str, str, str]:
        """转换为 (主体, 关系, 客体) 三元组"""
        return (self.subject, self.relation.value, self.object)

    def to_cypher(self) -> str:
        """
        生成 MERGE Cypher 语句 (预留 Neo4j 导入)。

        Returns:
            Cypher MERGE 语句
        """
        subj_label = "Entity"  # 实际使用时需根据实体类型确定
        obj_label = "Entity"

        return (
            f"MATCH (a:{subj_label} {{name: '{self.subject}'}})\n"
            f"MATCH (b:{obj_label} {{name: '{self.object}'}})\n"
            f"MERGE (a)-[:{self.relation.value}]->(b);"
        )


# ============================================================
# 关系注册表 (预留 — Phase 2+)
# ============================================================

class RelationRegistry:
    """
    关系注册表

    用于管理知识图谱中的关系，支持:
        - 按关系类型过滤
        - 按主体/客体实体查询
        - 批量导入/导出
        - Neo4j Cypher 生成

    当前为预留接口。
    """

    def __init__(self):
        self._relations: list[MedicalRelation] = []

    def add(self, relation: MedicalRelation) -> None:
        """添加关系"""
        self._relations.append(relation)

    def get_by_type(self, rel_type: RelationType) -> list[MedicalRelation]:
        """按类型过滤"""
        return [r for r in self._relations if r.relation == rel_type]

    def get_by_subject(self, subject_name: str) -> list[MedicalRelation]:
        """按主体查询"""
        return [r for r in self._relations if r.subject == subject_name]

    def get_by_object(self, object_name: str) -> list[MedicalRelation]:
        """按客体查询"""
        return [r for r in self._relations if r.object == object_name]

    def get_related(self, entity_name: str) -> list[MedicalRelation]:
        """获取与指定实体相关的所有关系"""
        return [
            r for r in self._relations
            if r.subject == entity_name or r.object == entity_name
        ]

    def to_triples(self) -> list[tuple[str, str, str]]:
        """导出为三元组列表"""
        return [r.to_triple() for r in self._relations]

    def __len__(self) -> int:
        return len(self._relations)
