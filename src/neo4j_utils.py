"""
Neo4j 知识图谱工具模块 (Phase 3 预留接口)

职责:
    - Neo4j Schema 定义和导出
    - 从 metadata 生成 Cypher CREATE/MERGE 语句
    - 关系模式映射
    - 批量导入工具（预留）
    - 图数据库连接管理（预留 Phase 5）

设计原则:
    - 所有 Cypher 生成函数独立于 Neo4j 连接
    - 支持"离线生成 → 手动导入"和"在线连接"两种模式
    - 节点标签使用 PascalCase，关系类型使用 UPPER_SNAKE_CASE
    - 遵循 medical_core 中定义的实体和关系类型

用法:
    from src.neo4j_utils import Neo4jSchema

    schema = Neo4jSchema()
    cypher = schema.generate_node_cypher(metadata_registry)
    schema.export_to_file(cypher, "data/processed/neo4j_import.cypher")
"""

import json
import logging
from pathlib import Path
from datetime import date
from typing import Callable, Optional

from neo4j import GraphDatabase

from .config import get_config

logger = logging.getLogger(__name__)


# ============================================================
# Neo4j Schema 定义
# ============================================================

class Neo4jSchema:
    """
    Neo4j 知识图谱 Schema 管理器

    定义节点标签、关系类型、约束和索引。
    基于 medical_core 的实体和关系设计。
    """

    # 节点标签 → 描述
    NODE_LABELS = {
        "Disease": "疾病",
        "Symptom": "症状",
        "Drug": "药品",
        "Examination": "检查",
        "Department": "科室",
        # 预留
        "Anatomy": "解剖部位",
        "Procedure": "手术/操作",
        "Biomarker": "生物标志物",
        "Guideline": "临床指南",
        "Complication": "并发症",
        "Population": "人群",
    }

    # 关系类型 → (主体标签, 客体标签, 描述)
    RELATIONSHIP_TYPES = {
        "HAS_SYMPTOM": ("Disease", "Symptom", "疾病→症状"),
        "HAS_COMPLICATION": ("Disease", "Complication", "疾病→并发症"),
        "HAS_RISK_FACTOR": ("Disease", "Disease", "危险因素→疾病"),
        "DIFFERENTIAL_DIAGNOSIS": ("Disease", "Disease", "鉴别诊断"),
        "TREATS": ("Drug", "Disease", "药物治疗"),
        "CONTRAINDICATED_FOR": ("Drug", "Disease", "禁忌用药"),
        "HAS_ADVERSE_REACTION": ("Drug", "Symptom", "药物不良反应"),
        "DRUG_INTERACTION": ("Drug", "Drug", "药物相互作用"),
        "ADMINISTERED_WITH": ("Drug", "Drug", "联合用药"),
        "DIAGNOSES": ("Examination", "Disease", "检查诊断"),
        "SCREENS_FOR": ("Examination", "Disease", "检查筛查"),
        "MONITORS": ("Examination", "Disease", "检查监测"),
        "TREATS_DISEASE": ("Department", "Disease", "科室治疗"),
        "LOCATED_IN": ("Symptom", "Anatomy", "症状部位"),
        "AFFECTS": ("Disease", "Anatomy", "疾病影响部位"),
        "HIGH_RISK_GROUP": ("Disease", "Population", "高危人群"),
        "SPECIAL_PRECAUTION": ("Drug", "Population", "特殊注意"),
    }

    # 约束定义
    CONSTRAINTS = [
        "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (n:Disease) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (n:Drug) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (n:Symptom) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT examination_name IF NOT EXISTS FOR (n:Examination) REQUIRE n.name IS UNIQUE",
    ]

    # 索引定义
    INDEXES = [
        "CREATE INDEX disease_icd10 IF NOT EXISTS FOR (n:Disease) ON (n.standard_code)",
        "CREATE INDEX drug_atc IF NOT EXISTS FOR (n:Drug) ON (n.standard_code)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.entity_type)",
    ]

    # ----------------------------------------------------------
    # 完整 Schema DDL
    # ----------------------------------------------------------

    def generate_schema_cypher(self) -> str:
        """
        生成完整的 Neo4j Schema DDL（约束 + 索引）。

        Returns:
            Cypher DDL 语句
        """
        lines = [
            "// ============================================",
            "// Medical Knowledge Graph — Neo4j Schema",
            f"// Generated: {date.today()}",
            "// Phase 5: Execute before first data import",
            "// ============================================",
            "",
            "// --- Constraints ---",
        ]

        for c in self.CONSTRAINTS:
            lines.append(f"{c};")
            lines.append("")

        lines.append("// --- Indexes ---")
        for idx in self.INDEXES:
            lines.append(f"{idx};")
            lines.append("")

        lines.append("// --- Schema definition complete ---")
        return "\n".join(lines)

    # ----------------------------------------------------------
    # 从 metadata 生成节点 Cypher
    # ----------------------------------------------------------

    def generate_node_cypher(self, metadata_registry) -> str:
        """
        从 MetadataRegistry 生成所有节点的 MERGE 语句。

        Args:
            metadata_registry: MetadataRegistry 实例

        Returns:
            Cypher 语句块
        """
        lines = [
            "// ============================================",
            "// Medical Knowledge Graph — Node Import",
            f"// Generated: {date.today()}",
            f"// Nodes: {len(metadata_registry)}",
            "// ============================================",
            "",
        ]

        for record in metadata_registry.get_all():
            label = self._entity_type_to_label(record.entity_type)
            props = record.to_neo4j_properties()

            # 构建属性字符串（转义单引号）
            props_parts = []
            for k, v in props.items():
                if isinstance(v, str):
                    escaped = v.replace("'", "\\'")
                    props_parts.append(f"n.{k} = '{escaped}'")
                elif isinstance(v, list):
                    props_parts.append(f"n.{k} = {json.dumps(v, ensure_ascii=False)}")

            props_str = ",\n      ".join(props_parts)

            lines.append(f"// {record.name} ({record.entity_type})")
            lines.append(f"MERGE (n:{label} {{name: '{record.name}'}})")
            if props_str:
                lines.append(f"SET {props_str}")
            lines.append(f"SET n.updated_at = date()")
            lines.append("")

        return "\n".join(lines)

    # ----------------------------------------------------------
    # 关系模式模板
    # ----------------------------------------------------------

    def generate_relation_templates(self) -> str:
        """
        生成关系类型的 Cypher 模板（供手动填充）。

        Returns:
            关系模板 Cypher
        """
        lines = [
            "// ============================================",
            "// Medical Knowledge Graph — Relation Templates",
            f"// Generated: {date.today()}",
            "// ============================================",
            "",
            "// 使用方式:",
            "//   1. 替换 {subject_name} 和 {object_name} 为实际实体名称",
            "//   2. 根据 medical_core/relations.py 中的 RelationType 选择关系",
            "//   3. 可添加 confidence/evidence/source 等属性",
            "",
        ]

        for rel_type, (subj_label, obj_label, desc) in self.RELATIONSHIP_TYPES.items():
            lines.append(f"// --- {rel_type}: {desc} ---")
            lines.append(
                f"MATCH (a:{subj_label} {{name: '{{subject_name}}'}})"
            )
            lines.append(
                f"MATCH (b:{obj_label} {{name: '{{object_name}}'}})"
            )
            lines.append(
                f"MERGE (a)-[:{rel_type} {{confidence: 1.0}}]->(b);"
            )
            lines.append("")

        return "\n".join(lines)

    # ----------------------------------------------------------
    # 导出
    # ----------------------------------------------------------

    def export_schema(self, output_dir: str) -> dict:
        """
        导出完整的 Neo4j Schema 到文件。

        Args:
            output_dir: 输出目录

        Returns:
            {"schema": path, "nodes": path, "relations": path}
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = {}

        # Schema DDL
        schema_path = output_path / "neo4j_schema.cypher"
        schema_path.write_text(self.generate_schema_cypher(), encoding="utf-8")
        files["schema"] = str(schema_path)

        # Relation templates
        rel_path = output_path / "neo4j_relation_templates.cypher"
        rel_path.write_text(self.generate_relation_templates(), encoding="utf-8")
        files["relations"] = str(rel_path)

        logger.info(
            f"Neo4j Schema 已导出: schema + relation templates → {output_dir}"
        )
        return files

    def export_nodes_from_registry(
        self,
        metadata_registry,
        output_path: str,
    ) -> str:
        """
        从 MetadataRegistry 导出节点 Cypher。

        Args:
            metadata_registry: MetadataRegistry 实例
            output_path: 输出文件路径

        Returns:
            文件路径
        """
        cypher = self.generate_node_cypher(metadata_registry)
        Path(output_path).write_text(cypher, encoding="utf-8")
        logger.info(f"节点 Cypher 已导出: {output_path}")
        return output_path

    # ----------------------------------------------------------
    # 工具函数
    # ----------------------------------------------------------

    @staticmethod
    def _entity_type_to_label(entity_type: str) -> str:
        """将 entity_type 转换为 Neo4j 节点标签（PascalCase）"""
        mapping = {
            "disease": "Disease",
            "symptom": "Symptom",
            "drug": "Drug",
            "examination": "Examination",
            "department": "Department",
            "guideline": "Guideline",
        }
        return mapping.get(entity_type, "Entity")

    def get_summary(self) -> dict:
        """返回 Schema 摘要"""
        return {
            "node_labels": list(self.NODE_LABELS.keys()),
            "node_label_count": len(self.NODE_LABELS),
            "relationship_types": list(self.RELATIONSHIP_TYPES.keys()),
            "relationship_count": len(self.RELATIONSHIP_TYPES),
            "constraints": len(self.CONSTRAINTS),
            "indexes": len(self.INDEXES),
        }

    # ----------------------------------------------------------
    # Schema 落地 (Phase 6 F1)
    # ----------------------------------------------------------

    def bootstrap(self, conn: "Neo4jConnection") -> dict:
        """在已连接的 Neo4j 上执行现有 Schema DDL（约束 + 索引），幂等。

        直接执行 self.CONSTRAINTS / self.INDEXES（均含 IF NOT EXISTS），
        重复执行不会报错、不会重复创建。**不新增任何未批准的实体/关系定义。**

        Args:
            conn: 已连接的 Neo4jConnection

        Returns:
            {
                "constraints": [{"name", "ok", "error"}],
                "indexes": [{"name", "ok", "error"}],
                "executed_total": int,
                "failed": int,
            }
        """
        results = {
            "constraints": [],
            "indexes": [],
            "executed_total": 0,
            "failed": 0,
        }

        for stmt in self.CONSTRAINTS:
            name = self._ddl_name(stmt)
            res = self._execute_ddl(conn, stmt)
            results["constraints"].append({"name": name, **res})
            results["executed_total"] += 1
            if not res["ok"]:
                results["failed"] += 1

        for stmt in self.INDEXES:
            name = self._ddl_name(stmt)
            res = self._execute_ddl(conn, stmt)
            results["indexes"].append({"name": name, **res})
            results["executed_total"] += 1
            if not res["ok"]:
                results["failed"] += 1

        return results

    def verify_schema(self, conn: "Neo4jConnection") -> dict:
        """查询当前 Neo4j 实际存在的约束与索引（用于验收）。"""
        try:
            constraints = conn.query(
                "SHOW CONSTRAINTS YIELD name, type, entityType "
                "RETURN name, type, entityType ORDER BY name"
            )
        except Exception as exc:
            constraints = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        try:
            indexes = conn.query(
                "SHOW INDEXES YIELD name, type, labelsOrTypes "
                "RETURN name, type, labelsOrTypes ORDER BY name"
            )
        except Exception as exc:
            indexes = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        return {"constraints": constraints, "indexes": indexes}

    @staticmethod
    def _ddl_name(stmt: str) -> str:
        """从 DDL 语句中粗略提取对象名（用于报告）。"""
        try:
            parts = stmt.split()
            idx = parts.index("IF") if "IF" in parts else -1
            return parts[idx - 1] if idx > 0 else stmt[:50]
        except Exception:
            return stmt[:50]

    @staticmethod
    def _execute_ddl(conn: "Neo4jConnection", stmt: str) -> dict:
        """执行单条 DDL，返回 ok / error。"""
        try:
            conn.execute(stmt)
            return {"ok": True, "error": None}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


# ============================================================
# Neo4j 连接管理 (Phase 6 F1 实现)
# ============================================================

class Neo4jConnection:
    """
    Neo4j 数据库连接管理器（Phase 6 F1 实现）。

    基于 neo4j Python driver，提供:
        - connect / close（driver 生命周期）
        - execute / query（单语句执行）
        - session / run_transaction（会话与事务管理）
        - execute_batch（批量执行，Schema DDL 用）
        - verify（连接 + 认证校验）
        - 基本异常处理（连接失败 / 认证失败给出明确错误）

    认证信息默认从项目 config（.env）读取；也可显式传入 uri/user/password。

    用法:
        conn = Neo4jConnection()          # 从 config 读取
        conn.connect()
        rows = conn.execute("RETURN 1 AS ok")
        conn.close()

        # 或上下文管理器
        with Neo4jConnection() as conn:
            conn.execute("RETURN 1 AS ok")
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        cfg = get_config()
        self.uri = uri or cfg.neo4j_uri
        self.user = user or cfg.neo4j_user
        self.password = password if password is not None else cfg.neo4j_password
        self._driver = None

    # ----------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------

    def connect(self) -> "Neo4jConnection":
        """建立 driver 连接并验证连通性/认证。

        Raises:
            RuntimeError: 连接或认证失败（带明确原因）
        """
        if self._driver is not None:
            return self
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                connection_timeout=15,
            )
            self._driver.verify_connectivity()
            logger.info(f"Neo4j 连接成功: {self.uri}")
        except Exception as exc:
            # 关闭可能半初始化的 driver
            if self._driver is not None:
                try:
                    self._driver.close()
                except Exception:
                    pass
                self._driver = None
            raise RuntimeError(
                f"Neo4j 连接/认证失败 ({self.uri}): {type(exc).__name__}: {str(exc)[:300]}"
            ) from exc
        return self

    def close(self) -> None:
        """关闭 driver 连接（幂等）。"""
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception as exc:
                logger.warning(f"Neo4j driver 关闭异常: {exc}")
            finally:
                self._driver = None

    def verify(self) -> bool:
        """校验连接 + 认证是否有效。"""
        self.connect()
        self._driver.verify_connectivity()
        return True

    # ----------------------------------------------------------
    # 属性 / 上下文
    # ----------------------------------------------------------

    @property
    def driver(self):
        """获取底层 driver（未连接时自动 connect）。"""
        if self._driver is None:
            self.connect()
        return self._driver

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    def session(self):
        """打开一个 driver session（调用方负责关闭）。"""
        return self.driver.session()

    def __enter__(self) -> "Neo4jConnection":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ----------------------------------------------------------
    # 执行
    # ----------------------------------------------------------

    def execute(self, cypher: str, params: Optional[dict] = None) -> list[dict]:
        """执行 Cypher 语句，返回记录列表（dict，JSON 可序列化）。

        单语句执行；抛出的异常由调用方处理（或由 execute_batch 聚合）。
        """
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [dict(r) for r in result]

    def query(self, cypher: str, params: Optional[dict] = None) -> list[dict]:
        """execute() 的别名（读查询语义）。"""
        return self.execute(cypher, params)

    def execute_batch(self, statements: list[str]) -> list[dict]:
        """批量执行多条语句，逐条捕获错误，不中断整体。

        Returns:
            [{"statement": str, "ok": bool, "error": str|None}, ...]
        """
        results: list[dict] = []
        with self.driver.session() as session:
            for stmt in statements:
                try:
                    session.run(stmt).consume()
                    results.append({
                        "statement": stmt[:80],
                        "ok": True,
                        "error": None,
                    })
                except Exception as exc:
                    results.append({
                        "statement": stmt[:80],
                        "ok": False,
                        "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    })
        return results

    def run_transaction(self, fn: Callable, *args, **kwargs):
        """在事务中执行调用方函数 fn(tx, *args, **kwargs)，提交并返回结果。

        失败自动回滚。
        """
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                result = fn(tx, *args, **kwargs)
                tx.commit()
                return result

    # ----------------------------------------------------------
    # 数据导入（Phase 6 F2 实现，F1 不启用）
    # ----------------------------------------------------------

    def import_nodes_from_registry(self, registry, batch_size: int = 100):
        """批量导入节点（Phase 6 F2）。

        F1 阶段不实现图数据导入；此接口保留，待 F2 落地。
        """
        raise NotImplementedError(
            "节点导入属 Phase 6 F2（curated/graphml 导入），F1 不实现。"
        )


# ============================================================
# 便捷函数
# ============================================================

def export_neo4j_artifacts(
    metadata_registry,
    output_dir: str,
) -> dict:
    """
    一键导出所有 Neo4j 相关文件。

    Args:
        metadata_registry: MetadataRegistry 实例
        output_dir: 输出目录

    Returns:
        导出文件路径字典
    """
    schema = Neo4jSchema()

    files = schema.export_schema(output_dir)

    # 同时导出节点数据
    nodes_path = str(Path(output_dir) / "neo4j_nodes.cypher")
    schema.export_nodes_from_registry(metadata_registry, nodes_path)
    files["nodes"] = nodes_path

    return files


# ============================================================
# 连接工厂（Phase 6 F1）
# ============================================================

def get_neo4j_connection(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> Neo4jConnection:
    """
    获取 Neo4j 连接实例（认证信息默认从项目 config / .env 读取）。

    用法:
        conn = get_neo4j_connection()
        with conn:
            rows = conn.execute("RETURN 1 AS ok")
    """
    return Neo4jConnection(uri=uri, user=user, password=password)
