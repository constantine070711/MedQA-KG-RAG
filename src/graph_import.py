"""Phase 6 F2 — 图数据导入（LightRAG graphml 原始层 + curated 临床层）。

职责:
    - 解析 LightRAG graphml → Entity 节点 + RELATED_TO 关系（原始层）
    - 解析 data/entities/entities.json → curated 临床层节点（Disease/Drug 等）
    - 参数化 UNWIND 批量导入（不把数据拼接进 Cypher）
    - 幂等（MERGE + 唯一约束）；缺失/异常/无法映射字段统计上报，不静默丢弃
    - curated 占位符关系（object="见正文…章节"）**不导入**；真实关系重建属 F3

不实现:
    - Neo4jGraphRetriever（F4）/ Evidence Fusion（F5）/ 关系重建（F3）
"""

import json
import logging
from pathlib import Path
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

# ---- 原始层常量 ----
ENTITY_LABEL = "Entity"
RELATED_TO = "RELATED_TO"

# ---- curated 类型 → Neo4j 标签（受控映射常量，非原始数据拼接）----
LABEL_MAP = {
    "disease": "Disease",
    "drug": "Drug",
    "symptom": "Symptom",
    "examination": "Examination",
    "department": "Department",
}

# ---- 占位符关系特征（curated relations 指向"见正文…章节"）----
PLACEHOLDER_MARKERS = ("见正文", "章节", "见相关")


def is_placeholder_relation(rel: dict) -> bool:
    """判断 curated 关系是否为占位符（object 指向正文章节引用）。

    F2 不将占位符当作真实医学关系写入。
    """
    obj = str(rel.get("object", ""))
    return any(m in obj for m in PLACEHOLDER_MARKERS)


# ============================================================
# 解析（纯函数，零 DB）
# ============================================================

def parse_graphml(graphml_path) -> dict:
    """解析 LightRAG graphml → Entity 节点 + RELATED_TO 关系（原始层）。

    Returns:
        {
            "nodes": [{entity_id, name, entity_type, description, source_id, file_path}],
            "edges": [{src, tgt, weight, description, keywords, source_id, file_path}],
            "skipped_nodes": [...], "skipped_edges": [...],
            "stats": {source_nodes, source_edges, parsed_nodes, parsed_edges,
                      skipped_nodes, skipped_edges, node_type_distribution,
                      attribute_completeness},
        }
    """
    G = nx.read_graphml(str(graphml_path))
    node_ids = set(G.nodes())

    nodes: list[dict] = []
    skipped_nodes: list[dict] = []
    for nid, d in G.nodes(data=True):
        entity_id = d.get("entity_id") or nid
        if not entity_id:
            skipped_nodes.append({"node_id": nid, "reason": "empty entity_id"})
            continue
        nodes.append({
            "entity_id": entity_id,
            "name": nid,
            "entity_type": d.get("entity_type", ""),
            "description": d.get("description", ""),
            "source_id": d.get("source_id", ""),
            "file_path": d.get("file_path", ""),
        })

    edges: list[dict] = []
    skipped_edges: list[dict] = []
    for s, t, d in G.edges(data=True):
        if s not in node_ids or t not in node_ids:
            skipped_edges.append({"src": s, "tgt": t, "reason": "endpoint not in node set"})
            continue
        edges.append({
            "src": s,
            "tgt": t,
            "weight": float(d.get("weight", 0.0) or 0.0),
            "description": d.get("description", ""),
            "keywords": d.get("keywords", ""),
            "source_id": d.get("source_id", ""),
            "file_path": d.get("file_path", ""),
        })

    node_type_dist: dict[str, int] = {}
    for n in nodes:
        node_type_dist[n["entity_type"]] = node_type_dist.get(n["entity_type"], 0) + 1

    return {
        "nodes": nodes,
        "edges": edges,
        "skipped_nodes": skipped_nodes,
        "skipped_edges": skipped_edges,
        "stats": {
            "source_nodes": G.number_of_nodes(),
            "source_edges": G.number_of_edges(),
            "parsed_nodes": len(nodes),
            "parsed_edges": len(edges),
            "skipped_nodes": len(skipped_nodes),
            "skipped_edges": len(skipped_edges),
            "node_type_distribution": node_type_dist,
            "attribute_completeness": {
                "nodes_with_description": sum(1 for n in nodes if n["description"]),
                "nodes_with_source_id": sum(1 for n in nodes if n["source_id"]),
                "edges_with_keywords": sum(1 for e in edges if e["keywords"]),
                "edges_with_weight": sum(1 for e in edges if e["weight"] is not None),
            },
        },
    }


def parse_curated(entities_path) -> dict:
    """解析 curated entities.json → 临床层节点。

    F2 只导入可靠实体/属性；curated 占位符关系**不导入**（关系重建属 F3）。

    Returns:
        {
            "nodes": [{name, entity_id, label, entity_type, standard_code,
                       standard_system, aliases, category, description, source}],
            "skipped": [...], "relations": [],
            "stats": {source_entities, source_relations, placeholder_relations,
                      imported_relations, imported_nodes, skipped_entities,
                      node_type_distribution, standard_code_coverage, alias_coverage},
        }
    """
    data = json.loads(Path(entities_path).read_text(encoding="utf-8"))

    nodes: list[dict] = []
    skipped: list[dict] = []
    type_dist: dict[str, int] = {}
    std_code_count = 0

    for e in data.get("entities", []):
        etype = e.get("entity_type", "")
        label = LABEL_MAP.get(etype)
        if not label:
            skipped.append({
                "name": e.get("name", ""),
                "entity_type": etype,
                "reason": "unsupported entity_type",
            })
            continue

        std_sys, std = "", ""
        for sc in e.get("standard_codes", []) or []:
            std_sys = sc.get("system", "") or ""
            std = sc.get("code", "") or ""
            if std:
                break
        aliases = [a.get("name", "") for a in e.get("aliases", []) or [] if a.get("name")]

        nodes.append({
            "name": e.get("name", ""),
            "entity_id": e.get("entity_id", ""),
            "label": label,
            "entity_type": etype,
            "standard_code": std,
            "standard_system": std_sys,
            "aliases": aliases,
            "category": e.get("category", ""),
            "description": e.get("description", ""),
            "source": e.get("source", ""),
        })
        type_dist[label] = type_dist.get(label, 0) + 1
        if std:
            std_code_count += 1

    rels = data.get("relations", [])
    placeholder_rels = sum(1 for r in rels if is_placeholder_relation(r))

    return {
        "nodes": nodes,
        "skipped": skipped,
        "relations": [],  # F2 不导入 curated 关系
        "stats": {
            "source_entities": len(data.get("entities", [])),
            "source_relations": len(rels),
            "placeholder_relations": placeholder_rels,
            "imported_relations": 0,
            "imported_nodes": len(nodes),
            "skipped_entities": len(skipped),
            "node_type_distribution": type_dist,
            "standard_code_coverage": std_code_count,
            "alias_coverage": sum(1 for n in nodes if n["aliases"]),
        },
    }


# ============================================================
# 导入（参数化 UNWIND 批量，幂等）
# ============================================================

def _run_batched(conn, cypher: str, items: list, batch_size: int, name: str) -> dict:
    """分批执行参数化 UNWIND；单批失败记录 error 不中断。"""
    total = len(items)
    ok = 0
    errors: list[dict] = []
    for i in range(0, total, batch_size):
        batch = items[i:i + batch_size]
        try:
            conn.execute(cypher, {"batch": batch})
            ok += len(batch)
        except Exception as exc:
            errors.append({"start": i, "size": len(batch),
                           "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
    return {"name": name, "total": total, "ok": ok, "errors": errors}


def ensure_entity_id_constraint(conn) -> dict:
    """确保 Entity.entity_id 唯一约束（幂等）。"""
    try:
        conn.execute(
            "CREATE CONSTRAINT entity_id IF NOT EXISTS "
            "FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE"
        )
        return {"ok": True, "error": None}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def import_entity_nodes(conn, items: list[dict], batch_size: int = 500) -> dict:
    """导入 Entity 节点（原始层），MERGE 以 entity_id 为键。"""
    cypher = f"""
    UNWIND $batch AS item
    MERGE (n:{ENTITY_LABEL} {{entity_id: item.entity_id}})
    SET n.name = item.name,
        n.entity_type = item.entity_type,
        n.description = item.description,
        n.source_id = item.source_id,
        n.file_path = item.file_path
    """
    return _run_batched(conn, cypher, items, batch_size, f"entity_nodes")


def import_curated_nodes(conn, label: str, items: list[dict], batch_size: int = 500) -> dict:
    """导入 curated 临床层节点（Disease/Drug 等），MERGE 以 name 为键。

    label 来自受控映射常量 LABEL_MAP，非原始数据。
    """
    cypher = f"""
    UNWIND $batch AS item
    MERGE (n:{label} {{name: item.name}})
    SET n.entity_id = item.entity_id,
        n.entity_type = item.entity_type,
        n.standard_code = item.standard_code,
        n.standard_system = item.standard_system,
        n.aliases = item.aliases,
        n.category = item.category,
        n.description = item.description,
        n.source = item.source
    """
    return _run_batched(conn, cypher, items, batch_size, f"curated_{label}")


def import_related_edges(conn, items: list[dict], batch_size: int = 500) -> dict:
    """导入 RELATED_TO 关系（原始层），MERGE 全模式避免重复。"""
    cypher = f"""
    UNWIND $batch AS item
    MATCH (a:{ENTITY_LABEL} {{entity_id: item.src}})
    MATCH (b:{ENTITY_LABEL} {{entity_id: item.tgt}})
    MERGE (a)-[r:{RELATED_TO}]->(b)
    SET r.weight = item.weight,
        r.description = item.description,
        r.keywords = item.keywords,
        r.source_id = item.source_id,
        r.file_path = item.file_path
    """
    return _run_batched(conn, cypher, items, batch_size, "related_edges")


# ============================================================
# 验证查询（只读）
# ============================================================

def count_entities(conn, label: Optional[str] = None) -> int:
    cypher = f"MATCH (n:{label or 'Entity'}) RETURN count(n) AS c"
    rows = conn.execute(cypher)
    return int(rows[0]["c"]) if rows else 0


def count_related_edges(conn) -> int:
    rows = conn.execute(f"MATCH ()-[r:{RELATED_TO}]->() RETURN count(r) AS c")
    return int(rows[0]["c"]) if rows else 0


def node_label_distribution(conn) -> list[dict]:
    return conn.execute("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY c DESC")


def relationship_type_distribution(conn) -> list[dict]:
    return conn.execute("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS c ORDER BY c DESC")


def duplicate_check(conn, label: str, key: str) -> list[dict]:
    """按唯一键查重复（应为空）。"""
    cypher = (
        f"MATCH (n:{label}) WITH n.{key} AS k, count(*) AS c WHERE c > 1 "
        f"RETURN k, c"
    )
    return conn.execute(cypher)


def orphan_check(conn) -> list[dict]:
    """检查孤立节点（无任何关系），返回类型分布。"""
    return conn.execute(
        "MATCH (n) WHERE NOT (n)--() RETURN labels(n)[0] AS label, count(*) AS c "
        "ORDER BY c DESC"
    )
