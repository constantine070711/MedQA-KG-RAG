"""Phase 6 F4 — Neo4jGraphRetriever（结构化图检索 API）。

职责:
    - 从 Neo4j 提供结构化图检索（search / get_entity / neighbors / path / related / drugs_treating）
    - 实体名称优先复用 D4（load_entity_name_map / resolve_entity_name），失败回退 name 精确匹配
    - 关系类型 / 实体类型 / 方向经白名单校验后进入 Cypher（仅固定合法常量）
    - 全部查询参数化（用户输入经参数，不拼接）；LIMIT / max_hops 有界
    - 输出稳定结构: {"query", "count", "nodes", "edges", "triples", "resolution", ...}

不实现:
    - Evidence Fusion（F5）/ RAG 编排 / 修改数据库 / 修改 D4 / 修改 Schema
"""

import logging
from pathlib import Path
from typing import Optional

from src.config import PROJECT_ROOT
from src.medical_core.entities import EntityType
from src.medical_core.relations import RelationType
from src.neo4j_utils import get_neo4j_connection
from evaluation.metrics import load_entity_name_map, resolve_entity_name

logger = logging.getLogger(__name__)

# ---- 白名单（受控常量，唯一允许进入 Cypher 的类型/方向）----
VALID_LABELS = {t.value for t in EntityType} | {"Entity"}
VALID_REL_TYPES = {t.value for t in RelationType} | {"RELATED_TO"}
VALID_DIRECTIONS = {"outgoing", "incoming", "both"}

DEFAULT_LIMIT = 20
MAX_LIMIT = 100
DEFAULT_HOPS = 3
MAX_HOPS = 6

# direction → Cypher 模式片段（center 永远是查询实体所在端点）
# 注意：incoming 与 outgoing 用同一带向模式，但 center 落在关系不同端：
#   outgoing: (center)-[r]->(neighbor)；incoming: (neighbor)-[r]->(center)。
_DIRECTION = {
    "outgoing": {"pattern": "(a)-[r]->(b)", "center": "a", "neighbor": "b"},
    "incoming": {"pattern": "(a)-[r]->(b)", "center": "b", "neighbor": "a"},
    "both": {"pattern": "(a)-[r]-(b)", "center": "a", "neighbor": "b"},
}

DEFAULT_ENTITIES = PROJECT_ROOT / "data" / "entities" / "entities.json"


def _clamp_limit(limit) -> int:
    try:
        return max(1, min(int(limit), MAX_LIMIT))
    except Exception:
        return DEFAULT_LIMIT


def _clamp_hops(max_hops) -> int:
    try:
        return max(1, min(int(max_hops), MAX_HOPS))
    except Exception:
        return DEFAULT_HOPS


def _rel_filter(rel_type: Optional[str]) -> Optional[str]:
    """校验 relation_type → Cypher 片段（":TREATS" 或 ""）。非法返回 None。"""
    if rel_type is None:
        return ""
    if rel_type in VALID_REL_TYPES:
        return f":{rel_type}"
    return None


def _label_for(entity_type: Optional[str]) -> Optional[str]:
    """校验 entity_type → Neo4j label。非法返回 None。"""
    if entity_type is None:
        return None
    if entity_type in VALID_LABELS:
        return entity_type
    return None


def _empty_result(query: str, note: str = "") -> dict:
    return {
        "query": query,
        "count": 0,
        "nodes": [],
        "edges": [],
        "triples": [],
        "resolution": {},
        "note": note,
    }


# ============================================================
# Neo4jGraphRetriever
# ============================================================

class Neo4jGraphRetriever:
    """Neo4j 结构化图检索器（只读）。"""

    def __init__(self, conn=None, entities_path=None):
        self._conn = conn  # 可注入（测试用 mock / 真实连接）
        self._entities_path = Path(entities_path) if entities_path else DEFAULT_ENTITIES
        self._name_map: Optional[dict] = None

    # ----------------------------------------------------------
    # 基础设施
    # ----------------------------------------------------------

    @property
    def conn(self):
        if self._conn is None:
            self._conn = get_neo4j_connection()
        if not self._conn.is_connected:
            self._conn.connect()
        return self._conn

    def _get_name_map(self) -> dict:
        if self._name_map is None:
            self._name_map = load_entity_name_map(str(self._entities_path))
        return self._name_map

    def _resolve_name(self, name: str) -> tuple[str, str]:
        """D4 解析名称 → (canonical, resolved_by)。resolved_by ∈ {"d4","exact"}。"""
        if not name or not name.strip():
            return name, "exact"
        canonical = resolve_entity_name(name, self._get_name_map())
        if canonical and canonical != name:
            return canonical, "d4"
        return name, "exact"

    @staticmethod
    def _names_for_resolution(name: str, canonical: str) -> list[str]:
        """去重保留 [原始名, canonical]，用于兼容 curated 中文名与 LightRAG 英文原始名。"""
        return list(dict.fromkeys([name, canonical]))

    # ----------------------------------------------------------
    # 序列化
    # ----------------------------------------------------------

    @staticmethod
    def _node_to_dict(n, labels=None) -> dict:
        props = dict(n) if hasattr(n, "items") else dict(n or {})
        if labels is None:
            labels = list(getattr(n, "labels", ())) if hasattr(n, "labels") else []
        entity_id = props.get("entity_id") or props.get("name") or ""
        return {
            "id": entity_id,
            "name": props.get("name", ""),
            "entity_type": props.get("entity_type", ""),
            "labels": list(labels),
            "properties": props,
        }

    @staticmethod
    def _edge_to_dict(src: str, tgt: str, rel_type: str, props: dict) -> dict:
        return {
            "source": src,
            "target": tgt,
            "relation_type": rel_type,
            "properties": props or {},
        }

    @staticmethod
    def _rel_to_dict(rel) -> dict:
        """neo4j Relationship → edge dict（使用真实方向）。"""
        props = dict(rel) if hasattr(rel, "items") else dict(rel or {})
        nodes = list(getattr(rel, "nodes", ())) or []
        src = getattr(nodes[0], "get", lambda k, d="": d)("name", "") if nodes else ""
        tgt = getattr(nodes[1], "get", lambda k, d="": d)("name", "") if len(nodes) > 1 else ""
        return Neo4jGraphRetriever._edge_to_dict(
            src, tgt, getattr(rel, "type", ""), props
        )

    def _build_result(self, query: str, nodes=None, edges=None, triples=None,
                      resolution=None, note="") -> dict:
        nodes = nodes or []
        edges = edges or []
        triples = triples or []
        return {
            "query": query,
            "count": len(edges) if edges else len(nodes),
            "nodes": nodes,
            "edges": edges,
            "triples": triples,
            "resolution": resolution or {},
            "note": note,
        }

    # ----------------------------------------------------------
    # API: search
    # ----------------------------------------------------------

    def search_entities(self, query: str, entity_type: Optional[str] = None,
                        limit: int = DEFAULT_LIMIT) -> dict:
        """按名称/别名模糊搜索实体（可选 entity_type 过滤）。"""
        limit = _clamp_limit(limit)
        qid = f"search_entities:{query}"
        label = _label_for(entity_type)
        if entity_type is not None and label is None:
            return _empty_result(qid, f"invalid entity_type={entity_type}")

        label_clause = f" AND labels(n)[0] = '{label}'" if label else ""
        cypher = (
            "MATCH (n) "
            "WHERE (n.name CONTAINS $q OR ANY(a IN n.aliases WHERE a CONTAINS $q)) "
            f"{label_clause} "
            "RETURN n, labels(n) AS lbl LIMIT $limit"
        )
        rows = self.conn.execute(cypher, {"q": query, "limit": limit})
        nodes = [self._node_to_dict(r.get("n"), r.get("lbl")) for r in rows]
        return self._build_result(qid, nodes=nodes)

    # ----------------------------------------------------------
    # API: get_entity
    # ----------------------------------------------------------

    def get_entity(self, name: str) -> dict:
        """按名称获取实体（D4 resolve 优先，回退精确匹配）。"""
        qid = f"get_entity:{name}"
        canonical, resolved_by = self._resolve_name(name)
        names = self._names_for_resolution(name, canonical)
        cypher = (
            "MATCH (n) WHERE n.name IN $names "
            "RETURN n, labels(n) AS lbl LIMIT $limit"
        )
        rows = self.conn.execute(cypher, {"names": names, "limit": 10})
        nodes = [self._node_to_dict(r.get("n"), r.get("lbl")) for r in rows]
        resolution = {"input": name, "canonical": canonical, "resolved_by": resolved_by}
        return self._build_result(qid, nodes=nodes, resolution=resolution)

    # ----------------------------------------------------------
    # API: get_neighbors
    # ----------------------------------------------------------

    def get_neighbors(self, name: str, relation_type: Optional[str] = None,
                      direction: str = "both", limit: int = DEFAULT_LIMIT) -> dict:
        """查询一跳邻居。direction ∈ {outgoing, incoming, both}。"""
        limit = _clamp_limit(limit)
        qid = f"get_neighbors:{name}"
        if direction not in VALID_DIRECTIONS:
            return _empty_result(qid, f"invalid direction={direction}")
        rel = _rel_filter(relation_type)
        if relation_type is not None and rel is None:
            return _empty_result(qid, f"invalid relation_type={relation_type}")

        canonical, resolved_by = self._resolve_name(name)
        names = self._names_for_resolution(name, canonical)
        d = _DIRECTION[direction]
        pattern = d["pattern"].replace("[r]", f"[r{rel}]") if rel else d["pattern"]
        center, neighbor = d["center"], d["neighbor"]

        cypher = (
            f"MATCH {pattern} "
            f"WHERE {center}.name IN $names "
            f"RETURN startNode(r).name AS r_src, endNode(r).name AS r_tgt, "
            f"type(r) AS rt, properties(r) AS rp, {neighbor} AS nb, labels({neighbor}) AS lbl "
            f"LIMIT $limit"
        )
        rows = self.conn.execute(cypher, {"names": names, "limit": limit})

        nodes, edges, triples = [], [], []
        for r in rows:
            edges.append(self._edge_to_dict(
                r.get("r_src", ""), r.get("r_tgt", ""), r.get("rt", ""), r.get("rp", {})
            ))
            triples.append([r.get("r_src", ""), r.get("rt", ""), r.get("r_tgt", "")])
            nb = r.get("nb")
            if nb is not None:
                node = self._node_to_dict(nb, r.get("lbl"))
                if not any(n["id"] == node["id"] for n in nodes):
                    nodes.append(node)

        resolution = {"input": name, "canonical": canonical, "resolved_by": resolved_by}
        return self._build_result(qid, nodes=nodes, edges=edges, triples=triples,
                                  resolution=resolution)

    # ----------------------------------------------------------
    # API: get_path
    # ----------------------------------------------------------

    def get_path(self, source: str, target: str, max_hops: int = DEFAULT_HOPS) -> dict:
        """查询两实体间有限长度最短路径（有界，无无限遍历）。"""
        hops = _clamp_hops(max_hops)
        qid = f"get_path:{source}->{target}"
        src, src_by = self._resolve_name(source)
        tgt, tgt_by = self._resolve_name(target)
        src_names = self._names_for_resolution(source, src)
        tgt_names = self._names_for_resolution(target, tgt)

        cypher = (
            f"MATCH p = shortestPath((a {{name: $src}})-[*1..{hops}]-(b {{name: $tgt}})) "
            "RETURN p LIMIT 1"
        )
        # 变量长度上界是受控 int（clamp 后），非用户字符串
        rows = self.conn.execute(cypher, {"src": src_names[0], "tgt": tgt_names[0]})

        if not rows or rows[0].get("p") is None:
            return _empty_result(qid, f"no path within max_hops={hops}")
        path = rows[0]["p"]
        nodes = [self._node_to_dict(n, list(getattr(n, "labels", ()))) for n in path.nodes]
        edges = [self._rel_to_dict(rel) for rel in path.relationships]
        triples = [[e["source"], e["relation_type"], e["target"]] for e in edges]
        resolution = {"source": source, "target": target,
                      "source_resolved_by": src_by, "target_resolved_by": tgt_by,
                      "max_hops": hops}
        return self._build_result(qid, nodes=nodes, edges=edges, triples=triples,
                                  resolution=resolution)

    # ----------------------------------------------------------
    # API: get_related
    # ----------------------------------------------------------

    def get_related(self, name: str, relation_type: Optional[str] = None,
                    limit: int = DEFAULT_LIMIT) -> dict:
        """获取与实体相关的结构化关系（双向，返回 nodes+edges+triples）。"""
        result = self.get_neighbors(name, relation_type=relation_type,
                                    direction="both", limit=limit)
        result["query"] = f"get_related:{name}"
        return result

    # ----------------------------------------------------------
    # API: get_drugs_treating_symptom
    # ----------------------------------------------------------

    def get_drugs_treating_symptom(self, symptom: str, limit: int = DEFAULT_LIMIT) -> dict:
        """查询治疗某症状的药物（诚实反映覆盖）。

        当前 curated 层无 Symptom 实体，TREATS 目标为 Disease；症状名通常无法命中 → 空结果 +
        覆盖说明。若目标恰为 Disease 名，可返回真实 Drug。
        """
        limit = _clamp_limit(limit)
        qid = f"get_drugs_treating_symptom:{symptom}"
        cypher = (
            "MATCH (d:Drug)-[r:TREATS]->(t) "
            "WHERE t.name = $symptom "
            "RETURN d, labels(d) AS lbl, t.name AS target LIMIT $limit"
        )
        rows = self.conn.execute(cypher, {"symptom": symptom, "limit": limit})
        nodes = [self._node_to_dict(r.get("d"), r.get("lbl")) for r in rows]
        result = self._build_result(qid, nodes=nodes)
        result["coverage_note"] = (
            "当前 curated 层无 Symptom 实体；TREATS 目标为 Disease。"
            "症状→药物查询通常返回空；需 Symptom 实体层或经 Disease 层查询。"
        )
        return result
