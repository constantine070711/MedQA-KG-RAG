# -*- coding: utf-8 -*-
"""Phase 6 F4 — Neo4jGraphRetriever 测试（stdlib unittest）。

覆盖:
    - search_entities / +entity_type / get_entity
    - get_neighbors（outgoing/incoming/both / relation_type 过滤）
    - get_path（max_hops 有界）/ get_related / get_drugs_treating_symptom
    - D4 解析成功 / 失败；空结果；查询异常
    - 参数化 Cypher（用户输入不拼接）
    - 非法 relation_type / direction 拒绝；limit / max_hops 边界
    - 返回结构完整性

分层:
    unit（mock driver，不依赖 DB）+ integration（真实 Neo4j，只读，不修改数据）

运行:
    python -m unittest tests.test_neo4j_graph -v
"""

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.neo4j_graph import (
    Neo4jGraphRetriever,
    VALID_DIRECTIONS,
    MAX_LIMIT,
    MAX_HOPS,
)
from src.neo4j_utils import get_neo4j_connection


# ============================================================
# Fake 连接（unit 用）
# ============================================================

class FakeConn:
    """mock Neo4jConnection：execute 返回 handler 结果或抛异常。"""

    def __init__(self, handler=None, raise_on_contains=None):
        self.handler = handler
        self.raise_on_contains = raise_on_contains
        self.is_connected = True
        self.calls: list[tuple[str, dict]] = []

    def connect(self):
        return self

    def execute(self, cypher, params=None):
        self.calls.append((cypher, params or {}))
        if self.raise_on_contains and self.raise_on_contains in cypher:
            raise RuntimeError(f"synthetic error: {self.raise_on_contains}")
        if self.handler:
            return self.handler(cypher, params or {})
        return []


def _node_record(name, label="Disease", entity_type="disease", **extra):
    n = {"name": name, "entity_id": f"id:{name}", "entity_type": entity_type, **extra}
    return {"n": n, "lbl": [label]}


# ============================================================
# 1. 参数校验（unit，不触库）
# ============================================================

class TestValidation(unittest.TestCase):
    def setUp(self):
        self.conn = FakeConn()
        self.r = Neo4jGraphRetriever(conn=self.conn)

    def test_invalid_direction_rejected(self):
        res = self.r.get_neighbors("高血压", direction="banana")
        self.assertEqual(res["count"], 0)
        self.assertIn("invalid direction", res["note"])

    def test_invalid_relation_type_rejected(self):
        res = self.r.get_neighbors("高血压", relation_type="KILLS")
        self.assertEqual(res["count"], 0)
        self.assertIn("invalid relation_type", res["note"])

    def test_invalid_entity_type_rejected(self):
        res = self.r.search_entities("高", entity_type="Banana")
        self.assertEqual(res["count"], 0)
        self.assertIn("invalid entity_type", res["note"])

    def test_valid_direction_accepted(self):
        # 合法 direction 不应被拒（应触库查询而非返回 empty）
        self.r.get_neighbors("高血压", direction="incoming")
        self.assertGreater(len(self.conn.calls), 0)


# ============================================================
# 2. 输出结构 + D4（unit，mock 记录）
# ============================================================

class TestOutputAndResolution(unittest.TestCase):
    def test_get_entity_output_structure(self):
        conn = FakeConn(handler=lambda c, p: [_node_record("高血压")])
        r = Neo4jGraphRetriever(conn=conn)
        res = r.get_entity("高血压")
        for key in ("query", "count", "nodes", "edges", "triples", "resolution"):
            self.assertIn(key, res)
        self.assertEqual(len(res["nodes"]), 1)
        n = res["nodes"][0]
        for key in ("id", "name", "entity_type", "labels", "properties"):
            self.assertIn(key, n)

    def test_d4_resolve_success(self):
        conn = FakeConn(handler=lambda c, p: [_node_record("高血压")])
        r = Neo4jGraphRetriever(conn=conn)
        res = r.get_entity("Hypertension")
        self.assertEqual(res["resolution"]["canonical"], "高血压")
        self.assertEqual(res["resolution"]["resolved_by"], "d4")
        # 查询参数应含 canonical 与原始名
        params = conn.calls[0][1]
        self.assertIn("高血压", params["names"])

    def test_d4_resolve_fail_fallback_exact(self):
        conn = FakeConn(handler=lambda c, p: [])
        r = Neo4jGraphRetriever(conn=conn)
        res = r.get_entity("完全不存在实体XYZ")
        self.assertEqual(res["resolution"]["resolved_by"], "exact")
        self.assertEqual(res["count"], 0)  # 空结果，不猜测

    def test_empty_result_structure(self):
        conn = FakeConn(handler=lambda c, p: [])
        r = Neo4jGraphRetriever(conn=conn)
        res = r.search_entities("不存在xyz")
        self.assertEqual(res["count"], 0)
        self.assertEqual(res["nodes"], [])
        self.assertEqual(res["edges"], [])

    def test_query_exception_surfaces(self):
        conn = FakeConn(raise_on_contains="MATCH")
        r = Neo4jGraphRetriever(conn=conn)
        with self.assertRaises(RuntimeError):
            r.search_entities("高血压")


# ============================================================
# 3. 参数化 Cypher + 安全边界（unit）
# ============================================================

class TestCypherSafety(unittest.TestCase):
    def test_user_input_not_in_cypher(self):
        payload = "高血压') OR 1=1 --"
        conn = FakeConn(handler=lambda c, p: [])
        r = Neo4jGraphRetriever(conn=conn)
        r.search_entities(payload)
        cypher, params = conn.calls[0]
        self.assertNotIn(payload, cypher)          # 用户输入不进 Cypher 字符串
        self.assertEqual(params["q"], payload)      # 仅作为参数

    def test_name_not_interpolated(self):
        payload = "二甲双胍'; DETACH DELETE n; //"
        conn = FakeConn(handler=lambda c, p: [_node_record("x")])
        r = Neo4jGraphRetriever(conn=conn)
        r.get_neighbors(payload)
        cypher, params = conn.calls[0]
        self.assertNotIn(payload, cypher)
        self.assertIn(payload, params["names"])

    def test_limit_clamped(self):
        conn = FakeConn(handler=lambda c, p: [])
        r = Neo4jGraphRetriever(conn=conn)
        r.search_entities("高", limit=99999)
        params = conn.calls[0][1]
        self.assertLessEqual(params["limit"], MAX_LIMIT)

    def test_max_hops_clamped(self):
        conn = FakeConn(handler=lambda c, p: [])
        r = Neo4jGraphRetriever(conn=conn)
        r.get_path("a", "b", max_hops=999)
        cypher, _ = conn.calls[0]
        self.assertIn(f"..{MAX_HOPS}]", cypher)  # 上界受控

    def test_valid_directions_allowed(self):
        self.assertEqual(VALID_DIRECTIONS, {"outgoing", "incoming", "both"})


# ============================================================
# 4. 集成测试（真实 Neo4j，只读）
# ============================================================

def _neo4j_available() -> bool:
    try:
        conn = get_neo4j_connection()
        conn.connect()
        conn.close()
        return True
    except Exception:
        return False


NEO4J_AVAILABLE = _neo4j_available()


@unittest.skipUnless(NEO4J_AVAILABLE, "Neo4j 容器不可达，跳过集成测试")
class TestGraphRetrieverIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = Neo4jGraphRetriever()

    def test_db_counts_unchanged(self):
        conn = get_neo4j_connection()
        conn.connect()
        nodes = conn.execute("MATCH (n) RETURN count(n) AS c")[0]["c"]
        rels = conn.execute("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        conn.close()
        self.assertEqual(nodes, 394)
        self.assertEqual(rels, 431)

    def test_search_entities(self):
        res = self.r.search_entities("二甲双胍")
        names = {n["name"] for n in res["nodes"]}
        self.assertIn("二甲双胍", names)

    def test_search_entities_with_type(self):
        res = self.r.search_entities("高血", entity_type="Disease")
        self.assertTrue(all(n["labels"] == ["Disease"] for n in res["nodes"]))

    def test_get_entity_d4(self):
        res = self.r.get_entity("Hypertension")
        self.assertEqual(res["resolution"]["resolved_by"], "d4")
        self.assertTrue(any(n["name"] == "高血压" for n in res["nodes"]))

    def test_neighbors_direction_outgoing(self):
        res = self.r.get_neighbors("二甲双胍", relation_type="TREATS", direction="outgoing")
        self.assertIn(["二甲双胍", "TREATS", "2型糖尿病"], res["triples"])

    def test_neighbors_incoming(self):
        res = self.r.get_neighbors("高血压", direction="incoming", limit=50)
        self.assertTrue(all(t[2] == "高血压" for t in res["triples"]))

    def test_relation_filter_treats(self):
        res = self.r.get_neighbors("奥美拉唑", relation_type="TREATS", direction="outgoing")
        self.assertIn(["奥美拉唑", "TREATS", "慢性胃炎"], res["triples"])

    def test_has_complication(self):
        res = self.r.get_neighbors("高脂血症", relation_type="HAS_COMPLICATION", direction="outgoing")
        self.assertIn(["高脂血症", "HAS_COMPLICATION", "冠心病"], res["triples"])
        self.assertIn(["高脂血症", "HAS_COMPLICATION", "脑卒中"], res["triples"])

    def test_path_bounded(self):
        res = self.r.get_path("二甲双胍", "2型糖尿病")
        self.assertGreaterEqual(res["count"], 1)

    def test_get_related(self):
        res = self.r.get_related("高脂血症")
        self.assertGreater(res["count"], 0)
        self.assertIn("triples", res)

    def test_get_drugs_treating_symptom_empty(self):
        # 当前无 Symptom 实体 → 空结果 + 覆盖说明（不伪造）
        res = self.r.get_drugs_treating_symptom("头晕")
        self.assertEqual(res["count"], 0)
        self.assertIn("coverage_note", res)

    def test_get_drugs_treating_disease_target(self):
        # TREATS 目标为 Disease，传入 Disease 名可返回真实药物
        res = self.r.get_drugs_treating_symptom("2型糖尿病")
        self.assertGreaterEqual(res["count"], 1)


if __name__ == "__main__":
    unittest.main()
