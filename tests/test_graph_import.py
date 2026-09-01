# -*- coding: utf-8 -*-
"""Phase 6 F2 — 图数据导入测试（stdlib unittest）。

覆盖:
    - 解析（unit，对真实输入文件 + 纯函数）
    - 占位符关系识别（unit）
    - 导入幂等（integration，真实 Neo4j 容器）

分层:
    TestGraphParse / TestPlaceholderRelation — 纯单元，可离线
    TestGraphImportIntegration — 需 Neo4j 容器（Bolt 可达才执行，否则 skip）

运行:
    python -m unittest tests.test_graph_import -v

不修改任何源数据（graphml / entities.json）；不硬编码题号 / Q016-Q028 / fixture 特例。
"""

import sys
import unittest
from pathlib import Path

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.neo4j_utils import get_neo4j_connection
from src.graph_import import (
    parse_graphml,
    parse_curated,
    is_placeholder_relation,
    import_entity_nodes,
    import_related_edges,
    import_curated_nodes,
    count_entities,
    count_related_edges,
    duplicate_check,
)

GRAPHML_PATH = _PROJECT_ROOT / "lightrag_cache" / "graph_chunk_entity_relation.graphml"
ENTITIES_PATH = _PROJECT_ROOT / "data" / "entities" / "entities.json"


# ============================================================
# 1. 解析（unit）
# ============================================================

class TestGraphParse(unittest.TestCase):
    def test_parse_graphml_real_counts(self):
        parsed = parse_graphml(GRAPHML_PATH)
        st = parsed["stats"]
        self.assertEqual(st["source_nodes"], 380)
        self.assertEqual(st["source_edges"], 422)
        self.assertEqual(st["parsed_nodes"], 380)
        self.assertEqual(st["parsed_edges"], 422)
        self.assertEqual(st["skipped_nodes"], 0)
        self.assertEqual(st["skipped_edges"], 0)

    def test_parse_graphml_node_fields(self):
        parsed = parse_graphml(GRAPHML_PATH)
        self.assertTrue(len(parsed["nodes"]) > 0)
        n = parsed["nodes"][0]
        for key in ("entity_id", "name", "entity_type", "description", "source_id", "file_path"):
            self.assertIn(key, n)

    def test_parse_graphml_edge_fields(self):
        parsed = parse_graphml(GRAPHML_PATH)
        self.assertTrue(len(parsed["edges"]) > 0)
        e = parsed["edges"][0]
        for key in ("src", "tgt", "weight", "description", "keywords", "source_id", "file_path"):
            self.assertIn(key, e)

    def test_parse_graphml_entity_type_generic(self):
        # LightRAG 实体类型为通用本体，非医疗类型 —— 原始层如实保留
        parsed = parse_graphml(GRAPHML_PATH)
        types = set(n["entity_type"] for n in parsed["nodes"])
        self.assertTrue(types, "entity_type 不应为空")

    def test_parse_curated_real_counts(self):
        parsed = parse_curated(ENTITIES_PATH)
        st = parsed["stats"]
        self.assertEqual(st["source_entities"], 14)
        self.assertEqual(st["imported_nodes"], 14)
        self.assertEqual(st["skipped_entities"], 0)
        self.assertEqual(st["source_relations"], 42)
        self.assertEqual(st["placeholder_relations"], 42)
        self.assertEqual(st["imported_relations"], 0)  # F2 不导入关系

    def test_parse_curated_type_distribution(self):
        parsed = parse_curated(ENTITIES_PATH)
        self.assertEqual(parsed["stats"]["node_type_distribution"],
                         {"Disease": 9, "Drug": 5})

    def test_parse_curated_standard_code_coverage(self):
        parsed = parse_curated(ENTITIES_PATH)
        self.assertEqual(parsed["stats"]["standard_code_coverage"], 14)

    def test_parse_curated_no_relations(self):
        # F2 占位符关系不进入导入列表
        parsed = parse_curated(ENTITIES_PATH)
        self.assertEqual(parsed["relations"], [])


class TestPlaceholderRelation(unittest.TestCase):
    def test_placeholder_recognized(self):
        for obj in ("见正文症状章节", "见正文治疗方法章节", "见相关并发症章节"):
            self.assertTrue(is_placeholder_relation({"object": obj}))

    def test_non_placeholder_not_flagged(self):
        self.assertFalse(is_placeholder_relation({"object": "2型糖尿病"}))
        self.assertFalse(is_placeholder_relation({"object": "二甲双胍"}))


# ============================================================
# 2. 导入（integration）
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
class TestGraphImportIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = get_neo4j_connection()
        cls.conn.connect()

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_entity_nodes_idempotent(self):
        parsed = parse_graphml(GRAPHML_PATH)
        r1 = import_entity_nodes(self.conn, parsed["nodes"], batch_size=200)
        r2 = import_entity_nodes(self.conn, parsed["nodes"], batch_size=200)
        self.assertEqual(r1["ok"], 380)
        self.assertEqual(r2["ok"], 380)
        self.assertEqual(count_entities(self.conn, "Entity"), 380)  # 幂等不重复
        self.assertEqual(duplicate_check(self.conn, "Entity", "entity_id"), [])

    def test_related_edges_idempotent(self):
        parsed = parse_graphml(GRAPHML_PATH)
        r1 = import_related_edges(self.conn, parsed["edges"], batch_size=200)
        r2 = import_related_edges(self.conn, parsed["edges"], batch_size=200)
        self.assertEqual(r1["ok"], 422)
        self.assertEqual(r2["ok"], 422)
        self.assertEqual(count_related_edges(self.conn), 422)  # 幂等不重复

    def test_curated_nodes_idempotent(self):
        parsed = parse_curated(ENTITIES_PATH)
        by_label = {}
        for n in parsed["nodes"]:
            by_label.setdefault(n["label"], []).append(n)
        for label, items in by_label.items():
            import_curated_nodes(self.conn, label, items, batch_size=100)
        self.assertEqual(count_entities(self.conn, "Disease"), 9)
        self.assertEqual(count_entities(self.conn, "Drug"), 5)
        self.assertEqual(duplicate_check(self.conn, "Disease", "name"), [])
        self.assertEqual(duplicate_check(self.conn, "Drug", "name"), [])

    def test_curated_standard_code_preserved(self):
        rows = self.conn.execute(
            'MATCH (n:Disease {name: "支气管哮喘"}) RETURN n.standard_code AS code LIMIT 1'
        )
        self.assertEqual(rows[0]["code"], "J45")


if __name__ == "__main__":
    unittest.main()
