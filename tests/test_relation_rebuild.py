# -*- coding: utf-8 -*-
"""Phase 6 F3 — curated 临床关系重建测试（stdlib unittest）。

覆盖:
    - .md 结构化字段解析（unit）
    - D4 实体解析成功 / unresolved 处理（unit）
    - 占位符正确跳过（unit，合成 .md）
    - 自环剔除（unit，真实数据 "哮喘" 别名在 "哮喘持续状态" 假阳性）
    - 关系类型合法性 / 不生成未经数据支持的关系（unit）
    - 导入 MERGE 幂等（integration，真实 Neo4j）

运行:
    python -m unittest tests.test_relation_rebuild -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.medical_core.relations import RelationType
from src.neo4j_utils import get_neo4j_connection
from src.relation_rebuild import (
    parse_curated_relations,
    import_relations,
    is_placeholder_text,
    VALID_REL_TYPES,
)
from src.graph_import import count_entities, relationship_type_distribution

RAW_DIR = _PROJECT_ROOT / "data" / "raw"
ENTITIES_PATH = _PROJECT_ROOT / "data" / "entities" / "entities.json"


# ============================================================
# 1. 解析（unit）
# ============================================================

class TestCuratedRelationParse(unittest.TestCase):
    def test_parse_yields_expected_relations(self):
        parsed = parse_curated_relations(RAW_DIR, ENTITIES_PATH)
        st = parsed["stats"]
        self.assertEqual(st["resolved_relations"], 9)
        self.assertEqual(st["by_rel_type"], {"TREATS": 6, "HAS_COMPLICATION": 3})

    def test_all_relations_valid_types(self):
        parsed = parse_curated_relations(RAW_DIR, ENTITIES_PATH)
        for rel in parsed["relations"]:
            self.assertIn(rel["rel_type"], VALID_REL_TYPES)
            self.assertIn(rel["rel_type"], {t.value for t in RelationType})

    def test_all_relations_two_curated_endpoints(self):
        parsed = parse_curated_relations(RAW_DIR, ENTITIES_PATH)
        for rel in parsed["relations"]:
            self.assertTrue(rel["subject"])
            self.assertTrue(rel["object"])
            self.assertNotEqual(rel["subject"], rel["object"])  # 无自环

    def test_no_unsupported_relation_types(self):
        # 当前数据只支持 TREATS / HAS_COMPLICATION（两端 curated）
        parsed = parse_curated_relations(RAW_DIR, ENTITIES_PATH)
        produced = set(parsed["stats"]["by_rel_type"].keys())
        self.assertTrue(produced <= {"TREATS", "HAS_COMPLICATION"}, produced)

    def test_self_relation_skipped(self):
        # "哮喘" 别名在 "哮喘持续状态" 中造成假阳性 → 自环剔除
        parsed = parse_curated_relations(RAW_DIR, ENTITIES_PATH)
        self.assertEqual(parsed["stats"]["self_skipped"], 1)
        self.assertEqual(parsed["self_skipped"][0]["subject"],
                         parsed["self_skipped"][0]["object"])

    def test_unresolved_mentions_recorded(self):
        # 症状/禁忌症等未 curated 目标如实记录（不静默丢弃）
        parsed = parse_curated_relations(RAW_DIR, ENTITIES_PATH)
        self.assertGreaterEqual(parsed["stats"]["unresolved_mentions"], 1)
        for u in parsed["unresolved"]:
            self.assertIn("raw", u)

    def test_d4_resolve_success(self):
        parsed = parse_curated_relations(RAW_DIR, ENTITIES_PATH)
        self.assertEqual(parsed["stats"]["d4_resolve_fail"], 0)
        self.assertGreater(parsed["stats"]["d4_resolve_success"], 0)

    def test_treats_relations_present(self):
        parsed = parse_curated_relations(RAW_DIR, ENTITIES_PATH)
        treats = {(r["subject"], r["object"]) for r in parsed["relations"]
                  if r["rel_type"] == "TREATS"}
        self.assertIn(("二甲双胍", "2型糖尿病"), treats)
        self.assertIn(("硝苯地平", "高血压"), treats)


class TestPlaceholderIsolation(unittest.TestCase):
    def test_placeholder_text_recognized(self):
        for text in ("见正文症状章节", "详见正文", "见相关并发症章节"):
            self.assertTrue(is_placeholder_text(text))

    def test_normal_text_not_placeholder(self):
        self.assertFalse(is_placeholder_text("多饮、多尿、多食、体重下降"))
        self.assertFalse(is_placeholder_text("根除幽门螺杆菌：四联疗法"))

    def test_synthetic_placeholder_md_skipped(self):
        # 合成含占位符章节的 .md（实体名为真实 curated 实体，否则整文件在章节解析前被跳过）
        # 验证占位符章节正确跳过、不产生关系
        md = (
            "---\nentity_type: \"disease\"\nname: \"2型糖尿病\"\n---\n\n"
            "# 2型糖尿病\n\n"
            "## 常见症状\n\n见正文症状章节\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "diseases").mkdir()
            (tmp_path / "diseases" / "test.md").write_text(md, encoding="utf-8")
            parsed = parse_curated_relations(tmp_path, ENTITIES_PATH)
            self.assertEqual(parsed["stats"]["placeholder_skipped"], 1)
            # 占位符不产生任何关系
            self.assertEqual(len(parsed["relations"]), 0)


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
class TestRelationImportIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = get_neo4j_connection()
        cls.conn.connect()

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_import_idempotent(self):
        parsed = parse_curated_relations(RAW_DIR, ENTITIES_PATH)
        r1 = import_relations(self.conn, parsed["relations"], batch_size=100)
        r2 = import_relations(self.conn, parsed["relations"], batch_size=100)
        for rel_type, res in r1.items():
            self.assertEqual(res["ok"], res["total"])
        for rel_type, res in r2.items():
            self.assertEqual(res["ok"], res["total"])
        # 幂等：TREATS 6 + HAS_COMPLICATION 3，无重复
        dist = {row["rel"]: row["c"] for row in relationship_type_distribution(self.conn)}
        self.assertEqual(dist.get("TREATS", 0), 6)
        self.assertEqual(dist.get("HAS_COMPLICATION", 0), 3)
        dup = self.conn.execute(
            "MATCH (a)-[r]->(b) WITH type(r) AS t, a.name AS s, b.name AS o, count(*) AS c "
            "WHERE c > 1 RETURN t, s, o, c"
        )
        self.assertEqual(len(dup), 0)

    def test_f2_raw_layer_untouched(self):
        # F2 原始层不被 F3 破坏
        self.assertEqual(count_entities(self.conn, "Entity"), 380)
        related = self.conn.execute("MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS c")[0]["c"]
        self.assertEqual(related, 422)

    def test_relation_endpoints_exist(self):
        # 每个 F3 关系两端节点存在（无孤立关系端点）
        rows = self.conn.execute(
            "MATCH (a)-[r]->(b) WHERE type(r) IN ['TREATS','HAS_COMPLICATION'] "
            "RETURN a.name AS s, b.name AS o"
        )
        for r in rows:
            self.assertTrue(r["s"] and r["o"])


if __name__ == "__main__":
    unittest.main()
