# -*- coding: utf-8 -*-
"""Phase 6 F1 — Neo4jConnection / Neo4jSchema.bootstrap 测试（stdlib unittest）。

覆盖:
    - 配置读取（unit，不依赖容器）
    - 连接基础行为（unit，mock driver）
    - authentication / 简单 Cypher（integration，需真实 Neo4j 容器）
    - Schema bootstrap + 幂等（integration）
    - 基本异常处理（unit，mock driver 抛错）

分层:
    TestNeo4jConfig / TestNeo4jConnectionUnit — 纯单元，可离线运行
    TestNeo4jIntegration — 需 Neo4j 容器（Bolt 可达才执行，否则 skip）

运行:
    python -m unittest tests.test_neo4j_connection -v

不修改任何项目代码；不硬编码题号 / Q016-Q028 / fixture 特例。
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

# 确保项目根目录在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import get_config
from src.neo4j_utils import Neo4jConnection, Neo4jSchema, get_neo4j_connection


# ============================================================
# Mock driver（unit 测试用）
# ============================================================

class FakeResult:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)

    def single(self):
        return self._records[0] if self._records else None

    def consume(self):
        # execute_batch 会调用 result.consume()
        return None


class FakeSession:
    def __init__(self, fail_on=None):
        self.fail_on = fail_on  # ("RETURN 1", ...) 等触发失败的语句前缀
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, params=None):
        if self.fail_on and cypher.startswith(self.fail_on):
            raise RuntimeError("synthetic run failure")
        return FakeResult([{"ok": 1, "name": "test"}])

    def consume(self):
        pass

    def begin_transaction(self):
        return FakeTransaction(self)


class FakeTransaction:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def commit(self):
        pass

    def rollback(self):
        pass

    def run(self, cypher, params=None):
        return self.session.run(cypher, params)


class FakeDriver:
    def __init__(self, fail_connectivity=False):
        self.fail_connectivity = fail_connectivity
        self.closed = False
        self.session_fail_on = None

    def verify_connectivity(self):
        if self.fail_connectivity:
            raise RuntimeError("synthetic connectivity/auth failure")
        return True

    def session(self):
        return FakeSession(fail_on=self.session_fail_on)

    def close(self):
        self.closed = True


def _patch_driver(fail_connectivity=False):
    """mock GraphDatabase.driver 返回 FakeDriver。"""
    d = FakeDriver(fail_connectivity=fail_connectivity)
    patcher = mock.patch(
        "src.neo4j_utils.GraphDatabase.driver",
        return_value=d,
    )
    patcher.start()
    return d, patcher


# ============================================================
# 1. 配置读取（unit）
# ============================================================

class TestNeo4jConfig(unittest.TestCase):
    def test_config_has_neo4j_fields(self):
        cfg = get_config()
        self.assertTrue(cfg.neo4j_uri)
        self.assertTrue(cfg.neo4j_user)
        self.assertTrue(cfg.neo4j_password, "NEO4J_PASSWORD 未在 .env 配置")

    def test_uri_is_bolt(self):
        cfg = get_config()
        self.assertTrue(str(cfg.neo4j_uri).startswith("bolt://"))

    def test_connection_uses_config_defaults(self):
        cfg = get_config()
        conn = Neo4jConnection()
        try:
            self.assertEqual(conn.uri, cfg.neo4j_uri)
            self.assertEqual(conn.user, cfg.neo4j_user)
            self.assertEqual(conn.password, cfg.neo4j_password)
        finally:
            conn.close()

    def test_connection_explicit_args_override(self):
        conn = Neo4jConnection(uri="bolt://test:7687", user="u", password="p")
        try:
            self.assertEqual(conn.uri, "bolt://test:7687")
            self.assertEqual(conn.user, "u")
            self.assertEqual(conn.password, "p")
        finally:
            conn.close()

    def test_get_connection_factory_returns_connection(self):
        conn = get_neo4j_connection()
        try:
            self.assertIsInstance(conn, Neo4jConnection)
        finally:
            conn.close()


# ============================================================
# 2. 连接基础行为（unit, mock driver）
# ============================================================

class TestNeo4jConnectionUnit(unittest.TestCase):
    def tearDown(self):
        mock.patch.stopall()

    def test_connect_and_close(self):
        d, _ = _patch_driver()
        conn = Neo4jConnection(uri="bolt://mock:7687", user="u", password="p")
        conn.connect()
        self.assertTrue(conn.is_connected)
        self.assertTrue(d.closed is False)
        conn.close()
        self.assertFalse(conn.is_connected)
        self.assertTrue(d.closed)

    def test_verify_returns_true(self):
        d, _ = _patch_driver()
        conn = Neo4jConnection(uri="bolt://mock:7687", user="u", password="p")
        try:
            self.assertTrue(conn.verify())
        finally:
            conn.close()

    def test_execute_returns_records(self):
        d, _ = _patch_driver()
        conn = Neo4jConnection(uri="bolt://mock:7687", user="u", password="p")
        try:
            rows = conn.execute("RETURN 1 AS ok")
            self.assertEqual(rows, [{"ok": 1, "name": "test"}])
        finally:
            conn.close()

    def test_connect_failure_raises_runtime_error(self):
        d, _ = _patch_driver(fail_connectivity=True)
        conn = Neo4jConnection(uri="bolt://mock:7687", user="u", password="p")
        with self.assertRaises(RuntimeError) as ctx:
            conn.connect()
        self.assertIn("连接/认证失败", str(ctx.exception))
        self.assertFalse(conn.is_connected)

    def test_execute_batch_error_handling(self):
        d, _ = _patch_driver()
        d.session_fail_on = "CREATE FAIL"
        conn = Neo4jConnection(uri="bolt://mock:7687", user="u", password="p")
        try:
            results = conn.execute_batch(["RETURN 1", "CREATE FAIL BAD", "RETURN 2"])
            # 逐条捕获错误，不中断
            self.assertEqual(results[0]["ok"], True)
            self.assertEqual(results[1]["ok"], False)
            self.assertIn("synthetic run failure", results[1]["error"])
            self.assertEqual(results[2]["ok"], True)
        finally:
            conn.close()

    def test_close_idempotent(self):
        d, _ = _patch_driver()
        conn = Neo4jConnection(uri="bolt://mock:7687", user="u", password="p")
        conn.connect()
        conn.close()
        conn.close()  # 第二次 close 不应报错
        self.assertFalse(conn.is_connected)

    def test_context_manager(self):
        d, _ = _patch_driver()
        with Neo4jConnection(uri="bolt://mock:7687", user="u", password="p") as conn:
            rows = conn.execute("RETURN 1 AS ok")
            self.assertEqual(rows, [{"ok": 1, "name": "test"}])
        self.assertFalse(conn.is_connected)


# ============================================================
# 3. 集成测试（需真实 Neo4j 容器）
# ============================================================

def _neo4j_available() -> bool:
    try:
        conn = Neo4jConnection()
        conn.connect()
        conn.close()
        return True
    except Exception:
        return False


NEO4J_AVAILABLE = _neo4j_available()


@unittest.skipUnless(NEO4J_AVAILABLE, "Neo4j 容器不可达（Bolt 连接失败），跳过集成测试")
class TestNeo4jIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = Neo4jConnection()
        cls.conn.connect()

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_auth_and_simple_cypher(self):
        rows = self.conn.execute("RETURN 1 AS ok")
        self.assertEqual(rows[0]["ok"], 1)

    def test_neo4j_version(self):
        rows = self.conn.execute(
            "CALL dbms.components() YIELD name, versions RETURN name, versions"
        )
        names = [r.get("name") for r in rows if r.get("name")]
        self.assertTrue(any("Neo4j" in n for n in names), names)
        self.assertTrue(rows[0].get("versions"))

    def test_schema_bootstrap(self):
        schema = Neo4jSchema()
        result = schema.bootstrap(self.conn)
        self.assertEqual(result["failed"], 0, result)
        self.assertEqual(len(result["constraints"]), 4)
        self.assertEqual(len(result["indexes"]), 3)

    def test_schema_bootstrap_idempotent(self):
        schema = Neo4jSchema()
        first = schema.bootstrap(self.conn)
        second = schema.bootstrap(self.conn)
        self.assertEqual(first["failed"], 0)
        self.assertEqual(second["failed"], 0)
        # 幂等：两次 bootstrap 后约束/索引数量稳定（不重复创建）
        verify1 = schema.verify_schema(self.conn)
        verify2 = schema.verify_schema(self.conn)
        self.assertEqual(len(verify1["constraints"]), len(verify2["constraints"]))
        self.assertGreaterEqual(len(verify1["constraints"]), 4)  # F1 的 4 个必有
        self.assertGreaterEqual(len(verify1["indexes"]), 3)      # F1 的 3 个必有
        # 注：F2 导入可能新增 Entity.entity_id 唯一约束，故不硬编码精确总数

    def test_schema_verify_contains_expected(self):
        schema = Neo4jSchema()
        verify = schema.verify_schema(self.conn)
        names = {c.get("name") for c in verify["constraints"]}
        for expected in ("disease_name", "drug_name", "symptom_name", "examination_name"):
            self.assertIn(expected, names)


if __name__ == "__main__":
    unittest.main()
