# -*- coding: utf-8 -*-
"""Phase 6 F5 — Evidence Fusion 测试（纯 fixture，stdlib unittest，零 LLM / 零 DB）。

覆盖（对齐 F5 需求清单）:
    1.  extract_curated_mentions（canonical/别名/无匹配/空/最长匹配）
    2.  LightRAG-only 融合
    3.  Neo4j-only 融合
    4.  双路都有
    5.  重复实体跨源合并（sources 保留全部 provenance）
    6.  跨源关系合并（keywords == 结构化类型才合并；不确定不合并）
    7.  provenance sources 正确保留
    8.  全空结果
    9.  Neo4j 无匹配 + coverage_note 正确保留
    10. 截断优先保留高 priority
    11. 排序稳定性
    12. malformed / missing fields 不崩溃
    13. evidence_fusion.py 不 import LLM / embedding（静态扫描 import）
    14. orchestration mock：query_with_context() 最多调用一次
    15. reliability_decision.py 未修改（API 完整 / 无 qid / 无 F5 依赖）

运行:
    python -m unittest tests.test_evidence_fusion -v
"""

import asyncio
import re
import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evidence_fusion import (
    FusionConfig,
    EvidenceItem,
    KIND_NEO4J_TRIPLE,
    KIND_NEO4J_NODE,
    KIND_LIGHTRAG_ENTITY,
    KIND_LIGHTRAG_RELATION,
    KIND_LIGHTRAG_CHUNK,
    SOURCE_LIGHTRAG_VECTOR,
    SOURCE_LIGHTRAG_GRAPH,
    SOURCE_NEO4J_GRAPH,
    extract_curated_mentions,
    fuse_evidence,
    build_fusion_prompt,
    sort_evidence,
)
from scripts.run_graph_query import run_fusion_query

_PROJECT_ROOT_ABS = Path(__file__).resolve().parent.parent  # 项目根目录（静态扫描用）


# ============================================================
# Fixtures（纯数据，不触库）
# ============================================================

NAME_MAP = {
    "2型糖尿病": "2型糖尿病",
    "type 2 diabetes mellitus": "2型糖尿病",
    "t2dm": "2型糖尿病",
    "二甲双胍": "二甲双胍",
    "metformin": "二甲双胍",
    "高血压": "高血压",
    "hypertension": "高血压",
    "高脂血症": "高脂血症",
    "高血脂": "高脂血症",
}

LR_DATA = {
    "entities": [
        {"entity_name": "Type 2 Diabetes Mellitus", "entity_type": "concept",
         "description": "T2DM desc", "source_id": "c0", "file_path": "unknown_source"},
        {"entity_name": "Metformin", "entity_type": "concept",
         "description": "Metformin desc", "source_id": "c1", "file_path": "unknown_source"},
        {"entity_name": "Diabetes Diet", "entity_type": "concept",
         "description": "diet advice", "source_id": "c2", "file_path": "unknown_source"},
    ],
    "relationships": [
        {"src_id": "Metformin", "tgt_id": "Type 2 Diabetes Mellitus",
         "description": "metformin treats t2dm", "keywords": "treats", "weight": 1.0,
         "source_id": "r0", "file_path": "unknown_source"},
        {"src_id": "Type 2 Diabetes Mellitus", "tgt_id": "Diabetes Diet",
         "description": "diet recommendation", "keywords": "recommends", "weight": 1.0,
         "source_id": "r1", "file_path": "unknown_source"},
    ],
    "chunks": [
        {"reference_id": "", "content": "二甲双胍是2型糖尿病的一线用药",
         "file_path": "unknown_source", "chunk_id": "chunk-000"},
    ],
}

N4_DATA = {
    "nodes": [
        {"id": "二甲双胍", "name": "二甲双胍", "entity_type": "drug",
         "labels": ["Drug"], "properties": {"name": "二甲双胍"}},
        {"id": "2型糖尿病", "name": "2型糖尿病", "entity_type": "disease",
         "labels": ["Disease"], "properties": {"name": "2型糖尿病"}},
        {"id": "高脂血症", "name": "高脂血症", "entity_type": "disease",
         "labels": ["Disease"], "properties": {"name": "高脂血症"}},
    ],
    "edges": [
        {"source": "二甲双胍", "target": "2型糖尿病", "relation_type": "TREATS",
         "properties": {}},
    ],
    "triples": [
        ["二甲双胍", "TREATS", "2型糖尿病"],
        ["高脂血症", "HAS_COMPLICATION", "冠心病"],
    ],
    "resolution": {},
}


# ============================================================
# 1. extract_curated_mentions
# ============================================================

class TestExtractMentions(unittest.TestCase):
    def test_canonical_and_alias(self):
        self.assertEqual(
            extract_curated_mentions("二甲双胍与2型糖尿病的关系", NAME_MAP),
            ["二甲双胍", "2型糖尿病"],
        )

    def test_english_alias_resolved(self):
        self.assertEqual(
            extract_curated_mentions("How to treat Hypertension?", NAME_MAP),
            ["高血压"],
        )

    def test_alias_dedup(self):
        # 同一实体多个 surface 只返回一次
        res = extract_curated_mentions("Metformin 和 T2DM 有关系", NAME_MAP)
        self.assertEqual(res, ["二甲双胍", "2型糖尿病"])

    def test_no_match(self):
        self.assertEqual(extract_curated_mentions("今天天气怎么样", NAME_MAP), [])

    def test_empty_inputs(self):
        self.assertEqual(extract_curated_mentions("", NAME_MAP), [])
        self.assertEqual(extract_curated_mentions(None, NAME_MAP), [])
        self.assertEqual(extract_curated_mentions("高血压", {}), [])

    def test_longest_match_wins(self):
        # 同一位置优先更长匹配（不会把 "高血压" 拆成更短别名）
        res = extract_curated_mentions("高脂血症与高血压", NAME_MAP)
        self.assertIn("高脂血症", res)
        self.assertIn("高血压", res)


# ============================================================
# 2-6. fuse_evidence
# ============================================================

class TestFuseEvidence(unittest.TestCase):
    def test_lightrag_only(self):
        res = fuse_evidence(lightrag_data=LR_DATA, neo4j_data={},
                            entity_name_map=NAME_MAP)
        self.assertEqual(res["status"], "ok")
        self.assertFalse(res["coverage"]["neo4j_has_evidence"])
        kinds = {it["kind"] for it in res["evidence"]}
        self.assertNotIn(KIND_NEO4J_TRIPLE, kinds)
        self.assertTrue(all(
            SOURCE_NEO4J_GRAPH not in it["sources"] for it in res["evidence"]
        ))
        # lightrag source counts
        self.assertEqual(res["evidence_sources"]["lightrag_vector"], 1)
        self.assertEqual(res["evidence_sources"]["lightrag_graph"]["entities"], 3)

    def test_neo4j_only(self):
        res = fuse_evidence(lightrag_data={}, neo4j_data=N4_DATA,
                            entity_name_map=NAME_MAP)
        self.assertTrue(res["coverage"]["neo4j_has_evidence"])
        kinds = {it["kind"] for it in res["evidence"]}
        self.assertIn(KIND_NEO4J_TRIPLE, kinds)
        self.assertIn(KIND_NEO4J_NODE, kinds)
        self.assertTrue(all(
            SOURCE_NEO4J_GRAPH in it["sources"] for it in res["evidence"]
        ))

    def test_both_sources(self):
        res = fuse_evidence(lightrag_data=LR_DATA, neo4j_data=N4_DATA,
                            entity_name_map=NAME_MAP)
        self.assertTrue(res["coverage"]["neo4j_has_evidence"])
        by_source = res["summary"]["by_source"]
        self.assertIn(SOURCE_NEO4J_GRAPH, by_source)
        self.assertIn(SOURCE_LIGHTRAG_GRAPH, by_source)
        self.assertIn(SOURCE_LIGHTRAG_VECTOR, by_source)

    def test_cross_source_entity_merge(self):
        # Metformin(LR alias) 与 二甲双胍(N4) → 同 identity 合并，sources 保留全部
        res = fuse_evidence(lightrag_data=LR_DATA, neo4j_data=N4_DATA,
                            entity_name_map=NAME_MAP)
        merged = [it for it in res["evidence"]
                  if it["name"] == "二甲双胍" and it["kind"] == KIND_NEO4J_NODE]
        self.assertEqual(len(merged), 1)
        item = merged[0]
        self.assertEqual(sorted(item["sources"]),
                         [SOURCE_LIGHTRAG_GRAPH, SOURCE_NEO4J_GRAPH])
        # payload 保留 LightRAG 实体
        self.assertIn("lightrag_entity", item["payload"])
        # 不应出现第二条独立 lightrag_entity 二甲双胍
        lr_dups = [it for it in res["evidence"]
                   if it["kind"] == KIND_LIGHTRAG_ENTITY and it["name"] == "二甲双胍"]
        self.assertEqual(lr_dups, [])
        self.assertGreaterEqual(res["summary"]["deduplicated"], 1)

    def test_cross_source_relation_merge(self):
        # LR Metformin--treats-->T2DM 与 N4 二甲双胍--TREATS-->2型糖尿病 合并
        res = fuse_evidence(lightrag_data=LR_DATA, neo4j_data=N4_DATA,
                            entity_name_map=NAME_MAP)
        triple = [it for it in res["evidence"]
                  if it["kind"] == KIND_NEO4J_TRIPLE
                  and it["name"].startswith("二甲双胍 -TREATS->")]
        self.assertEqual(len(triple), 1)
        self.assertIn(SOURCE_LIGHTRAG_GRAPH, triple[0]["sources"])
        self.assertIn("lightrag_relation", triple[0]["payload"])

    def test_unresolved_relation_not_merged(self):
        # T2DM--recommends-->Diabetes Diet：target 未 curated → 不跨源合并，保留独立 LR 关系
        res = fuse_evidence(lightrag_data=LR_DATA, neo4j_data=N4_DATA,
                            entity_name_map=NAME_MAP)
        lr_rel = [it for it in res["evidence"]
                  if it["kind"] == KIND_LIGHTRAG_RELATION
                  and "recommends" in it["name"]]
        self.assertEqual(len(lr_rel), 1)
        self.assertEqual(lr_rel[0]["sources"], [SOURCE_LIGHTRAG_GRAPH])

    def test_provenance_preserved(self):
        res = fuse_evidence(lightrag_data=LR_DATA, neo4j_data=N4_DATA,
                            entity_name_map=NAME_MAP)
        for it in res["evidence"]:
            self.assertGreaterEqual(len(it["sources"]), 1)
            for src in it["sources"]:
                self.assertIn(src, (SOURCE_LIGHTRAG_VECTOR, SOURCE_LIGHTRAG_GRAPH,
                                    SOURCE_NEO4J_GRAPH))

    def test_all_empty(self):
        res = fuse_evidence(lightrag_data={}, neo4j_data={},
                            entity_name_map=NAME_MAP)
        self.assertEqual(res["evidence"], [])
        self.assertFalse(res["coverage"]["neo4j_has_evidence"])
        self.assertEqual(res["evidence_sources"]["lightrag_vector"], 0)
        self.assertTrue(any("未产生" in n for n in res["coverage"]["notes"]))

    def test_neo4j_no_match_coverage_note(self):
        # Neo4j 返回空 + note（未命中 curated 实体）
        n4 = {"nodes": [], "edges": [], "triples": [],
              "note": "问题未命中 curated 实体，Neo4j 检索返回空。"}
        res = fuse_evidence(lightrag_data=LR_DATA, neo4j_data=n4,
                            entity_name_map=NAME_MAP)
        self.assertFalse(res["coverage"]["neo4j_has_evidence"])
        self.assertTrue(any("未命中" in n for n in res["coverage"]["notes"]))
        self.assertIsNone(res["prompt_section"])

    def test_coverage_note_preserved(self):
        n4 = {**N4_DATA, "coverage_note": "当前 curated 层无 Symptom 实体；症状→药物查询通常返回空。"}
        res = fuse_evidence(lightrag_data=LR_DATA, neo4j_data=n4,
                            entity_name_map=NAME_MAP)
        self.assertTrue(any("Symptom" in n for n in res["coverage"]["notes"]))

    def test_neo4j_unavailable(self):
        n4 = {"unavailable": True, "error": "RuntimeError: connection refused",
              "nodes": [], "edges": [], "triples": []}
        res = fuse_evidence(lightrag_data=LR_DATA, neo4j_data=n4,
                            entity_name_map=NAME_MAP)
        self.assertFalse(res["coverage"]["neo4j_has_evidence"])
        self.assertTrue(any("不可用" in n for n in res["coverage"]["notes"]))
        self.assertTrue(all(SOURCE_NEO4J_GRAPH not in it["sources"]
                            for it in res["evidence"]))


# ============================================================
# 7. 截断 + 排序
# ============================================================

class TestTruncationAndSort(unittest.TestCase):
    def _big_neo4j(self):
        triples = [[f"实体{i}", "TREATS", "2型糖尿病"] for i in range(40)]
        nodes = [{"id": f"实体{i}", "name": f"实体{i}", "entity_type": "disease",
                  "labels": ["Disease"], "properties": {}} for i in range(40)]
        return {"nodes": nodes, "edges": [], "triples": triples}

    def test_truncation_keeps_high_priority(self):
        cfg = FusionConfig(max_neo4j_triples=5, max_neo4j_nodes=5,
                           max_lightrag_entities=1, max_lightrag_relations=1,
                           max_chunks=1, max_total=12)
        res = fuse_evidence(lightrag_data=LR_DATA,
                            neo4j_data=self._big_neo4j(),
                            entity_name_map=NAME_MAP, config=cfg)
        self.assertLessEqual(len(res["evidence"]), 12)
        # 截断数 = 总量 - 保留量（一致性）
        self.assertEqual(res["summary"]["total"] - res["summary"]["truncated"],
                         len(res["evidence"]))
        self.assertGreater(res["summary"]["truncated"], 0)
        kinds = [it["kind"] for it in res["evidence"]]
        # 高优先级在前（priority 单调不减）
        priorities = [it["priority"] for it in res["evidence"]]
        self.assertEqual(priorities, sorted(priorities))
        self.assertEqual(kinds[0], KIND_NEO4J_TRIPLE)
        self.assertIn(KIND_NEO4J_NODE, kinds)
        self.assertEqual(res["evidence_sources"]["neo4j_graph"]["triples"], 40)

    def test_sort_stable(self):
        r1 = fuse_evidence(lightrag_data=LR_DATA, neo4j_data=N4_DATA,
                           entity_name_map=NAME_MAP)
        r2 = fuse_evidence(lightrag_data=LR_DATA, neo4j_data=N4_DATA,
                           entity_name_map=NAME_MAP)
        self.assertEqual(r1["evidence"], r2["evidence"])

    def test_sort_priority_order(self):
        items = [
            EvidenceItem(id="chunk::x", kind=KIND_LIGHTRAG_CHUNK, name="x",
                         payload={}, sources=[SOURCE_LIGHTRAG_VECTOR], priority=4),
            EvidenceItem(id="t", kind=KIND_NEO4J_TRIPLE, name="t",
                         payload={}, sources=[SOURCE_NEO4J_GRAPH], priority=0),
            EvidenceItem(id="e", kind=KIND_LIGHTRAG_ENTITY, name="e",
                         payload={}, sources=[SOURCE_LIGHTRAG_GRAPH], priority=3),
            EvidenceItem(id="n", kind=KIND_NEO4J_NODE, name="n",
                         payload={}, sources=[SOURCE_NEO4J_GRAPH], priority=1),
            EvidenceItem(id="r", kind=KIND_LIGHTRAG_RELATION, name="r",
                         payload={}, sources=[SOURCE_LIGHTRAG_GRAPH], priority=2),
        ]
        sorted_items = sort_evidence(items)
        self.assertEqual([it.priority for it in sorted_items], [0, 1, 2, 3, 4])


# ============================================================
# 8. build_fusion_prompt
# ============================================================

class TestBuildPrompt(unittest.TestCase):
    def test_with_base_prompt(self):
        p = build_fusion_prompt(N4_DATA, base_prompt="你是医疗助手。")
        self.assertIn("你是医疗助手。", p)
        self.assertIn("TREATS", p)
        self.assertIn("检索到的", p)

    def test_without_base_prompt(self):
        p = build_fusion_prompt(N4_DATA, base_prompt=None)
        self.assertIsNotNone(p)
        self.assertIn("TREATS", p)

    def test_no_evidence_returns_base(self):
        self.assertEqual(build_fusion_prompt({}, base_prompt="B"), "B")
        self.assertIsNone(build_fusion_prompt({}, base_prompt=None))

    def test_unavailable_returns_base(self):
        n4 = {"unavailable": True, "error": "x"}
        self.assertEqual(build_fusion_prompt(n4, base_prompt="B"), "B")

    def test_capped_by_max_prompt_chars(self):
        cfg = FusionConfig(max_prompt_chars=80)
        big = {"nodes": [], "edges": [], "triples": [[f"实体{i}", "TREATS", "2型糖尿病"]
                                                     for i in range(50)]}
        p = build_fusion_prompt(big, base_prompt=None, config=cfg)
        self.assertLessEqual(len(p), 80 + 30)  # 截断标记后缀允许少量溢出


# ============================================================
# 9. malformed / missing fields
# ============================================================

class TestMalformed(unittest.TestCase):
    def test_entity_missing_name(self):
        lr = {"entities": [{"entity_type": "concept"}],
              "relationships": [], "chunks": []}
        res = fuse_evidence(lightrag_data=lr, neo4j_data={},
                            entity_name_map=NAME_MAP)
        self.assertEqual([it for it in res["evidence"] if it["kind"] == KIND_LIGHTRAG_ENTITY], [])

    def test_chunks_none(self):
        lr = {"entities": [], "relationships": [], "chunks": None}
        res = fuse_evidence(lightrag_data=lr, neo4j_data=N4_DATA,
                            entity_name_map=NAME_MAP)
        self.assertEqual(res["evidence_sources"]["lightrag_vector"], 0)

    def test_edges_missing(self):
        # edges 缺失 → triples 为空；但有节点 → 仍如实保留节点证据，不伪造关系
        n4 = {"nodes": [{"id": "x", "name": "高血压", "labels": ["Disease"], "properties": {}}],
              "triples": [], "edges": None}
        res = fuse_evidence(lightrag_data={}, neo4j_data=n4,
                            entity_name_map=NAME_MAP)
        self.assertEqual(res["evidence_sources"]["neo4j_graph"]["triples"], 0)
        self.assertTrue(res["coverage"]["neo4j_has_evidence"])  # 有节点即证据
        kinds = [it["kind"] for it in res["evidence"]]
        self.assertIn(KIND_NEO4J_NODE, kinds)
        self.assertNotIn(KIND_NEO4J_TRIPLE, kinds)  # 不伪造关系

    def test_malformed_triples_skipped(self):
        n4 = {**N4_DATA, "triples": [["a", "b"], "bad", [1, 2, 3], ["二甲双胍", "TREATS", "2型糖尿病"]]}
        res = fuse_evidence(lightrag_data={}, neo4j_data=n4,
                            entity_name_map=NAME_MAP)
        triple_names = [it["name"] for it in res["evidence"]
                        if it["kind"] == KIND_NEO4J_TRIPLE]
        self.assertIn("二甲双胍 -TREATS-> 2型糖尿病", triple_names)
        self.assertEqual(len(triple_names), 1)

    def test_prompt_section_none_when_no_neo4j(self):
        res = fuse_evidence(lightrag_data=LR_DATA, neo4j_data={},
                            entity_name_map=NAME_MAP)
        self.assertIsNone(res["prompt_section"])


# ============================================================
# 10. 静态：evidence_fusion 不 import LLM / embedding
# ============================================================

class TestNoLLMImport(unittest.TestCase):
    def test_no_llm_or_embedding_import(self):
        src = Path(_PROJECT_ROOT_ABS / "src" / "evidence_fusion.py").read_text(encoding="utf-8")
        forbidden = ("openai", "lightrag", "sentence_transformers", "llm_client",
                     "embedder", "import torch", "from torch")
        import_lines = [ln for ln in src.splitlines()
                        if ln.lstrip().startswith(("import ", "from "))]
        for line in import_lines:
            for kw in forbidden:
                self.assertNotIn(kw, line,
                                 f"evidence_fusion.py 禁止 import 含 '{kw}': {line}")


# ============================================================
# 11. orchestration mock：query_with_context 最多一次
# ============================================================

class FakeEngine:
    """mock RAGEngine：记录 query_with_context 调用次数。"""
    def __init__(self):
        self.calls = 0
        self.last_system_prompt = None

    async def query_with_context(self, question, mode="mix", system_prompt=None,
                                 top_k=60, chunk_top_k=20):
        self.calls += 1
        self.last_system_prompt = system_prompt
        return {
            "status": "success",
            "data": {"entities": [], "relationships": [], "chunks": []},
            "metadata": {},
            "llm_response": "这是一个测试性回答，不构成医疗建议。",
        }


class FakeRetriever:
    """mock Neo4jGraphRetriever：返回空（无 Neo4j 证据）。"""
    def get_entity(self, name):
        return {"query": "", "count": 0, "nodes": [], "edges": [], "triples": [],
                "resolution": {}}

    def get_neighbors(self, name, relation_type=None, direction="both", limit=20):
        return {"query": "", "count": 0, "nodes": [], "edges": [], "triples": [],
                "resolution": {}}


class TestOrchestrationMock(unittest.TestCase):
    def test_query_with_context_called_once(self):
        eng = FakeEngine()
        result = asyncio.run(run_fusion_query(
            "二甲双胍治什么病？",
            engine=eng,
            retriever=FakeRetriever(),
            name_map=NAME_MAP,
        ))
        self.assertEqual(eng.calls, 1)
        self.assertEqual(result["llm_generation_count"], 1)
        # 关键字段齐全
        for key in ("question", "fused_context", "system_prompt", "llm_response",
                    "safety", "refusal", "citation", "uncertainty", "metrics",
                    "reliability_decision"):
            self.assertIn(key, result)
        # 决策可序列化且结构完整
        d = result["reliability_decision"]
        self.assertIn(d["decision"], ("allow", "revise", "refuse"))
        self.assertIn("is_final_refusal", d)
        # Neo4j mock 返回空 → 无 neo4j 证据、非不可用
        self.assertFalse(result["neo4j_unavailable"])

    def test_blocked_no_llm_call(self):
        eng = FakeEngine()
        result = asyncio.run(run_fusion_query(
            "我不想活了怎么办", engine=eng, retriever=FakeRetriever(),
            name_map=NAME_MAP,
        ))
        self.assertEqual(eng.calls, 0)
        self.assertEqual(result["llm_generation_count"], 0)
        self.assertTrue(result["reliability_decision"]["is_final_refusal"])


# ============================================================
# 12. reliability_decision.py 未修改
# ============================================================

class TestReliabilityDecisionUnchanged(unittest.TestCase):
    def test_api_intact_and_no_f5_dependency(self):
        src = Path(_PROJECT_ROOT_ABS / "evaluation" / "reliability_decision.py").read_text(
            encoding="utf-8")
        # 无 F5 依赖（import 段不引用 F5 / Neo4j 模块）
        import_lines = [ln for ln in src.splitlines()
                        if ln.lstrip().startswith(("import ", "from "))]
        for line in import_lines:
            for kw in ("evidence_fusion", "run_graph_query", "neo4j", "src."):
                self.assertNotIn(kw, line)
        # 核心 API 存在
        from evaluation.reliability_decision import (
            make_reliability_decision, Decision, ReliabilityDecision,
        )
        self.assertTrue(callable(make_reliability_decision))
        self.assertEqual(Decision.REFUSE.value, "refuse")
        self.assertTrue(hasattr(ReliabilityDecision, "to_dict"))

    def test_make_reliability_decision_smoke(self):
        from evaluation.reliability_decision import make_reliability_decision
        d = make_reliability_decision(question="高血压是什么",
                                      safety_check=None,
                                      uncertainty_analysis=None,
                                      citation_analysis=None,
                                      m8=None, m9=None, m10=None, m12=None,
                                      refusal_result={"is_refusal": False},
                                      answer="高血压是一种...")
        self.assertEqual(d.decision, "allow")
        self.assertFalse(d.is_final_refusal)


if __name__ == "__main__":
    unittest.main()
