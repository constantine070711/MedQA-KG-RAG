# -*- coding: utf-8 -*-
"""Phase 10-A.5 Step B1 — CMeIE → custom_kg 构建器单元测试。

覆盖：chunks/entities/relationships schema / 多 predicate 防覆盖合并 /
      entity 去重 / provenance 可追溯 / 零 LLM / 幂等确定性 / 输入不被修改 /
      类型映射 / UNMAPPED predicate 保留。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_adapter.medkb_custom_kg import (  # noqa: E402
    build_custom_kg,
    build_custom_kg_from_file,
)


def _rec(sample_id, text, spo_list):
    return {"sample_id": sample_id, "text": text, "spo_list": spo_list}


def _spo(subj, pred, obj, subj_type="疾病", obj_type="疾病"):
    return {
        "subject": subj, "subject_type": subj_type, "predicate": pred,
        "object": {"@value": obj}, "object_type": {"@value": obj_type},
    }


class TestBuildSchema(unittest.TestCase):
    def test_chunks_per_record(self):
        recs = [_rec(0, "高血压@高血压是常见病。", [_spo("高血压", "并发症", "脑卒中")]),
                _rec(1, "糖尿病@糖尿病是慢性病。", [_spo("糖尿病", "并发症", "肾病")])]
        r = build_custom_kg(recs)
        self.assertEqual(len(r["chunks"]), 2)
        self.assertEqual(r["chunks"][0]["content"], "高血压@高血压是常见病。")
        self.assertEqual(r["chunks"][0]["source_id"], "medkb_cmeie_00000")
        self.assertIn("file_path", r["chunks"][0])
        self.assertIn("chunk_order_index", r["chunks"][0])

    def test_entities_from_subject_object(self):
        recs = [_rec(0, "t", [_spo("高血压", "并发症", "脑卒中")])]
        r = build_custom_kg(recs)
        names = {e["entity_name"]: e["entity_type"] for e in r["entities"]}
        self.assertEqual(names["高血压"], "disease")
        self.assertEqual(names["脑卒中"], "disease")

    def test_relationship_schema(self):
        recs = [_rec(0, "t", [_spo("高血压", "并发症", "脑卒中")])]
        r = build_custom_kg(recs)
        rel = r["relationships"][0]
        self.assertEqual(rel["src_id"], "高血压")
        self.assertEqual(rel["tgt_id"], "脑卒中")
        self.assertIn("description", rel)
        self.assertIn("keywords", rel)
        self.assertIn("weight", rel)
        self.assertIn("source_id", rel)
        self.assertEqual(rel["keywords"], "HAS_COMPLICATION")


class TestMultiPredicate(unittest.TestCase):
    def test_same_pair_multi_predicate_merged(self):
        """同一实体对多 predicate → 单边合并，不覆盖、不丢失。"""
        recs = [_rec(0, "t", [
                    _spo("消化性溃疡病", "风险评估因素", "重症监护室患者"),
                    _spo("消化性溃疡病", "高危因素", "重症监护室患者")])]
        r = build_custom_kg(recs)
        self.assertEqual(len(r["relationships"]), 1)   # 合并为单边
        rel = r["relationships"][0]
        self.assertIn("风险评估因素", rel["description"])
        self.assertIn("高危因素", rel["description"])
        self.assertIn("HAS_RISK_FACTOR", rel["keywords"])   # 高危因素映射
        self.assertIn("风险评估因素", rel["keywords"])       # UNMAPPED 保留原词
        self.assertEqual(r["stats"]["multi_predicate_pairs"], 1)
        self.assertEqual(r["stats"]["max_predicates_per_pair"], 2)

    def test_distinct_pairs_not_merged(self):
        """不同实体对不得合并。"""
        recs = [_rec(0, "t", [_spo("高血压", "并发症", "脑卒中"),
                              _spo("高血压", "并发症", "冠心病")])]
        r = build_custom_kg(recs)
        self.assertEqual(len(r["relationships"]), 2)


class TestEntityDedup(unittest.TestCase):
    def test_same_entity_deduped(self):
        recs = [_rec(0, "t", [_spo("高血压", "并发症", "脑卒中")]),
                _rec(1, "t", [_spo("高血压", "并发症", "冠心病")])]
        r = build_custom_kg(recs)
        names = [e["entity_name"] for e in r["entities"]]
        self.assertEqual(names.count("高血压"), 1)   # 跨记录去重
        self.assertIn("脑卒中", names)
        self.assertIn("冠心病", names)


class TestUnmappedPredicate(unittest.TestCase):
    def test_unmapped_preserved(self):
        """UNMAPPED predicate：keywords 保留原词，不伪造。"""
        recs = [_rec(0, "t", [_spo("产后抑郁症", "临床表现", "情绪低落")])]
        r = build_custom_kg(recs)
        rel = r["relationships"][0]
        self.assertEqual(rel["description"], "临床表现")
        self.assertEqual(rel["keywords"], "临床表现")


class TestProvenance(unittest.TestCase):
    def test_manifest_traceable(self):
        recs = [_rec(7, "高血压@原文。", [_spo("高血压", "并发症", "脑卒中")])]
        r = build_custom_kg(recs)
        m = r["provenance_manifest"]
        self.assertEqual(m["dataset"], "CMeIE")
        self.assertEqual(m["license"], "CC-BY-NC 4.0")
        # chunk → sample_id
        self.assertEqual(m["chunks"]["medkb_cmeie_00007"]["sample_id"], 7)
        # entity → source chunk → sample
        ent = m["entities"]["高血压"]
        self.assertEqual(ent["source_sample_id"], 7)
        # relationship → source sample
        self.assertEqual(m["relationships"][0]["source_sample_id"], 7)
        self.assertEqual(m["relationships"][0]["predicates"], "并发症")


class TestDeterministicAndInput(unittest.TestCase):
    def test_deterministic(self):
        recs = [_rec(0, "t", [_spo("高血压", "并发症", "脑卒中")])]
        a = build_custom_kg(recs)
        b = build_custom_kg(recs)
        self.assertEqual(a["relationships"], b["relationships"])
        self.assertEqual(a["entities"], b["entities"])

    def test_input_not_modified(self):
        recs = [_rec(0, "高血压@原文。", [_spo("高血压", "并发症", "脑卒中")])]
        before = json.dumps(recs, ensure_ascii=False)
        build_custom_kg(recs)
        self.assertEqual(json.dumps(recs, ensure_ascii=False), before)

    def test_from_file_limit(self):
        tmp = tempfile.mkdtemp()
        p = Path(tmp) / "CMeIE_train.jsonl"
        lines = [json.dumps({"text": f"t{i}", "spo_list": [
                    {"subject": f"D{i}", "subject_type": "疾病", "predicate": "并发症",
                     "object": {"@value": "脑卒中"}, "object_type": {"@value": "疾病"}}]},
                    ensure_ascii=False)
                 for i in range(3)]
        p.write_text("\n".join(lines), encoding="utf-8")
        r = build_custom_kg_from_file(p, limit=2)
        self.assertEqual(r["stats"]["records"], 2)
        self.assertEqual(len(r["chunks"]), 2)


class TestZeroLLM(unittest.TestCase):
    def test_no_forbidden_import(self):
        """builder 不 import LLM / LightRAG / 生产核心。"""
        src = Path(Path(__file__).resolve().parent.parent
                   / "src" / "data_adapter" / "medkb_custom_kg.py").read_text(encoding="utf-8")
        for forbidden in ("lightrag", "src.llm_client", "src.agent",
                          "src.rag_engine", "src.evidence_fusion", "src.neo4j",
                          "src.medical_core"):
            self.assertNotIn(forbidden, src,
                             f"builder 不得 import {forbidden}")


if __name__ == "__main__":
    unittest.main()
