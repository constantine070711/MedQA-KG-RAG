# -*- coding: utf-8 -*-
"""Phase 10-A.5 — 真实医学知识库（CMeIE）独立 adapter 单元测试。

覆盖：正常转换 / malformed / 缺失字段 / 空数据 / 重复数据 / provenance /
      entity linking（L0-L3 + L4）/ unresolved relation / 输入文件不被修改 /
      输出 schema 正确 / 数据隔离 / 不 import 生产核心。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_adapter.medical_kb_adapter import (  # noqa: E402
    MedKBEntity,
    MedKBRelation,
    MEDKB_RELATION_MAP,
    convert,
    ingest,
    link_entities,
    link_entities_l4,
    parse_cmeie_record,
    validate_entity,
    validate_relation,
)


def _rec(subject="高血压", subject_type="疾病", pred="并发症", obj="脑卒中",
         obj_type="疾病", combined=False, text=None):
    return {
        "text": text or f"{subject}@高血压是常见病，{subject}可能引起{obj}。",
        "spo_list": [{
            "Combined": combined,
            "predicate": pred,
            "subject": subject,
            "subject_type": subject_type,
            "object": {"@value": obj},
            "object_type": {"@value": obj_type},
        }]
    }


class TestParseNormal(unittest.TestCase):
    def test_normal_record(self):
        res = parse_cmeie_record(_rec(), record_idx=0)
        self.assertEqual(len(res["relations"]), 1)
        self.assertEqual(len(res["entities"]), 2)
        rel = res["relations"][0]
        self.assertEqual(rel.subject, "高血压")
        self.assertEqual(rel.object, "脑卒中")
        self.assertEqual(rel.relation_type, "HAS_COMPLICATION")
        self.assertEqual(rel.relation, "HAS_COMPLICATION")
        self.assertEqual(rel.confidence, 1.0)
        self.assertEqual(rel.subject_type, "疾病")
        self.assertEqual(rel.object_type, "疾病")

    def test_entities_typed(self):
        res = parse_cmeie_record(_rec(), record_idx=0)
        types = {e.name: e.entity_type for e in res["entities"]}
        self.assertEqual(types["高血压"], "disease")
        self.assertEqual(types["脑卒中"], "disease")


class TestParseObjectDict(unittest.TestCase):
    def test_object_dict_unwrapped(self):
        """object 为 {"@value": ...} 时必须解包为字符串。"""
        res = parse_cmeie_record(_rec(), record_idx=0)
        rel = res["relations"][0]
        self.assertEqual(rel.object, "脑卒中")
        self.assertNotIn("{", rel.object)
        self.assertNotIn("@value", rel.object)

    def test_object_string_ok(self):
        """兼容 object 已是字符串的情况。"""
        rec = _rec()
        rec["spo_list"][0]["object"] = "脑卒中"
        rec["spo_list"][0]["object_type"] = "疾病"
        res = parse_cmeie_record(rec, record_idx=0)
        self.assertEqual(res["relations"][0].object, "脑卒中")


class TestParseMalformed(unittest.TestCase):
    def test_record_not_dict(self):
        res = parse_cmeie_record([1, 2], record_idx=0)
        self.assertEqual(res["relations"], [])
        self.assertEqual(res["entities"], [])
        self.assertEqual(len(res["skipped"]), 1)

    def test_spo_not_list(self):
        rec = {"text": "x", "spo_list": "notalist"}
        res = parse_cmeie_record(rec, record_idx=0)
        self.assertEqual(res["relations"], [])

    def test_spo_item_not_dict(self):
        rec = _rec()
        rec["spo_list"] = ["bad"]
        res = parse_cmeie_record(rec, record_idx=0)
        self.assertEqual(len(res["skipped"]), 1)
        self.assertEqual(res["relations"], [])

    def test_spo_missing_fields(self):
        rec = _rec()
        rec["spo_list"] = [{"subject": "高血压", "predicate": "并发症"}]  # 缺 object
        res = parse_cmeie_record(rec, record_idx=0)
        self.assertEqual(res["relations"], [])
        self.assertEqual(len(res["skipped"]), 1)


class TestParseEmpty(unittest.TestCase):
    def test_empty_text(self):
        rec = {"text": "", "spo_list": []}
        res = parse_cmeie_record(rec, record_idx=0)
        self.assertEqual(res["entities"], [])
        self.assertEqual(res["relations"], [])
        self.assertEqual(len(res["skipped"]), 1)


class TestRelationMapping(unittest.TestCase):
    def test_mapped(self):
        self.assertEqual(MEDKB_RELATION_MAP.get("并发症"), "HAS_COMPLICATION")
        self.assertEqual(MEDKB_RELATION_MAP.get("筛查"), "SCREENS_FOR")
        self.assertEqual(MEDKB_RELATION_MAP.get("鉴别诊断"), "DIFFERENTIAL_DIAGNOSIS")
        self.assertEqual(MEDKB_RELATION_MAP.get("高危因素"), "HAS_RISK_FACTOR")

    def test_unmapped_honest(self):
        """未映射 predicate 必须如实 UNMAPPED，不强行映射。"""
        rec = _rec(pred="临床表现", obj="头晕", obj_type="症状")
        rel = parse_cmeie_record(rec, record_idx=0)["relations"][0]
        self.assertEqual(rel.relation_type, "UNMAPPED")
        self.assertEqual(rel.relation, "临床表现")   # 保留原词
        self.assertEqual(rel.confidence, 0.5)
        self.assertEqual(rel.quality["level"], "warn")

    def test_type_mapping_honest(self):
        """无法映射的类型如实 unknown + warning，不强行归类。"""
        rec = _rec(subject="某人群", subject_type="社会学", pred="发病率",
                   obj="10%", obj_type="流行病学")
        res = parse_cmeie_record(rec, record_idx=0)
        for e in res["entities"]:
            self.assertEqual(e.entity_type, "unknown")
            self.assertTrue(any("unknown" in w for w in e.quality["warnings"]))


class TestIngestDedup(unittest.TestCase):
    def _write(self, lines, suffix=".jsonl"):
        tmp = tempfile.mkdtemp()
        p = Path(tmp) / f"CMeIE_train{suffix}"
        p.write_text("\n".join(lines), encoding="utf-8")
        return tmp

    def test_duplicate_triples_dedup(self):
        lines = [
            json.dumps(_rec(), ensure_ascii=False),
            json.dumps(_rec(), ensure_ascii=False),  # 完全重复
        ]
        res = ingest(self._write(lines), split="train")
        self.assertEqual(res["stats"]["records_loaded"], 2)
        self.assertEqual(res["stats"]["relations_raw"], 2)
        self.assertEqual(res["stats"]["relations_dedup"], 1)

    def test_limit(self):
        lines = [json.dumps(_rec(subject=f"D{i}"), ensure_ascii=False) for i in range(5)]
        res = ingest(self._write(lines), split="train", limit=2)
        self.assertEqual(res["stats"]["records_loaded"], 2)

    def test_empty_input(self):
        res = ingest(self._write([""]), split="train")
        self.assertEqual(res["entities"], [])
        self.assertEqual(res["relations"], [])


class TestProvenance(unittest.TestCase):
    def test_every_item_has_provenance(self):
        res = parse_cmeie_record(_rec(), record_idx=42)
        for e in res["entities"]:
            self.assertIn("provenance", e.to_dict())
            prov = e.provenance
            self.assertEqual(prov["dataset"], "CMeIE")
            self.assertEqual(prov["license"], "CC-BY-NC 4.0")
            self.assertEqual(prov["sample_id"], "42")
        for r in res["relations"]:
            self.assertEqual(r.provenance["dataset"], "CMeIE")
            self.assertEqual(r.provenance["sample_id"], "42")
            self.assertEqual(r.provenance["license"], "CC-BY-NC 4.0")


class TestConvertSchema(unittest.TestCase):
    def test_convert_fields(self):
        # 直接构造 ingest 结果
        e = MedKBEntity(entity_id="medkb:x:0:0", name="高血压", entity_type="disease",
                        raw_text="t", provenance={"dataset": "CMeIE"})
        r = MedKBRelation(subject="高血压", relation="HAS_COMPLICATION", object="脑卒中",
                          relation_type="HAS_COMPLICATION",
                          provenance={"dataset": "CMeIE"})
        conv = convert({"entities": [e], "relations": [r]})
        ent = conv["entities"][0]
        rel = conv["relations"][0]
        for key in ("entity_id", "name", "entity_type", "provenance", "quality", "raw_text"):
            self.assertIn(key, ent)
        for key in ("subject", "relation", "object", "relation_type", "provenance", "quality"):
            self.assertIn(key, rel)
        # convert 回写校验后的 quality
        self.assertEqual(ent["quality"]["level"], "pass")
        self.assertEqual(rel["quality"]["level"], "pass")

    def test_validate_detects_missing(self):
        e = MedKBEntity(entity_id="", name="", entity_type="", provenance={})
        v = validate_entity(e)
        self.assertEqual(v["level"], "fail")
        r = MedKBRelation(subject="", relation="", object="", relation_type="",
                          provenance={})
        v2 = validate_relation(r)
        self.assertEqual(v2["level"], "fail")


class TestLinking(unittest.TestCase):
    def test_link_entities_l4(self):
        """L4 相似度候选：弱信号标记 L4，不误标 matched。"""
        prod = {"二甲双胍": "二甲双胍", "高血压": "高血压"}
        unmatched = [{"entity_id": "medkb:x", "name": "高血压病",
                      "entity_type": "disease", "matched": False,
                      "canonical": None, "match_level": None, "confidence": 0.0}]
        out = link_entities_l4(unmatched, prod, threshold=0.8)
        hit = out[0]
        self.assertEqual(hit["match_level"], "L4")
        self.assertFalse(hit["matched"])   # 弱信号不得标 matched
        self.assertGreaterEqual(hit["confidence"], 0.8)
        self.assertEqual(hit["canonical"], "高血压")

    def test_link_entities_pipeline(self):
        """L0-L3 复用 cblue：生产名精确匹配 → linked。"""
        res = parse_cmeie_record(_rec(), record_idx=0)
        linking = link_entities(res["entities"], {"高血压": "高血压"}, {"E10": "高血压"})
        by_name = {l["name"]: l for l in linking["linked"]}
        # 生产名表中无 脑卒中 → 应 unmatched 或 candidate，不误合
        self.assertIn("高血压", by_name)


class TestInputNotModified(unittest.TestCase):
    def test_ingest_reads_only(self):
        tmp = tempfile.mkdtemp()
        p = Path(tmp) / "CMeIE_train.jsonl"
        content = json.dumps(_rec(), ensure_ascii=False)
        p.write_text(content, encoding="utf-8")
        ingest(str(tmp), split="train")
        self.assertEqual(p.read_text(encoding="utf-8"), content)  # 输入未被修改


class TestIsolation(unittest.TestCase):
    def test_no_production_core_import(self):
        """adapter 不得 import 生产核心（agent/rag_engine/evidence_fusion/neo4j）。"""
        import src.data_adapter.medical_kb_adapter as m
        src = Path(m.__file__).read_text(encoding="utf-8")
        for forbidden in ("src.agent", "src.rag_engine", "src.evidence_fusion",
                          "src.neo4j", "src.medical_core"):
            self.assertNotIn(forbidden, src,
                             f"adapter 不得 import {forbidden}")


if __name__ == "__main__":
    unittest.main()
