# -*- coding: utf-8 -*-
"""Phase 10-A — CBLUE 适配器单元测试（stdlib unittest，零 LLM / DB / 网络）。

覆盖要求：
    1. 正常 CBLUE 样例转换（CMeEE-V2 / CMeIE-V2）
    2. 缺失字段
    3. malformed 数据
    4. provenance 是否保留
    5. entity/relation schema 正确
    6. 不修改输入数据
    7. 空数据处理

运行:
    python -m unittest tests.test_cblue_adapter -v
"""

import json
import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_adapter import cblue_adapter as ca  # noqa: E402


# ============================================================
# Fixtures（CBLUE 官方 schema 合成样本）
# ============================================================

CMEEE_RECORD = {
    "sample_id": "ee_0001",
    "text": "患者3天前出现发热、咳嗽，伴头痛。",
    "entities": [
        {"start_idx": 7, "end_idx": 9, "type": "sym", "entity": "发热"},
        {"start_idx": 10, "end_idx": 12, "type": "sym", "entity": "咳嗽"},
        {"start_idx": 15, "end_idx": 17, "type": "sym", "entity": "头痛"},
    ],
}

CMEIE_RECORD = {
    "sample_id": "ie_0001",
    "text": "阿司匹林用于治疗冠心病，但其可能引起胃肠道不良反应。",
    "spo_list": [
        {"subject": "阿司匹林", "predicate": "治疗", "object": "冠心病"},
        {"subject": "阿司匹林", "predicate": "不良反应", "object": "胃肠道反应"},
        {"subject": "阿司匹林", "predicate": "未知关系", "object": "X"},
    ],
}


class TestCMeEEParse(unittest.TestCase):
    """1. 正常 CMeEE-V2 转换 + 5. schema 正确"""

    def test_normal_record(self):
        res = ca.parse_cmeee_record(CMEEE_RECORD, split="train", source_file="s.json")
        self.assertEqual(len(res["entities"]), 3)
        self.assertEqual(len(res["skipped"]), 0)
        e = res["entities"][0]
        self.assertEqual(e.name, "发热")
        self.assertEqual(e.entity_type, "symptom")
        self.assertEqual(e.entity_id, "cblue:cmeee_v2:train:ee_0001:0")

    def test_entity_schema_fields(self):
        res = ca.parse_cmeee_record(CMEEE_RECORD, source_file="s.json")
        d = res["entities"][0].to_dict()
        for k in ("entity_id", "name", "entity_type", "aliases", "standard_codes",
                  "raw_text", "provenance", "quality"):
            self.assertIn(k, d)
        # raw_text 保留原文（含"发热"等原文子串）
        self.assertIn("发热", d["raw_text"])
        self.assertEqual(d["quality"]["level"], "pass")

    def test_type_mapping_all_9(self):
        # 9 类 CBLUE 类型都有映射（bod/mic 保留为字符串 + warning）
        for cb, exp in ca.CMEEE_TYPE_MAP.items():
            rec = {"sample_id": f"t_{cb}", "text": "样本文本内容足够长。",
                   "entities": [{"start_idx": 0, "end_idx": 2, "type": cb, "entity": "X"}]}
            res = ca.parse_cmeee_record(rec, source_file="s.json")
            self.assertEqual(res["entities"][0].entity_type, exp, cb)

    def test_aliases_and_codes_default_empty(self):
        res = ca.parse_cmeee_record(CMEEE_RECORD, source_file="s.json")
        e = res["entities"][0]
        self.assertEqual(e.aliases, [])
        self.assertEqual(e.standard_codes, [])


class TestCMeIEParse(unittest.TestCase):
    """1. 正常 CMeIE-V2 转换 + 5. schema 正确"""

    def test_normal_record(self):
        res = ca.parse_cmeie_record(CMEIE_RECORD, split="train", source_file="s.json")
        # 3 条 spo；阿司匹林/冠心病/胃肠道反应/X 四个实体候选
        self.assertEqual(len(res["relations"]), 3)
        self.assertEqual(len(res["entities"]), 4)
        self.assertEqual(len(res["skipped"]), 0)

    def test_relation_mapping(self):
        res = ca.parse_cmeie_record(CMEIE_RECORD, source_file="s.json")
        rels = {r.relation: r for r in res["relations"]}
        self.assertEqual(rels["TREATS"].relation_type, "TREATS")
        self.assertEqual(rels["HAS_ADVERSE_REACTION"].relation_type, "HAS_ADVERSE_REACTION")
        # 未映射 predicate → UNMAPPED，relation 保留原词
        self.assertEqual(rels["未知关系"].relation_type, "UNMAPPED")
        self.assertEqual(rels["未知关系"].relation, "未知关系")
        self.assertLess(rels["未知关系"].confidence, 1.0)

    def test_relation_schema_fields(self):
        res = ca.parse_cmeie_record(CMEIE_RECORD, source_file="s.json")
        d = res["relations"][0].to_dict()
        for k in ("subject", "relation", "object", "relation_type",
                  "confidence", "provenance", "quality"):
            self.assertIn(k, d)

    def test_cmeie_entity_unknown_type_warning(self):
        res = ca.parse_cmeie_record(CMEIE_RECORD, source_file="s.json")
        e = res["entities"][0]
        self.assertEqual(e.entity_type, "unknown")
        self.assertEqual(e.quality["level"], "warn")


class TestMissingFields(unittest.TestCase):
    """2. 缺失字段"""

    def test_missing_sample_id(self):
        rec = {"text": "文本", "entities": []}
        res = ca.parse_cmeee_record(rec, source_file="s.json")
        self.assertEqual(res["entities"], [])
        self.assertEqual(len(res["skipped"]), 1)

    def test_missing_entities_list(self):
        rec = {"sample_id": "x", "text": "文本内容。", "entities": None}
        res = ca.parse_cmeee_record(rec, source_file="s.json")
        self.assertEqual(res["entities"], [])

    def test_missing_spo_list(self):
        rec = {"sample_id": "x", "text": "文本内容。", "spo_list": None}
        res = ca.parse_cmeie_record(rec, source_file="s.json")
        self.assertEqual(res["relations"], [])

    def test_spo_missing_fields(self):
        rec = {"sample_id": "x", "text": "文本。",
               "spo_list": [{"subject": "", "predicate": "治疗", "object": ""}]}
        res = ca.parse_cmeie_record(rec, source_file="s.json")
        self.assertEqual(res["relations"], [])
        self.assertEqual(len(res["skipped"]), 1)


class TestMalformed(unittest.TestCase):
    """3. malformed 数据"""

    def test_record_not_dict(self):
        res = ca.parse_cmeee_record("not a dict", source_file="s.json")
        self.assertEqual(res["entities"], [])
        self.assertEqual(len(res["skipped"]), 1)
        res2 = ca.parse_cmeie_record([1, 2], source_file="s.json")
        self.assertEqual(res2["relations"], [])

    def test_entity_annotation_not_dict(self):
        rec = {"sample_id": "x", "text": "文本内容。", "entities": ["bad", None, 3]}
        res = ca.parse_cmeee_record(rec, source_file="s.json")
        self.assertEqual(len(res["skipped"]), 3)

    def test_empty_entity_name_falls_back_to_span(self):
        rec = {"sample_id": "x", "text": "患者出现发热症状。",
               "entities": [{"start_idx": 4, "end_idx": 6, "type": "sym", "entity": ""}]}
        res = ca.parse_cmeee_record(rec, source_file="s.json")
        self.assertEqual(len(res["entities"]), 1)
        self.assertEqual(res["entities"][0].name, "发热")

    def test_entity_name_span_error(self):
        rec = {"sample_id": "x", "text": "文本。",
               "entities": [{"type": "sym", "entity": ""}]}  # 无 start/end
        res = ca.parse_cmeee_record(rec, source_file="s.json")
        self.assertEqual(res["entities"], [])
        self.assertEqual(len(res["skipped"]), 1)

    def test_jsonl_parse_error_record(self):
        # _read_records 对坏行产出 _parse_error 标记
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                         encoding="utf-8") as f:
            f.write('{"sample_id": "ok", "text": "正常。", "entities": []}\n')
            f.write("not-json-line\n")
            p = Path(f.name)
        try:
            recs = ca._read_records(p)
            self.assertEqual(len(recs), 2)
            self.assertTrue(recs[1].get("_parse_error"))
        finally:
            p.unlink(missing_ok=True)


class TestProvenance(unittest.TestCase):
    """4. provenance 保留"""

    def test_provenance_fields(self):
        res = ca.parse_cmeie_record(CMEIE_RECORD, split="train", source_file="s.json")
        p = res["relations"][0].provenance
        self.assertEqual(p["dataset"], "CMeIE-V2")
        self.assertEqual(p["split"], "train")
        self.assertEqual(p["sample_id"], "ie_0001")
        self.assertEqual(p["source_file"], "s.json")
        self.assertEqual(p["license"], "CC BY-NC-SA 4.0")
        self.assertEqual(p["adapter"], "cblue")

    def test_provenance_index_in_prepare(self):
        # prepare 的 provenance.json 索引应覆盖所有实体/关系
        from src.data_adapter import cblue_adapter as ca
        res = ca.ingest(_tmp_dir_with([("cmeie_v2.json", [CMEIE_RECORD])]),
                        "cmeie_v2")
        conv = ca.convert(res)
        prov = {}
        for e in conv["entities"]:
            prov[e["entity_id"]] = e["provenance"]
        for r in conv["relations"]:
            prov[f"rel:{r['subject']}:{r['relation_type']}:{r['object']}"] = r["provenance"]
        self.assertEqual(len(prov), len(conv["entities"]) + len(conv["relations"]))
        self.assertTrue(all(p.get("license") == "CC BY-NC-SA 4.0" for p in prov.values()))

    def test_missing_provenance_count(self):
        res = ca.parse_cmeee_record(CMEEE_RECORD, source_file="s.json")
        e = res["entities"][0].to_dict()
        self.assertTrue(e["provenance"])  # 正常记录必有 provenance
        # validate 应能识别空 provenance
        bad = {"entity_id": "x", "name": "n", "entity_type": "symptom",
               "aliases": [], "standard_codes": [], "raw_text": "t",
               "provenance": {}, "quality": {"level": "pass", "errors": [], "warnings": []}}
        q = ca.validate_entity(ca.CBLUEEntity(**bad))
        self.assertIn("missing provenance", q["errors"])


class TestSchemaValidation(unittest.TestCase):
    """5. entity/relation schema 校验"""

    def test_validate_entity_ok(self):
        e = ca.parse_cmeee_record(CMEEE_RECORD, source_file="s.json")["entities"][0]
        q = ca.validate_entity(e)
        self.assertEqual(q["level"], "pass")

    def test_validate_relation_unmapped_warns(self):
        res = ca.parse_cmeie_record(CMEIE_RECORD, source_file="s.json")
        unmapped = [r for r in res["relations"] if r.relation_type == "UNMAPPED"][0]
        q = ca.validate_relation(unmapped)
        self.assertEqual(q["level"], "warn")
        self.assertTrue(any("unmapped" in w for w in q["warnings"]))

    def test_validate_relation_unknown_type_error(self):
        r = ca.CBLUERelation(subject="a", relation="X", object="b",
                             relation_type="BAD_TYPE", provenance={"a": 1})
        q = ca.validate_relation(r)
        self.assertEqual(q["level"], "fail")
        self.assertTrue(any("unknown relation_type" in e for e in q["errors"]))

    def test_convert_reruns_validation(self):
        res = ca.ingest(_tmp_dir_with([("cmeee_v2.json", [CMEEE_RECORD])]), "cmeee_v2")
        conv = ca.convert(res)
        self.assertEqual(len(conv["entities"]), 3)
        for e in conv["entities"]:
            self.assertIn("quality", e)


class TestNoInputMutation(unittest.TestCase):
    """6. 不修改输入数据"""

    def test_record_not_mutated(self):
        rec = json.loads(json.dumps(CMEEE_RECORD))
        before = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        ca.parse_cmeee_record(rec, source_file="s.json")
        after = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        self.assertEqual(before, after)

    def test_cmeie_record_not_mutated(self):
        rec = json.loads(json.dumps(CMEIE_RECORD))
        before = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        ca.parse_cmeie_record(rec, source_file="s.json")
        self.assertEqual(before, json.dumps(rec, ensure_ascii=False, sort_keys=True))

    def test_ingest_does_not_mutate_files(self):
        d = _tmp_dir_with([("cmeie_v2.json", [CMEIE_RECORD])])
        f = d / "cmeie_v2.json"
        before = f.read_text(encoding="utf-8")
        ca.ingest(d, "cmeie_v2")
        self.assertEqual(f.read_text(encoding="utf-8"), before)


class TestEmptyData(unittest.TestCase):
    """7. 空数据处理"""

    def test_empty_text(self):
        res = ca.parse_cmeee_record({"sample_id": "x", "text": "  ", "entities": []},
                                    source_file="s.json")
        self.assertEqual(res["entities"], [])
        self.assertEqual(len(res["skipped"]), 1)

    def test_empty_json_array(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as f:
            f.write("[]")
            p = Path(f.name)
        try:
            self.assertEqual(ca._read_records(p), [])
        finally:
            p.unlink(missing_ok=True)

    def test_empty_jsonl(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                         encoding="utf-8") as f:
            f.write("")
            p = Path(f.name)
        try:
            self.assertEqual(ca._read_records(p), [])
        finally:
            p.unlink(missing_ok=True)

    def test_empty_record_ingest(self):
        res = ca.ingest(_tmp_dir_with([("cmeee_v2.json", [])]), "cmeee_v2")
        self.assertEqual(res["records_loaded"], 0)
        self.assertEqual(res["entities"], [])


# ============================================================
# Entity Linking 候选（只读，不 merge）
# ============================================================

class TestEntityLinking(unittest.TestCase):
    def test_link_known_entity(self):
        # 生产 entities.json 含 阿司匹林（entities.json 有）；L1 精确匹配
        prod_map = ca.load_production_name_map()
        self.assertIn("阿司匹林", prod_map.values())
        e = ca.CBLUEEntity(entity_id="cblue:cmeie_v2:train:ie_0001:0",
                           name="阿司匹林", entity_type="unknown", provenance={"d": 1})
        res = ca.link_entities([e], prod_map, ca.load_production_codes())
        self.assertEqual(len(res["linked"]), 1)
        self.assertEqual(res["linked"][0]["canonical"], "阿司匹林")
        self.assertEqual(res["linked"][0]["match_level"], "L1")

    def test_link_via_alias(self):
        prod_map = ca.load_production_name_map()
        # 2型糖尿病 别名 Type 2 Diabetes Mellitus → L2
        e = ca.CBLUEEntity(entity_id="cblue:x", name="Type 2 Diabetes Mellitus",
                           entity_type="unknown", provenance={})
        res = ca.link_entities([e], prod_map, ca.load_production_codes())
        self.assertEqual(len(res["linked"]), 1)
        self.assertEqual(res["linked"][0]["canonical"], "2型糖尿病")
        self.assertIn(res["linked"][0]["match_level"], ("L1", "L2"))

    def test_unmatched(self):
        e = ca.CBLUEEntity(entity_id="cblue:x", name="完全不存在的虚构实体ZZZ",
                           entity_type="unknown", provenance={})
        res = ca.link_entities([e], {}, {})
        self.assertEqual(len(res["unmatched"]), 1)


def _tmp_dir_with(files: list[tuple]) -> Path:
    """创建临时目录写入合成 CBLUE 文件，返回目录 Path。"""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="cblue_test_"))
    for name, records in files:
        (d / name).write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return d


if __name__ == "__main__":
    unittest.main()
