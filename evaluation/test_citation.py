"""Phase 5B B3 — CitationExtractor / CitationVerifier 单元测试。

运行方式：
    python -m unittest evaluation.test_citation -v

覆盖用例（对齐 PROJECT_HANDOFF B3 测试要求）：
    1. [来源: xxx]            → source_tag
    2. ### References + - [N]  → references_section
    3. - [1] xxx 裸 bullet     → 正常提取
    4. * [1] xxx               → 提取英文 entity_name
    5. 中文 entity name 归一化 → grounded
    6. 英文 entity name（D4）  → grounded
    7. alias                   → grounded
    8. source_id→chunk_id 映射 → source_id_chunk_mapping
    9. citation 存在但无法 grounded → ungrounded
    10. 没有 citation          → present=False
    11. 空 answer              → 全默认，不崩溃
"""

import unittest
from pathlib import Path

from evaluation.citation_extractor import CitationResult, ReferenceItem, extract_citations
from evaluation.citation_verifier import analyze_citations, verify_citations
from evaluation.metrics import load_entity_name_map

# 真实 entities.json 的 D4 映射（别名用例用真实数据证明 D4 路径）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENTITIES_JSON = _PROJECT_ROOT / "data" / "entities" / "entities.json"

# 合成检索实体（模拟 aquery_data().entities[] 结构）
_DIABETES_ENTITY = {
    "entity_name": "2型糖尿病",
    "entity_type": "concept",
    "description": "一种慢性代谢性疾病",
    "source_id": "doc-aaa-chunk-000",
    "file_path": "unknown_source",
}
_ASTHMA_ENTITY = {
    "entity_name": "支气管哮喘",
    "entity_type": "concept",
    "description": "一种慢性气道炎症性疾病",
    "source_id": "doc-bbb-chunk-000",
    "file_path": "unknown_source",
}

# 合成检索 chunk
_DIABETES_CHUNK = {
    "content": "2型糖尿病的常见症状包括多饮、多尿、体重下降。",
    "chunk_id": "doc-aaa-chunk-000",
    "file_path": "unknown_source",
}


def _name_map():
    if _ENTITIES_JSON.exists():
        return load_entity_name_map(str(_ENTITIES_JSON))
    # 兜底（无真实数据时仍能测 Tier 1 基本路径）
    return {
        "2型糖尿病": "2型糖尿病",
        "type 2 diabetes mellitus": "2型糖尿病",
        "支气管哮喘": "支气管哮喘",
        "哮喘": "支气管哮喘",
    }


class TestCitationExtractor(unittest.TestCase):
    """extract_citations() 的格式提取测试。"""

    def test_source_tag(self):
        answer = "二甲双胍用于治疗2型糖尿病。[来源: data/raw/drugs/metformin.md]"
        result = extract_citations(answer=answer)
        self.assertTrue(result.citation_present)
        self.assertEqual(result.citation_format, "source_tag")
        self.assertEqual(len(result.reference_items), 1)
        self.assertEqual(result.reference_items[0].source_file, "metformin.md")

    def test_references_section_list_items(self):
        answer = "### References\n- [1] 二甲双胍 (Metformin)\n- [2] 2型糖尿病\n"
        result = extract_citations(answer=answer)
        self.assertTrue(result.citation_present)
        self.assertEqual(result.citation_format, "references_section")
        self.assertEqual(len(result.reference_items), 2)
        self.assertEqual(result.reference_items[0].index, 1)
        self.assertEqual(result.reference_items[0].entity_name, "二甲双胍")
        self.assertEqual(result.reference_items[1].index, 2)
        self.assertEqual(result.reference_items[1].entity_name, "2型糖尿病")

    def test_bare_bullet_item(self):
        answer = "## References\n- [1] 高血压\n"
        result = extract_citations(answer=answer)
        self.assertTrue(result.citation_present)
        self.assertEqual(result.citation_format, "references_section")
        self.assertEqual(len(result.reference_items), 1)
        self.assertEqual(result.reference_items[0].entity_name, "高血压")

    def test_star_bullet_english(self):
        answer = "### References\n* [1] Bronchial Asthma\n"
        result = extract_citations(answer=answer)
        self.assertTrue(result.citation_present)
        self.assertEqual(result.citation_format, "references_section")
        self.assertEqual(len(result.reference_items), 1)
        self.assertEqual(result.reference_items[0].entity_name, "Bronchial Asthma")

    def test_no_citation(self):
        result = extract_citations(answer="高血压的常见症状包括头痛和头晕。")
        self.assertFalse(result.citation_present)
        self.assertIsNone(result.citation_format)
        self.assertEqual(result.reference_items, [])

    def test_empty_answer(self):
        result = extract_citations(answer="")
        self.assertFalse(result.citation_present)
        self.assertIsNone(result.citation_format)
        self.assertEqual(result.reference_items, [])
        self.assertFalse(result.citation_grounded)
        self.assertEqual(result.grounded_ratio, 0.0)


class TestCitationVerifier(unittest.TestCase):
    """verify_citations() / analyze_citations() 的追溯测试。"""

    def test_chinese_entity_grounded(self):
        answer = "### References\n- [1] 2型糖尿病\n"
        result = analyze_citations(
            answer=answer,
            retrieved_entities=[_DIABETES_ENTITY],
            retrieved_chunks=[_DIABETES_CHUNK],
            entity_name_map=_name_map(),
        )
        self.assertTrue(result.citation_grounded)
        self.assertIn("2型糖尿病", result.grounded_entities)
        self.assertEqual(result.ungrounded_entities, [])
        self.assertEqual(result.grounded_ratio, 1.0)

    def test_english_entity_d4_normalization(self):
        # 引用英文名，检索实体是中文名，D4 归一化后应匹配
        answer = "### References\n- [1] Type 2 Diabetes Mellitus\n"
        result = analyze_citations(
            answer=answer,
            retrieved_entities=[_DIABETES_ENTITY],
            retrieved_chunks=[_DIABETES_CHUNK],
            entity_name_map=_name_map(),
        )
        self.assertTrue(result.citation_grounded)
        self.assertIn("2型糖尿病", result.grounded_entities)

    def test_alias_grounded(self):
        # 引用别名「哮喘」，检索实体是「支气管哮喘」
        answer = "### References\n- [1] 哮喘\n"
        result = analyze_citations(
            answer=answer,
            retrieved_entities=[_ASTHMA_ENTITY],
            retrieved_chunks=[],
            entity_name_map=_name_map(),
        )
        self.assertTrue(result.citation_grounded)
        self.assertIn("支气管哮喘", result.grounded_entities)

    def test_source_id_chunk_mapping(self):
        # Tier 1 命中 + source_id 指向 chunk → grounding_method 应为 source_id_chunk_mapping
        answer = "### References\n- [1] 2型糖尿病\n"
        result = analyze_citations(
            answer=answer,
            retrieved_entities=[_DIABETES_ENTITY],
            retrieved_chunks=[_DIABETES_CHUNK],
            entity_name_map=_name_map(),
        )
        self.assertTrue(result.citation_grounded)
        self.assertEqual(result.grounding_method, "source_id_chunk_mapping")

    def test_ungrounded_citation(self):
        # 引用实体不在检索结果中，也无法子串匹配 → ungrounded
        answer = "### References\n- [1] 不存在实体XYZ\n"
        result = analyze_citations(
            answer=answer,
            retrieved_entities=[_DIABETES_ENTITY],
            retrieved_chunks=[_DIABETES_CHUNK],
            entity_name_map=_name_map(),
        )
        self.assertTrue(result.citation_present)
        self.assertFalse(result.citation_grounded)
        self.assertEqual(result.grounded_entities, [])
        self.assertEqual(len(result.ungrounded_entities), 1)
        self.assertEqual(result.grounded_ratio, 0.0)
        self.assertEqual(result.grounding_method, "none")

    def test_chunk_content_substring_fallback(self):
        # Tier 1 未命中（检索实体不匹配），但引用文本命中 chunk 内容 → chunk_content_substring
        answer = "### References\n- [1] 体重下降\n"
        result = analyze_citations(
            answer=answer,
            retrieved_entities=[_ASTHMA_ENTITY],
            retrieved_chunks=[_DIABETES_CHUNK],
            entity_name_map=_name_map(),
        )
        self.assertTrue(result.citation_grounded)
        self.assertEqual(result.grounding_method, "chunk_content_substring")

    def test_empty_answer_no_crash(self):
        result = analyze_citations(
            answer="",
            retrieved_entities=[_DIABETES_ENTITY],
            retrieved_chunks=[_DIABETES_CHUNK],
            entity_name_map=_name_map(),
        )
        self.assertFalse(result.citation_present)
        self.assertFalse(result.citation_grounded)
        self.assertEqual(result.total_references, 0)
        self.assertEqual(result.grounding_method, "none")


if __name__ == "__main__":
    unittest.main(verbosity=2)
