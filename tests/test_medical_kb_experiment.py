# -*- coding: utf-8 -*-
"""Phase 10-A.5 Step 10-A — Experiment Corpus 准备单元测试。

覆盖：5000 条限制 / deterministic ordering / input 不被修改 / 生产隔离 /
      experiment 目录独立 / provenance 保留 / 空数据处理 / duplicate 处理 /
      rerun 幂等 / 输出 schema / dry-run 预估字段。
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 通过文件路径加载 scripts 脚本（scripts/ 非包）
_SPEC = importlib.util.spec_from_file_location(
    "medkb_exp_prepare", PROJECT_ROOT / "scripts" / "medical_kb_experiment_prepare.py")
MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MOD)
build_corpus = MOD.build_corpus
write_outputs = MOD.write_outputs


def _mk_train(tmp: Path, n: int = 6, dup_text: bool = False) -> Path:
    """构造合成 CMeIE train jsonl。"""
    p = tmp / "CMeIE_train.jsonl"
    lines = []
    for i in range(n):
        text = f"高血压@{i}号疾病有头晕和头痛症状，需注意。"
        if dup_text and i % 2 == 1:
            text = f"高血压@{i}号疾病有头晕和头痛症状，需注意。"  # 与前一重复
        lines.append(json.dumps({
            "text": text,
            "spo_list": [{
                "Combined": False,
                "predicate": "并发症",
                "subject": f"疾病{i}",
                "subject_type": "疾病",
                "object": {"@value": "脑卒中"},
                "object_type": {"@value": "疾病"},
            }]
        }, ensure_ascii=False))
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


class TestBuildCorpus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.train = _mk_train(Path(self.tmp), n=6)
        self.cleaned = Path(self.tmp) / "cleaned_records.jsonl"
        _cleaned_lines = [json.dumps({"provenance": {"sample_id": str(i)}})
                          for i in range(6)]
        self.cleaned.write_text("\n".join(_cleaned_lines), encoding="utf-8")

    def test_limit(self):
        res = build_corpus(limit=3, verbose=False,
                           train_src=self.train, cleaned=self.cleaned)
        self.assertEqual(len(res["documents"]), 3)
        self.assertEqual(res["summary"]["corpus_size"], 3)
        self.assertEqual(res["summary"]["limit"], 3)

    def test_default_limit_cap(self):
        """默认 limit=5000，少于 5000 时取全部可用。"""
        res = build_corpus(limit=None, verbose=False,
                           train_src=self.train, cleaned=self.cleaned)
        self.assertEqual(len(res["documents"]), 6)

    def test_deterministic_ordering(self):
        a = build_corpus(limit=6, verbose=False,
                         train_src=self.train, cleaned=self.cleaned)
        b = build_corpus(limit=6, verbose=False,
                         train_src=self.train, cleaned=self.cleaned)
        self.assertEqual([d["doc_id"] for d in a["documents"]],
                         [d["doc_id"] for d in b["documents"]])
        self.assertEqual(a["documents"], b["documents"])

    def test_input_not_modified(self):
        before = self.train.read_bytes()
        build_corpus(limit=6, verbose=False,
                     train_src=self.train, cleaned=self.cleaned)
        self.assertEqual(self.train.read_bytes(), before)

    def test_provenance_preserved(self):
        res = build_corpus(limit=2, verbose=False,
                           train_src=self.train, cleaned=self.cleaned)
        for d in res["documents"]:
            prov = d["provenance"]
            self.assertEqual(prov["dataset"], "CMeIE")
            self.assertEqual(prov["license"], "CC-BY-NC 4.0")
            self.assertIn("sample_id", prov)
            self.assertIn("source_file", prov)
            self.assertEqual(d["sample_id"], int(prov["sample_id"]))

    def test_empty_lines_handled(self):
        p = Path(self.tmp) / "CMeIE_train.jsonl"
        p.write_text("\n\n" + _mk_train(Path(self.tmp), n=2).read_text(encoding="utf-8") + "\n\n",
                     encoding="utf-8")
        res = build_corpus(limit=6, verbose=False,
                           train_src=p, cleaned=self.cleaned)
        # 空行被跳过，sample_id 仍按行号顺序
        self.assertGreaterEqual(len(res["documents"]), 2)

    def test_duplicate_text_counted(self):
        dup = Path(self.tmp) / "dup.jsonl"
        text = "高血压@重复文本内容。"
        dup.write_text("\n".join([
            json.dumps({"text": text, "spo_list": []}, ensure_ascii=False),
            json.dumps({"text": text, "spo_list": []}, ensure_ascii=False),
            json.dumps({"text": "另一段文本内容。", "spo_list": []}, ensure_ascii=False),
        ]), encoding="utf-8")
        res = build_corpus(limit=3, verbose=False,
                           train_src=dup, cleaned=self.cleaned)
        self.assertEqual(res["summary"]["duplicate_text_in_corpus"], 1)
        self.assertEqual(res["summary"]["unique_text_count"], 2)

    def test_summary_schema(self):
        res = build_corpus(limit=2, verbose=False,
                           train_src=self.train, cleaned=self.cleaned)
        s = res["summary"]
        for key in ("corpus_size", "total_chars", "avg_chars_per_doc",
                    "total_tokens_est", "est_chunks", "chunk_token_size",
                    "est_llm_extract_calls", "duplicate_text_in_corpus",
                    "deterministic", "reuses_existing_doc", "production_untouched"):
            self.assertIn(key, s)
        # dry-run 预估字段必须为非负数值
        self.assertGreaterEqual(s["est_chunks"], 0)
        self.assertGreaterEqual(s["est_llm_extract_calls"], 0)


class TestWriteOutputs(unittest.TestCase):
    def test_writes_to_specified_dir(self):
        tmp = Path(tempfile.mkdtemp())
        out = tmp / "exp_out"
        write_outputs([{"doc_id": "x", "text": "t"}], {"corpus_size": 1}, out)
        self.assertTrue((out / "documents.jsonl").exists())
        self.assertTrue((out / "summary.json").exists())
        docs = [json.loads(l) for l in (out / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(docs[0]["doc_id"], "x")
        self.assertEqual(json.loads((out / "summary.json").read_text(encoding="utf-8"))["corpus_size"], 1)

    def test_idempotent_write(self):
        tmp = Path(tempfile.mkdtemp())
        out = tmp / "exp_out"
        data = [{"doc_id": f"medkb_{i}", "text": f"t{i}"} for i in range(3)]
        write_outputs(data, {"corpus_size": 3}, out)
        h1 = (out / "documents.jsonl").read_bytes()
        write_outputs(data, {"corpus_size": 3}, out)
        self.assertEqual((out / "documents.jsonl").read_bytes(), h1)

    def test_out_dir_independent_from_production(self):
        """写出的目录是传入的 out_dir，不触碰 data/processed/medical_kb（生产产物）。"""
        tmp = Path(tempfile.mkdtemp())
        out = tmp / "exp_out"
        write_outputs([], {"corpus_size": 0}, out)
        # 不在 production 产物目录创建文件
        self.assertFalse((PROJECT_ROOT / "data" / "processed" / "medical_kb"
                          / "documents.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
