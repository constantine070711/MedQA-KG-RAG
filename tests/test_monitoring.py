# -*- coding: utf-8 -*-
"""Phase 10-C C2+C3 — 旁路可观测性层单元测试（stdlib，零 LLM / DB / 网络）。

覆盖要求（C2，17 项）：
    1. request_id 唯一
    2. request_id 长度正确
    3. question_hash 不包含 question 原文
    4. answer/final_response 原文不进入日志
    5. prompt / modified_prompt / rejection_message 不进入日志
    6. API key/password/token 敏感字段不进入日志
    7. sanitize 处理嵌套 dict/list
    8. 缺失 metrics 不崩溃
    9. 日志写入失败不影响 final_response
    10. log_event 异常静默降级
    11. monitoring 不修改 final_response
    12. BLOCKED 不产生额外 generation
    13. memory_used 正确记录
    14. JSONL 每行可解析
    15. 多次调用 request_id 不重复
    16-17. 白名单无 question/answer 原文键 + sanitize 不修改输入

C3 新增（7 项）：
    - 日志轮转：未超阈值不轮转 / 超阈值滚动到 .1 / 保留 backup_count 份 /
      禁用（max_bytes=None）/ 轮转失败静默 / log_run 端到端行守恒
    - UI 多轮 + Memory 日志：真实 run_agent_sync 路径多轮 request_id 唯一、
      memory_used 序列、blocked 轮 gen=0/not_checked、无 question 原文泄漏

运行:
    python -m unittest tests.test_monitoring -v
"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.monitoring import (  # noqa: E402
    _rotate_if_needed,
    log_event,
    log_run,
    new_request_id,
    question_hash,
    sanitize,
)
from app.ui_helpers import build_conversation_context, run_agent_sync  # noqa: E402


def _make_response(**overrides) -> dict:
    base = {
        "question": "二甲双胍用于治疗什么疾病？",
        "answer": "二甲双胍用于治疗2型糖尿病。",
        "safety_level": "safe",
        "intent": "medical_knowledge",
        "mentions": ["二甲双胍"],
        "neo4j_unavailable": False,
        "llm_generation_count": 1,
        "reliability_decision": {
            "decision": "allow", "reason_code": "RELIABLE_ANSWER",
            "is_final_refusal": False, "confidence": 0.99,
            "reliability_score": 0.95,
        },
        "citations": {"citation_present": True, "grounded_ratio": 1.0},
        "fused_context": {
            "evidence_sources": {"neo4j_graph": {"nodes": 5, "edges": 4, "triples": 3}},
            "evidence": [], "summary": {},
        },
        "coverage": {"neo4j_has_evidence": True, "neo4j_unavailable": False},
        "errors": [],
    }
    base.update(overrides)
    return base


def _blocked_response() -> dict:
    return _make_response(
        safety_level="blocked",
        llm_generation_count=0,
        neo4j_unavailable=False,
        reliability_decision={
            "decision": "refuse", "reason_code": "SAFETY_BLOCKED",
            "is_final_refusal": True, "confidence": 1.0, "reliability_score": 0.1,
        },
        fused_context={"evidence_sources": {"neo4j_graph": {"nodes": 0, "edges": 0, "triples": 0}},
                       "evidence": [], "summary": {}},
        coverage={"neo4j_has_evidence": False, "neo4j_unavailable": False},
        answer="检测到安全边界。",
    )


def _read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


class TestRequestId(unittest.TestCase):
    def test_unique(self):  # 1
        self.assertNotEqual(new_request_id(), new_request_id())

    def test_length(self):  # 2
        rid = new_request_id()
        self.assertEqual(len(rid), 12)
        all(hexdigit in "0123456789abcdef" for hexdigit in rid)

    def test_many_unique(self):  # 15
        ids = {new_request_id() for _ in range(200)}
        self.assertEqual(len(ids), 200)


class TestQuestionHash(unittest.TestCase):
    def test_hash_does_not_contain_question(self):  # 3
        q = "我的非常具体的问题文本"
        h = question_hash(q)
        self.assertNotIn(q, h)
        self.assertEqual(len(h), 16)
        # 相同输入 → 相同 hash（确定性）
        self.assertEqual(question_hash(q), question_hash(q))
        # 不同输入 → 不同 hash（大概率）
        self.assertNotEqual(question_hash(q), question_hash("其他问题"))


class TestLogNoSensitiveLeak(unittest.TestCase):
    def test_answer_not_in_log(self):  # 4
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.jsonl"
            fr = _make_response(answer="机密回答内容ZZZ")
            log_run("r1", "问题文本", fr, 123.4, out=out)
            lines = _read_lines(out)
            self.assertEqual(len(lines), 1)
            self.assertNotIn("机密回答内容ZZZ", lines[0]["question_hash"] or "")
            blob = json.dumps(lines[0], ensure_ascii=False)
            self.assertNotIn("机密回答内容ZZZ", blob)

    def test_prompt_and_rejection_not_in_log(self):  # 5
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.jsonl"
            # 即便 final_response 里混入 prompt 类字段（白名单外的键），log_run 也只提取白名单
            fr = _make_response(
                system_prompt="系统提示词TOP_SECRET",
                modified_prompt="修改后提示词TOP_SECRET",
                rejection_message="拒答消息TOP_SECRET",
            )
            log_run("r1", "q", fr, 1.0, out=out)
            lines = _read_lines(out)
            blob = json.dumps(lines[0], ensure_ascii=False)
            self.assertNotIn("TOP_SECRET", blob)

    def test_api_key_password_token_not_in_log(self):  # 6
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.jsonl"
            fr = _make_response(
                errors=[{"type": "X", "message": "auth failed key=sk-abc123456789 password=secret123 token=xyz"}],
            )
            log_run("r1", "q", fr, 1.0, out=out)
            lines = _read_lines(out)
            blob = json.dumps(lines[0], ensure_ascii=False)
            self.assertNotIn("sk-abc123456789", blob)
            self.assertNotIn("secret123", blob)
            self.assertNotIn("password", blob.lower())


class TestSanitize(unittest.TestCase):
    def test_nested_dict_list(self):  # 7
        data = {"a": [{"b": "password=secret123", "c": "sk-abcdef123456"}], "d": {"e": "ok"}}
        out = sanitize(data)
        self.assertNotIn("secret123", json.dumps(out, ensure_ascii=False))
        self.assertNotIn("sk-abcdef123456", json.dumps(out, ensure_ascii=False))
        self.assertEqual(out["d"]["e"], "ok")  # 正常文本保留
        self.assertIsNot(data, out)  # 返回新结构

    def test_does_not_mutate_input(self):
        data = {"a": "password=secret123"}
        sanitize(data)
        self.assertEqual(data["a"], "password=secret123")


class TestLogFailureSafety(unittest.TestCase):
    def test_missing_metrics_no_crash(self):  # 8
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.jsonl"
            ok = log_run("r1", "q", {}, 1.0, out=out)  # 空 final_response
            self.assertTrue(ok)
            lines = _read_lines(out)
            self.assertEqual(len(lines), 1)
            self.assertIsNone(lines[0].get("decision"))  # 缺失 → null，不伪造

    def test_write_failure_does_not_affect_final_response(self):  # 9
        # 父路径是一个文件 → mkdir 失败 → log 失败，但 fr 原样
        with tempfile.TemporaryDirectory() as d:
            blocker = Path(d) / "blocker"
            blocker.write_text("i am a file", encoding="utf-8")
            bad_out = blocker / "sub" / "m.jsonl"
            fr = _make_response()
            before = copy.deepcopy(fr)
            ok = log_run("r1", "q", fr, 1.0, out=bad_out)
            self.assertFalse(ok)
            self.assertEqual(fr, before)  # final_response 不变

    def test_log_event_serialize_failure_silent(self):  # 10
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.jsonl"
            # set 无法 JSON 序列化 → log_event 捕获并返回 False，不抛异常
            ok = log_event("r1", "info", "test", out=out, bad_field={1, 2, 3})
            self.assertFalse(ok)

    def test_log_run_does_not_mutate_final_response(self):  # 11
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.jsonl"
            fr = _make_response()
            before = copy.deepcopy(fr)
            log_run("r1", "q", fr, 1.0, out=out)
            self.assertEqual(fr, before)


class TestObservationFields(unittest.TestCase):
    def test_blocked_no_extra_generation(self):  # 12
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.jsonl"
            fr = _blocked_response()
            before = copy.deepcopy(fr)
            log_run("r1", "我不想活了", fr, 10.0, out=out)
            self.assertEqual(fr, before)
            rec = _read_lines(out)[0]
            self.assertEqual(rec["safety_level"], "blocked")
            self.assertEqual(rec["generation_count"], 0)
            self.assertEqual(rec["neo4j_status"], "not_checked")
            self.assertEqual(rec["reason_code"], "SAFETY_BLOCKED")
            self.assertTrue(rec["is_final_refusal"])

    def test_memory_used_recorded(self):  # 13
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.jsonl"
            log_run("r1", "q", _make_response(), 1.0,
                    conversation_context="前文", out=out)
            log_run("r2", "q", _make_response(), 1.0,
                    conversation_context=None, out=out)
            recs = _read_lines(out)
            self.assertTrue(recs[0]["memory_used"])
            self.assertFalse(recs[1]["memory_used"])

    def test_jsonl_each_line_parsable(self):  # 14
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.jsonl"
            for i in range(5):
                log_run(f"r{i}", f"问题{i}", _make_response(), float(i), out=out)
            raw = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(raw), 5)
            for ln in raw:
                obj = json.loads(ln)  # 每行可独立解析
                self.assertIn("request_id", obj)
                self.assertIn("timestamp", obj)
                self.assertIn("question_hash", obj)

    def test_whitelist_no_raw_question_key(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.jsonl"
            log_run("r1", "具体问题文本", _make_response(), 1.0, out=out)
            rec = _read_lines(out)[0]
            self.assertIn("question_hash", rec)
            self.assertNotIn("question", rec)  # 不落 question 原文
            self.assertNotIn("answer", rec)


class TestLogRotation(unittest.TestCase):
    """Phase 10-C C3 — 日志大小轮转（_rotate_if_needed + log_run 透传）。"""

    def test_no_rotation_below_max_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "medqa.jsonl"
            p.write_text("x" * 100, encoding="utf-8")
            _rotate_if_needed(p, max_bytes=200, backup_count=2)
            self.assertTrue(p.exists())  # 未超过阈值 → 不轮转
            self.assertFalse((Path(d) / "medqa.jsonl.1").exists())

    def test_rotate_when_exceeded(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "medqa.jsonl"
            p.write_text("x" * 100, encoding="utf-8")
            _rotate_if_needed(p, max_bytes=50, backup_count=2)
            self.assertFalse(p.exists())  # 当前文件已滚为 .1
            self.assertTrue((Path(d) / "medqa.jsonl.1").exists())

    def test_rotate_keeps_backup_count(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "medqa.jsonl"
            p.write_text("CC", encoding="utf-8")
            (Path(d) / "medqa.jsonl.1").write_text("A", encoding="utf-8")
            (Path(d) / "medqa.jsonl.2").write_text("B", encoding="utf-8")
            _rotate_if_needed(p, max_bytes=1, backup_count=2)
            # 滚动后：.1=最新(CC)、.2=旧.1(A)；旧.2(B) 超出 backup_count 被丢弃
            self.assertEqual((Path(d) / "medqa.jsonl.1").read_text(encoding="utf-8"), "CC")
            self.assertEqual((Path(d) / "medqa.jsonl.2").read_text(encoding="utf-8"), "A")
            self.assertFalse((Path(d) / "medqa.jsonl.3").exists())
            self.assertFalse(p.exists())

    def test_rotate_disabled_when_max_bytes_none(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "medqa.jsonl"
            p.write_text("x" * 5000, encoding="utf-8")
            _rotate_if_needed(p, max_bytes=None, backup_count=2)
            self.assertTrue(p.exists())
            self.assertFalse((Path(d) / "medqa.jsonl.1").exists())

    def test_rotation_failure_silent(self):
        # .1 目标是目录 → rename/unlink 失败 → 静默（不抛异常）
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "medqa.jsonl"
            p.write_text("x" * 200, encoding="utf-8")
            (Path(d) / "medqa.jsonl.1").mkdir()
            _rotate_if_needed(p, max_bytes=100, backup_count=1)  # 不应抛异常

    def test_log_run_rotates_and_preserves_rows(self):
        # 端到端：单行 ~479B；max_bytes=900 → 每文件容纳 2 行后轮转，
        # backup_count=2 → 当前 + .1 各 2 行 = 4 行守恒，且每行可独立解析
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.jsonl"
            for i in range(4):
                self.assertTrue(
                    log_run(f"r{i}", f"问题{i}", _make_response(), float(i),
                            out=out, max_bytes=900, backup_count=2)
                )
            total = 0
            seen: set = set()
            for f in (out, Path(f"{out}.1"), Path(f"{out}.2")):
                if f.exists():
                    for ln in f.read_text(encoding="utf-8").splitlines():
                        rec = json.loads(ln)  # 每行可独立解析
                        seen.add(rec["request_id"])
                        total += 1
            self.assertEqual(total, 4)  # 无行丢失
            self.assertEqual(seen, {"r0", "r1", "r2", "r3"})


class _MultiTurnAgent:
    """驱动 run_agent_sync 多轮的假 Agent：记录调用参数，按问题返回正常/blocked 响应。"""

    def __init__(self):
        self.calls = []  # [{"question", "context"}]

    async def run(self, question, conversation_context=None):
        self.calls.append({"question": question, "context": conversation_context})
        q = str(question)
        if any(k in q for k in ("我不想活", "自杀")):
            return _blocked_response()
        return _make_response(answer=f"回答：{q}")


class TestMultiTurnLogging(unittest.TestCase):
    """Phase 10-C C3a — UI 多轮 + Memory 日志场景补验（真实 run_agent_sync 路径）。

    场景：首轮无 memory → 二轮追问（memory 生效）→ blocked 轮（memory 传入但不参与生成）
          → 换主题正常轮（blocked 轮不入 memory）。
    """

    def test_multiturn_memory_log_rows(self):
        log_path = _PROJECT_ROOT / "data" / "logs" / "medqa.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        before = len(log_path.read_text(encoding="utf-8").splitlines()) if log_path.exists() else 0

        agent = _MultiTurnAgent()
        history: list = []
        turns = [
            "二甲双胍用于治疗什么疾病？",  # 1 首轮：无 memory
            "它有什么常见副作用？",        # 2 追问：memory 生效
            "我不想活了",                  # 3 blocked：context 传入但 generation=0
            "高血压怎么治疗？",            # 4 换主题：blocked 轮不入 memory，仍含前两轮
        ]
        for q in turns:
            memory = build_conversation_context(history)
            fr = run_agent_sync(agent, q, conversation_context=memory)
            history.append({"question": q, "final_response": fr})

        # 本次新增恰好 4 行
        rows = [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines()[before:]]
        self.assertEqual(len(rows), 4)

        # 每轮 request_id 唯一（12 hex）
        ids = [r["request_id"] for r in rows]
        self.assertEqual(len(set(ids)), 4)
        self.assertTrue(all(len(i) == 12 and i.isalnum() for i in ids))

        # memory_used：首轮 False，之后 context 已传入（含 blocked 轮）为 True
        self.assertEqual([r["memory_used"] for r in rows], [False, True, True, True])

        # 第 3 轮（blocked）：gen=0 / not_checked / final_refusal；context 虽传入但不参与生成
        self.assertEqual(rows[2]["safety_level"], "blocked")
        self.assertEqual(rows[2]["generation_count"], 0)
        self.assertEqual(rows[2]["neo4j_status"], "not_checked")
        self.assertTrue(rows[2]["is_final_refusal"])

        # 第 4 轮换主题正常（非 blocked）
        self.assertNotEqual(rows[3]["safety_level"], "blocked")

        # 假 Agent 收到的 context：首轮 None，之后非空（与 UI 传入一致）
        self.assertIsNone(agent.calls[0]["context"])
        self.assertTrue(all(c["context"] for c in agent.calls[1:]))

        # 整段对话日志无 question 原文泄漏
        blob = json.dumps(rows, ensure_ascii=False)
        for q in turns:
            self.assertNotIn(q, blob)


if __name__ == "__main__":
    unittest.main()
