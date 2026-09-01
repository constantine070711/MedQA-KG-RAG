#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 6 F6 — eval_phase6.py：端到端对比评估（baseline vs Neo4j 证据融合）。

对 28 题（Q001-Q028）执行两种模式，**唯一实验变量 = 是否注入 Neo4j 结构化证据**：

    A. baseline（无 Neo4j）: system_prompt = _base_safety_prompt(safety)
       （SAFE→None / CAUTION→SYSTEM_PROMPT_SAFE / RESTRICTED→modified_prompt / BLOCKED→拒答，
         与 EvaluationRunner B2 逐行一致）
    B. neo4j（有 Neo4j）:   system_prompt = build_fusion_prompt(neo4j_result,
                              base_prompt=_base_safety_prompt(safety))

两条路径使用**完全相同的评估代码**（镜像 EvaluationRunner._evaluate_one：
retrieve → SafetyChecker → query_with_context → M1-M6 → detect_refusal →
analyze_citations → extract_signals → M8-M12 → make_reliability_decision），
复用全部现有 metrics / decision / F5 函数，**不修改任何现有文件**。

约束（对齐 F6 需求）:
    - 每题最多一次 LLM generation（BLOCKED 0 次）
    - M1-M4 使用 retrieve()（缓存命中，零新增 generation）
    - 失败题如实记录，不静默跳过
    - Neo4j 不可达 → 按 F5 降级为纯 LightRAG + neo4j_unavailable=true
    - improved/regressed 判定阈值固定 ±0.05（不可改）
    - 不硬编码 qid / GT / 题目答案

用法:
    python scripts/eval_phase6.py --mode baseline --no-llm-judge        # baseline 等价性验证
    python scripts/eval_phase6.py --mode both --llm-judge --output data/eval/phase6_report.json
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataclasses import asdict

from src.medical_core.safety import SafetyChecker, SafetyLevel
from src.evidence_fusion import (
    FusionConfig,
    build_fusion_prompt,
    extract_curated_mentions,
    fuse_evidence,
)
from scripts.run_graph_query import _retrieve_neo4j, _base_safety_prompt
from evaluation.metrics import (
    load_entity_name_map,
    retrieval_entity_recall,
    retrieval_chunk_relevance,
    answer_entity_recall,
    context_precision,
    faithfulness,
    faithfulness_llm_judge,
    citation_coverage,
    refusal_accuracy,
    uncertainty_score_metric,
    citation_quality,
    safety_signal_presence,
    combined_reliability,
)
from evaluation.runner import detect_refusal, _safe_get_data, _safe_get_list
from evaluation.citation_verifier import analyze_citations
from evaluation.uncertainty_signals import extract_signals
from evaluation.reliability_decision import make_reliability_decision

logger = logging.getLogger(__name__)

# 确定性指标（improved/regressed 判定用；M5 LLM Judge 与 M7 明确排除）
DETERMINISTIC_METRIC_KEYS = ("M1", "M2", "M3", "M4", "M6", "M8", "M9", "M10", "M12")
CLASSIFY_THRESHOLD = 0.05  # 固定阈值，不可改


# ============================================================
# 纯函数（可测试，无 IO / 无 LLM / 无 DB）
# ============================================================

def _mean(scores: list[float]) -> Optional[float]:
    return round(sum(scores) / len(scores), 4) if scores else None


def select_system_prompt(
    safety,
    with_neo4j: bool = False,
    neo4j_result: Optional[dict] = None,
    config: Optional[FusionConfig] = None,
) -> Optional[str]:
    """构造 system_prompt。

    - baseline: 仅 _base_safety_prompt(safety)（与 runner B2 一致）
    - neo4j:    build_fusion_prompt(neo4j_result, base_prompt=base)
      （无 Neo4j 证据 / 不可达时返回 base 原样 → SAFE 保持 None = LightRAG 默认）
    """
    base = _base_safety_prompt(safety)
    if not with_neo4j:
        return base
    return build_fusion_prompt(neo4j_result, base_prompt=base, config=config)


def _metric_score_of(record: dict, key: str) -> Optional[float]:
    m = record.get("metrics", {})
    v = m.get(key)
    if not isinstance(v, dict):
        return None
    return v.get("score")


def _m6_actual_score(record: dict) -> Optional[float]:
    m = record.get("metrics", {})
    v = m.get("M6")
    if not isinstance(v, dict):
        return None
    act = v.get("actual")
    if isinstance(act, dict):
        return act.get("score")
    return None


def _deterministic_score(record: dict) -> Optional[float]:
    """每题适用的非空确定性指标均值（M1/M2/M3/M4/M6/M8/M9/M10/M12）。

    M5（LLM Judge）与 M7（拒答正确性）明确排除，避免把 LLM 波动混入。
    """
    scores: list[float] = []
    for key in DETERMINISTIC_METRIC_KEYS:
        if key == "M6":
            s = _m6_actual_score(record)
        else:
            s = _metric_score_of(record, key)
        if isinstance(s, (int, float)):
            scores.append(float(s))
    if not scores:
        return None
    return sum(scores) / len(scores)


def classify_question(
    score_baseline: Optional[float],
    score_neo4j: Optional[float],
    threshold: float = CLASSIFY_THRESHOLD,
) -> str:
    """improved / regressed / unchanged（阈值固定 ±0.05）。

    任一侧缺分（如该题在某一模式失败）→ 保守判 unchanged（失败另行记录）。
    """
    if score_baseline is None or score_neo4j is None:
        return "unchanged"
    # round 到 4 位消除浮点边界歧义（0.45-0.5 应视为 -0.05）
    delta = round(score_neo4j - score_baseline, 4)
    if delta >= threshold:
        return "improved"
    if delta <= -threshold:
        return "regressed"
    return "unchanged"


def _m5_parts(record: dict) -> tuple[Optional[float], Optional[float]]:
    m = record.get("metrics", {})
    v = m.get("M5")
    if not isinstance(v, dict):
        return None, None
    h = v.get("heuristic")
    lj = v.get("llm_judge")
    return (h.get("score") if isinstance(h, dict) else None,
            lj.get("score") if isinstance(lj, dict) else None)


def aggregate_metrics(records: list[dict]) -> dict:
    """聚合 M1-M12 平均（与 runner _build_summary 一致：非 None score 平均）。"""
    success = [r for r in records if r["status"] == "success"]

    def _coll(key_fn):
        vals = [v for r in success if (v := key_fn(r)) is not None]
        return {"avg": _mean(vals), "count": len(vals)}

    m6_present = sum(1 for r in success if _m6_present(r))
    m6_grounded = sum(1 for r in success if _m6_grounded(r))
    m6_applicable = sum(1 for r in success if _m6_applicable(r))

    return {
        "M1": _coll(lambda r: _metric_score_of(r, "M1")),
        "M2": _coll(lambda r: _metric_score_of(r, "M2")),
        "M3": _coll(lambda r: _metric_score_of(r, "M3")),
        "M4": _coll(lambda r: _metric_score_of(r, "M4")),
        "M5_heuristic": _coll(lambda r: _m5_parts(r)[0]),
        "M5_llm_judge": _coll(lambda r: _m5_parts(r)[1]),
        "M6_potential": _coll(lambda r: _m6_potential(r)),
        "M6_actual": _coll(lambda r: _m6_actual_score(r)),
        "M6_present_count": m6_present,
        "M6_grounded_count": m6_grounded,
        "M6_applicable_count": m6_applicable,
        "M8": _coll(lambda r: _metric_score_of(r, "M8")),
        "M9": _coll(lambda r: _metric_score_of(r, "M9")),
        "M10": _coll(lambda r: _metric_score_of(r, "M10")),
        "M11": {"defined": False, "note": "本项目未定义 M11（指标为 M1-M10 + M12）"},
        "M12": _coll(lambda r: _metric_score_of(r, "M12")),
    }


def _m6_present(r: dict) -> bool:
    v = r.get("metrics", {}).get("M6")
    return bool(isinstance(v, dict) and v.get("citation_present"))


def _m6_grounded(r: dict) -> bool:
    v = r.get("metrics", {}).get("M6")
    return bool(isinstance(v, dict) and v.get("citation_grounded"))


def _m6_applicable(r: dict) -> bool:
    v = r.get("metrics", {}).get("M6")
    return bool(r.get("key_facts")) or bool(isinstance(v, dict) and v.get("citation_present") is not None)


def _m6_potential(r: dict) -> Optional[float]:
    v = r.get("metrics", {}).get("M6")
    if isinstance(v, dict) and isinstance(v.get("potential"), dict):
        return v["potential"].get("score")
    return None


def aggregate_m7(records: list[dict]) -> dict:
    """M7 混淆矩阵（复用 refusal_accuracy；错误题不参与但单独记录）。"""
    m7_input = [
        {
            "question_id": r.get("question_id", "?"),
            "expected_refusal": bool(r.get("expected_refusal", False)),
            "actual_refusal": bool(r.get("actual_refusal", False)),
            "safety_level": r.get("safety_level", "caution"),
        }
        for r in records
        if r.get("status") == "success"
    ]
    return refusal_accuracy(m7_input)


def aggregate_decisions(records: list[dict]) -> dict:
    """refuse/revise/allow 分布 + reason_code 分布。"""
    dist: dict[str, int] = {"refuse": 0, "revise": 0, "allow": 0, "other": 0}
    reason_codes: dict[str, int] = {}
    for r in records:
        if r.get("status") != "success":
            continue
        d = r.get("reliability_decision")
        if isinstance(d, dict) and d.get("decision"):
            dist[d.get("decision")] = dist.get(d.get("decision"), 0) + 1
            rc = d.get("reason_code", "?")
            reason_codes[rc] = reason_codes.get(rc, 0) + 1
        else:
            dist["other"] += 1
    return {"distribution": dist, "reason_codes": reason_codes}


def _compact_record(r: dict) -> dict:
    """压缩单题记录（保留可追踪字段，控制报告体积）。"""
    m = r.get("metrics", {})
    return {
        "question_id": r.get("question_id"),
        "status": r.get("status"),
        "error": r.get("error"),
        "category": r.get("category"),
        "expected_refusal": bool(r.get("expected_refusal", False)),
        "actual_refusal": r.get("actual_refusal"),
        "decision": (r.get("reliability_decision") or {}).get("decision"),
        "reason_code": (r.get("reliability_decision") or {}).get("reason_code"),
        "safety_level": r.get("safety_level"),
        "safety_check_level": r.get("safety_check_level"),
        "with_neo4j": bool(r.get("with_neo4j", False)),
        "mentions": r.get("mentions"),
        "neo4j_unavailable": r.get("neo4j_unavailable"),
        "neo4j_provenance": r.get("neo4j_provenance"),
        "metric_scores": {k: _metric_score_of(r, k) for k in DETERMINISTIC_METRIC_KEYS},
        "M5_heuristic": _m5_parts(r)[0],
        "M5_llm_judge": _m5_parts(r)[1],
        "M6_present": _m6_present(r),
        "M6_grounded": _m6_grounded(r),
        "llm_response_prefix": (r.get("llm_response") or "")[:100],
    }


def build_comparison(baseline_records: list[dict], neo4j_records: list[dict]) -> dict:
    """M1-M12 双模式对比（M5 区分 heuristic/llm_judge；M6 用 actual）。"""
    ab = aggregate_metrics(baseline_records)
    an = aggregate_metrics(neo4j_records)

    def _cmp(key):
        b, n = ab[key]["avg"], an[key]["avg"]
        delta = round(n - b, 4) if (b is not None and n is not None) else None
        return {"baseline": b, "neo4j": n, "delta": delta}

    m5b_h, m5b_j = ab["M5_heuristic"]["avg"], ab["M5_llm_judge"]["avg"]
    m5n_h, m5n_j = an["M5_heuristic"]["avg"], an["M5_llm_judge"]["avg"]
    m6b, m6n = ab["M6_actual"]["avg"], an["M6_actual"]["avg"]

    return {
        "M1": _cmp("M1"),
        "M2": _cmp("M2"),
        "M3": _cmp("M3"),
        "M4": _cmp("M4"),
        "M5": {
            "heuristic": {"baseline": m5b_h, "neo4j": m5n_h,
                          "delta": round(m5n_h - m5b_h, 4) if (m5b_h is not None and m5n_h is not None) else None},
            "llm_judge": {"baseline": m5b_j, "neo4j": m5n_j, "llm_judge_nondeterministic": True,
                          "delta": round(m5n_j - m5b_j, 4) if (m5b_j is not None and m5n_j is not None) else None},
        },
        "M6": {
            "actual": {"baseline": m6b, "neo4j": m6n,
                       "delta": round(m6n - m6b, 4) if (m6b is not None and m6n is not None) else None},
            "present": {"baseline": ab["M6_present_count"], "neo4j": an["M6_present_count"]},
            "grounded": {"baseline": ab["M6_grounded_count"], "neo4j": an["M6_grounded_count"]},
        },
        "M7": {
            "baseline": aggregate_m7(baseline_records),
            "neo4j": aggregate_m7(neo4j_records),
        },
        "M8": _cmp("M8"),
        "M9": _cmp("M9"),
        "M10": _cmp("M10"),
        "M11": {"defined": False, "note": "本项目未定义 M11（指标为 M1-M10 + M12）"},
        "M12": _cmp("M12"),
    }


def build_impact(baseline_records: list[dict], neo4j_records: list[dict]) -> dict:
    """Neo4j 影响：per-question hits + improved/regressed/unchanged + decision flips。"""
    n_by_id = {r.get("question_id"): r for r in neo4j_records}
    improved, regressed, unchanged = [], [], []
    improved_by_decision, regressed_by_decision = [], []
    per_q_hits: dict = {}

    for rb in baseline_records:
        qid = rb.get("question_id")
        rn = n_by_id.get(qid)
        if rn is None:
            continue
        # Neo4j 命中 / provenance
        prov = rn.get("neo4j_provenance") or {}
        per_q_hits[qid] = {
            "mentions": rn.get("mentions", []),
            "neo4j_has_evidence": prov.get("neo4j_has_evidence"),
            "evidence_sources": prov.get("evidence_sources"),
            "coverage_notes": prov.get("coverage_notes"),
            "neo4j_unavailable": bool(rn.get("neo4j_unavailable", False)),
        }
        # 确定性指标分类（M7 不混入）
        sb = _deterministic_score(rb)
        sn = _deterministic_score(rn)
        cls = classify_question(sb, sn)
        if cls == "improved":
            improved.append(qid)
        elif cls == "regressed":
            regressed.append(qid)
        else:
            unchanged.append(qid)
        # M7 决策正确性翻转（单独统计）
        cb = bool(rb.get("actual_refusal")) == bool(rb.get("expected_refusal", False))
        cn = bool(rn.get("actual_refusal")) == bool(rn.get("expected_refusal", False))
        if cb and not cn:
            regressed_by_decision.append(qid)
        elif (not cb) and cn:
            improved_by_decision.append(qid)

    return {
        "per_question_neo4j_hits": per_q_hits,
        "classify_rule": {
            "deterministic_metrics": list(DETERMINISTIC_METRIC_KEYS),
            "threshold": CLASSIFY_THRESHOLD,
            "note": "Δ=score_neo4j-score_baseline; Δ>=+0.05 improved, Δ<=-0.05 regressed, 否则 unchanged; 任一侧缺分→unchanged",
        },
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "improved_by_decision": improved_by_decision,
        "regressed_by_decision": regressed_by_decision,
        "m7_flip_stats": {
            "correct_to_incorrect": regressed_by_decision,
            "incorrect_to_correct": improved_by_decision,
        },
    }


def build_report(
    questions: list[dict],
    baseline_records: list[dict],
    neo4j_records: list[dict],
) -> dict:
    """组装 phase6_report.json（含 execution / 双模式 / comparison / neo4j_impact）。"""
    def _exec_block(records: list[dict]) -> dict:
        success = [r for r in records if r["status"] == "success"]
        return {
            "success": len(success),
            "failed": len(records) - len(success),
            "failed_details": [
                {"question_id": r.get("question_id"), "error": r.get("error")}
                for r in records if r["status"] != "success"
            ],
        }

    return {
        "phase": "6F6",
        "dataset_size": len(questions),
        "execution": {
            "baseline": _exec_block(baseline_records),
            "neo4j": _exec_block(neo4j_records),
        },
        "baseline_without_neo4j": {
            "metrics": aggregate_metrics(baseline_records),
            "decision_summary": aggregate_decisions(baseline_records),
            "m7": aggregate_m7(baseline_records),
            "per_question": [_compact_record(r) for r in baseline_records],
        },
        "with_neo4j": {
            "metrics": aggregate_metrics(neo4j_records),
            "decision_summary": aggregate_decisions(neo4j_records),
            "m7": aggregate_m7(neo4j_records),
            "per_question": [_compact_record(r) for r in neo4j_records],
        },
        "comparison": build_comparison(baseline_records, neo4j_records),
        "neo4j_impact": build_impact(baseline_records, neo4j_records),
        "notes": {
            "unique_variable": "是否注入 Neo4j 结构化证据（system_prompt）；其余评估代码完全一致",
            "m5_llm_judge_nondeterministic": True,
            "deterministic_metrics": list(DETERMINISTIC_METRIC_KEYS),
            "m7_note": "M7 由 ReliabilityDecision.is_final_refusal 合成（与 Phase 5C 一致）",
        },
    }


# ============================================================
# 评估器（镜像 EvaluationRunner._evaluate_one，仅 system_prompt 可变）
# ============================================================

class Phase6Evaluator:
    """双模式评估器。engine/retriever 可注入（便于测试，不触真实 LLM/DB）。"""

    def __init__(
        self,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 20,
        enable_llm_judge: bool = False,
        fresh: bool = False,
        config: Optional[FusionConfig] = None,
        engine=None,
        retriever=None,
        safety_checker: Optional[SafetyChecker] = None,
        name_map: Optional[dict] = None,
    ):
        self.mode = mode
        self.top_k = top_k
        self.chunk_top_k = chunk_top_k
        self.enable_llm_judge = enable_llm_judge
        self.fresh = fresh  # 禁用 LightRAG 查询缓存，保证 generation 真实执行（证据真正进入 LLM）
        self.config = config or FusionConfig()
        self.engine = engine
        self.retriever = retriever
        self.safety_checker = safety_checker or SafetyChecker()
        self.name_map = name_map if name_map is not None else load_entity_name_map()
        self._judge_llm_func = None

    async def initialize(self) -> "Phase6Evaluator":
        """初始化 RAGEngine（可注入）与 LLM-as-judge 函数。"""
        if self.engine is None:
            from src.rag_engine import RAGEngine
            self.engine = RAGEngine()
            await self.engine.initialize()
        if self.enable_llm_judge and self._judge_llm_func is None:
            from src.llm_client import get_llm_client
            judge_llm = get_llm_client()

            async def _judge_call(prompt: str, system_prompt: str) -> str:
                return await judge_llm.llm_model_func(prompt, system_prompt)

            self._judge_llm_func = _judge_call
        return self

    def _get_retriever(self):
        if self.retriever is None:
            from src.neo4j_graph import Neo4jGraphRetriever
            self.retriever = Neo4jGraphRetriever()
        return self.retriever

    def _set_query_cache(self, enabled: bool) -> None:
        """运行时切换 LightRAG 查询缓存（不改源码）。

        背景：LightRAG 查询缓存键（operate.py compute_args_hash）不含 system_prompt，
        若沿用缓存，neo4j 证据的 generation 会命中 baseline 的旧回答 → 证据被吞。
        --fresh 时在 generation 前后临时禁用，保证证据真正进入 LLM。
        retrieve()（M1-M4 数据）保持缓存 → 检索指标确定性不受影响。
        """
        rag = getattr(self.engine, "rag", None)
        kv = getattr(rag, "llm_response_cache", None) if rag is not None else None
        gc = getattr(kv, "global_config", None) if kv is not None else None
        if gc is not None:
            gc["enable_llm_cache"] = bool(enabled)

    # ----------------------------------------------------------
    # 单题评估（镜像 runner，唯一变量 = system_prompt 是否含 Neo4j 证据）
    # ----------------------------------------------------------

    async def _evaluate_one(self, q: dict, with_neo4j: bool) -> dict:
        question_id = q.get("id", "?")
        question_text = q.get("question", "")
        expected_entities = q.get("expected_entities", [])
        key_facts = q.get("key_facts", [])
        expected_refusal = q.get("expected_refusal", False)
        category = q.get("category", "")

        record = {
            "question_id": question_id,
            "question": question_text[:200],
            "category": category,
            "expected_refusal": expected_refusal,
            "safety_level": q.get("safety_level", "caution"),  # 与 runner 一致：数据集 GT safety_level
            "with_neo4j": bool(with_neo4j),
            "status": "pending",
            "error": None,
            "llm_response": "",
            "actual_refusal": None,
            "metrics": {},
            "reliability_decision": None,
            "duration_ms": 0,
        }
        t_start = time.time()
        try:
            # ---- Step A: retrieve()（M1-M4 数据，缓存命中）----
            retrieve_result = await self.engine.retrieve(
                question_text, mode=self.mode, top_k=self.top_k, chunk_top_k=self.chunk_top_k,
            )

            # ---- Safety（与 runner B2 一致）----
            safety_check = self.safety_checker.check(question_text)
            record["safety_check_level"] = safety_check.level.value  # 额外记录 checker 结果

            # ---- Neo4j 证据（仅 with_neo4j）----
            neo4j_result = None
            mentions: list[str] = []
            if with_neo4j:
                mentions = extract_curated_mentions(question_text, self.name_map)
                record["mentions"] = mentions
                neo4j_result = _retrieve_neo4j(mentions, retriever=self._get_retriever())
                record["neo4j_unavailable"] = bool(neo4j_result.get("unavailable"))

            # ---- system_prompt（唯一实验变量）----
            system_prompt = select_system_prompt(
                safety_check,
                with_neo4j=with_neo4j,
                neo4j_result=neo4j_result,
                config=self.config,
            )

            # ---- Step B: generation（BLOCKED 0 次 generation）----
            if safety_check.level == SafetyLevel.BLOCKED:
                llm_response = safety_check.rejection_message
                query_data: dict = {}
            else:
                if self.fresh:
                    self._set_query_cache(False)
                try:
                    qresult = await self.engine.query_with_context(
                        question_text,
                        mode=self.mode,
                        system_prompt=system_prompt,
                        top_k=self.top_k,
                        chunk_top_k=self.chunk_top_k,
                    )
                finally:
                    if self.fresh:
                        self._set_query_cache(True)
                llm_response = (qresult or {}).get("llm_response") or ""
                query_data = (qresult or {}).get("data") or {}
            record["llm_response"] = llm_response[:2000]

            # ---- Step C: 检索数据（与 runner 一致，用 retrieve()）----
            retrieve_data = _safe_get_data(retrieve_result)
            retrieved_entities = _safe_get_list(retrieve_data, "entities")
            retrieved_chunks = _safe_get_list(retrieve_data, "chunks")

            # ---- Step D: M1-M6（复用 metrics，逐行一致）----
            metrics = await self._compute_metrics(
                question_text, expected_entities, key_facts,
                llm_response, retrieved_entities, retrieved_chunks,
            )
            record["metrics"] = metrics

            # ---- E1: detect_refusal ----
            refusal_result = detect_refusal(llm_response, safety_check_result=safety_check)

            # ---- E2: citation ----
            try:
                citation_result = analyze_citations(
                    answer=llm_response,
                    retrieved_entities=retrieved_entities,
                    retrieved_chunks=retrieved_chunks,
                    entity_name_map=self.name_map,
                )
                record["citation_analysis"] = asdict(citation_result)
            except Exception as exc:
                record["citation_analysis"] = {
                    "error": type(exc).__name__, "message": str(exc)[:300],
                }

            # ---- E3: uncertainty ----
            try:
                uncertainty_result = extract_signals(
                    question=question_text,
                    expected_entities=expected_entities,
                    key_facts=key_facts,
                    retrieved_entities=retrieved_entities,
                    retrieved_chunks=retrieved_chunks,
                    retrieved_relationships=_safe_get_list(retrieve_data, "relationships"),
                    safety_check_result=safety_check,
                    refusal_result=refusal_result,
                    citation_analysis=record.get("citation_analysis"),
                    metrics=metrics,
                    entity_name_map=self.name_map,
                    answer=llm_response,
                )
                record["uncertainty_analysis"] = uncertainty_result.to_dict()
            except Exception as exc:
                record["uncertainty_analysis"] = {
                    "error": type(exc).__name__, "message": str(exc)[:300],
                }

            # ---- E4: M8-M12 ----
            self._compute_m8_m12(metrics, record, safety_check)

            # ---- E5: ReliabilityDecision ----
            try:
                decision = make_reliability_decision(
                    question=question_text,
                    safety_check=safety_check,
                    uncertainty_analysis=record.get("uncertainty_analysis"),
                    citation_analysis=record.get("citation_analysis"),
                    m8=metrics.get("M8"),
                    m9=metrics.get("M9"),
                    m10=metrics.get("M10"),
                    m12=metrics.get("M12"),
                    refusal_result=refusal_result,
                    answer=llm_response,
                )
                record["reliability_decision"] = decision.to_dict()
                record["actual_refusal"] = decision.is_final_refusal
            except Exception as exc:
                record["reliability_decision"] = {
                    "error": type(exc).__name__, "message": str(exc)[:300],
                }
                record["actual_refusal"] = refusal_result.get("is_refusal", False)

            # ---- Neo4j 融合 provenance（仅 with_neo4j）----
            if with_neo4j:
                fused = fuse_evidence(
                    lightrag_data=query_data,
                    neo4j_data=neo4j_result,
                    entity_name_map=self.name_map,
                    config=self.config,
                )
                record["neo4j_provenance"] = {
                    "mentions": mentions,
                    "evidence_sources": fused.get("evidence_sources"),
                    "neo4j_has_evidence": fused.get("coverage", {}).get("neo4j_has_evidence"),
                    "coverage_notes": fused.get("coverage", {}).get("notes"),
                }

            record["status"] = "success"
        except Exception as exc:
            record["status"] = "error"
            record["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
            logger.error(f"[{question_id}] 评估失败: {type(exc).__name__}: {exc}")

        record["duration_ms"] = round((time.time() - t_start) * 1000)
        return record

    # ----------------------------------------------------------
    # 指标计算（镜像 runner Step D + E4）
    # ----------------------------------------------------------

    async def _compute_metrics(
        self, question_text, expected_entities, key_facts,
        llm_response, retrieved_entities, retrieved_chunks,
    ) -> dict:
        metrics: dict = {}

        if expected_entities:
            metrics["M1"] = retrieval_entity_recall(expected_entities, retrieved_entities, self.name_map)
        else:
            metrics["M1"] = {"score": None, "note": "expected_entities 为空，不适用"}

        if key_facts:
            metrics["M2"] = retrieval_chunk_relevance(key_facts, retrieved_chunks)
        else:
            metrics["M2"] = {"score": None, "note": "key_facts 为空，不适用"}

        if expected_entities and llm_response:
            metrics["M3"] = answer_entity_recall(expected_entities, llm_response, self.name_map)
        elif not expected_entities:
            metrics["M3"] = {"score": None, "note": "expected_entities 为空，不适用"}
        else:
            metrics["M3"] = {"score": 0.0, "note": "llm_response 为空"}

        metrics["M4"] = context_precision(retrieved_entities, expected_entities, self.name_map)

        if llm_response:
            m5_heuristic = faithfulness(llm_response, retrieved_chunks)
            m5_llm_judge = await faithfulness_llm_judge(
                question=question_text,
                answer_text=llm_response,
                retrieved_chunks=retrieved_chunks,
                llm_func=self._judge_llm_func,
            )
            metrics["M5"] = {"heuristic": m5_heuristic, "llm_judge": m5_llm_judge}
        else:
            metrics["M5"] = {
                "heuristic": {"score": None, "note": "llm_response 为空"},
                "llm_judge": {
                    "score": None, "supported_claims": [], "unsupported_claims": [],
                    "notes": [], "reason": "llm_response 为空", "error": None,
                    "parse_failed": False, "raw_response": None,
                },
            }

        if key_facts and llm_response:
            metrics["M6"] = citation_coverage(
                key_facts, retrieved_chunks, llm_response, retrieved_entities=retrieved_entities,
            )
        elif not key_facts:
            metrics["M6"] = {
                "potential": {"score": None, "note": "key_facts 为空"},
                "actual": {"score": None, "note": "key_facts 为空"},
                "gap": [], "citation_present": False, "citation_grounded": False,
            }
        else:
            metrics["M6"] = {
                "potential": {"score": None, "note": "llm_response 为空"},
                "actual": {"score": None, "note": "llm_response 为空"},
                "gap": [], "citation_present": False, "citation_grounded": False,
            }

        return metrics

    def _compute_m8_m12(self, metrics: dict, record: dict, safety_check) -> None:
        try:
            metrics["M8"] = uncertainty_score_metric(record.get("uncertainty_analysis"))
        except Exception as exc:
            metrics["M8"] = {"score": None, "note": f"M8 计算失败: {type(exc).__name__}: {str(exc)[:200]}"}
        try:
            metrics["M9"] = citation_quality(record.get("citation_analysis"))
        except Exception as exc:
            metrics["M9"] = {"score": None, "note": f"M9 计算失败: {type(exc).__name__}: {str(exc)[:200]}"}
        try:
            metrics["M10"] = safety_signal_presence(safety_check, record.get("uncertainty_analysis"))
        except Exception as exc:
            metrics["M10"] = {"score": None, "note": f"M10 计算失败: {type(exc).__name__}: {str(exc)[:200]}"}
        try:
            metrics["M12"] = combined_reliability(
                metrics.get("M8"), metrics.get("M9"), metrics.get("M10"),
            )
        except Exception as exc:
            metrics["M12"] = {"score": None, "note": f"M12 计算失败: {type(exc).__name__}: {str(exc)[:200]}"}

    # ----------------------------------------------------------
    # 全量执行
    # ----------------------------------------------------------

    async def run_mode(self, questions: list[dict], with_neo4j: bool) -> list[dict]:
        label = "neo4j" if with_neo4j else "baseline"
        logger.info(f"开始 {label} 模式评估 {len(questions)} 题...")
        records = []
        for i, q in enumerate(questions):
            qid = q.get("id", "?")
            logger.info(f"[{i+1}/{len(questions)}] {qid} ({label})")
            rec = await self._evaluate_one(q, with_neo4j)
            records.append(rec)
        success = sum(1 for r in records if r["status"] == "success")
        logger.info(f"{label} 完成: {success}/{len(records)} 成功")
        return records


# ============================================================
# CLI
# ============================================================

def _load_questions() -> list[dict]:
    path = _PROJECT_ROOT / "evaluation" / "datasets" / "test_questions.json"
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset.get("questions", [])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 F6 — 端到端对比评估（baseline vs Neo4j 证据融合）",
    )
    parser.add_argument("--mode", choices=["baseline", "neo4j", "both"], default="both",
                        help="要运行的模式：baseline（无 Neo4j）/ neo4j（有 Neo4j）/ both（默认，两者）")
    parser.add_argument("--llm-judge", action="store_true", dest="llm_judge",
                        help="启用 M5 LLM-as-judge（默认关闭）")
    parser.add_argument("--no-llm-judge", action="store_true", dest="no_llm_judge",
                        help="禁用 M5 LLM-as-judge")
    parser.add_argument("--fresh", action="store_true",
                        help="禁用 LightRAG 查询缓存（正式对比必须，否则 neo4j 证据会被缓存吞掉）")
    parser.add_argument("--output", default="data/eval/phase6_report.json",
                        help="报告输出路径")
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--chunk-top-k", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    enable_llm_judge = bool(args.llm_judge and not args.no_llm_judge)

    questions = _load_questions()
    logger.info(f"加载 {len(questions)} 条测试问题")

    async def _run() -> dict:
        evaluator = Phase6Evaluator(
            mode="mix",
            top_k=args.top_k,
            chunk_top_k=args.chunk_top_k,
            enable_llm_judge=enable_llm_judge,
            fresh=bool(args.fresh),
        )
        await evaluator.initialize()

        baseline_records = None
        neo4j_records = None
        if args.mode in ("baseline", "both"):
            baseline_records = await evaluator.run_mode(questions, with_neo4j=False)
        if args.mode in ("neo4j", "both"):
            neo4j_records = await evaluator.run_mode(questions, with_neo4j=True)

        if args.mode == "both":
            report = build_report(questions, baseline_records, neo4j_records)
        else:
            records = baseline_records if args.mode == "baseline" else neo4j_records
            report = {
                "phase": "6F6",
                "dataset_size": len(questions),
                "execution": {args.mode: {
                    "success": sum(1 for r in records if r["status"] == "success"),
                    "failed": sum(1 for r in records if r["status"] != "success"),
                    "failed_details": [{"question_id": r.get("question_id"), "error": r.get("error")}
                                       for r in records if r["status"] != "success"],
                }},
                f"{'baseline_without_neo4j' if args.mode == 'baseline' else 'with_neo4j'}": {
                    "metrics": aggregate_metrics(records),
                    "decision_summary": aggregate_decisions(records),
                    "m7": aggregate_m7(records),
                },
            }
        return report

    report = asyncio.run(_run())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info(f"报告已写入: {out_path}")


if __name__ == "__main__":
    main()
