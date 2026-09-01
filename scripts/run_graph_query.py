#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 6 F5 — run_graph_query.py：证据融合查询编排入口。

流程（严格顺序，一次用户问题最多一次 LLM generation）:

    question
      → SafetyChecker.check(question)                          # 复用 B2
      → extract_curated_mentions(question, name_map)           # 零 LLM，D4 确定性识别
      → Neo4jGraphRetriever（每 mention: get_entity + get_neighbors both）  # 纯 Cypher，零 LLM
      → build_fusion_prompt(base safety prompt + Neo4j 证据段) # 纯函数
      → RAGEngine.query_with_context(question, system_prompt=…)  # 唯一一次 LLM generation
      → fuse_evidence(lightrag_data + neo4j_data)              # 完整融合上下文
      → 复用信号链路: detect_refusal / analyze_citations / extract_signals
        / metrics M8-M12 / make_reliability_decision           # 全部复用，不修改
      → 结构化 JSON 输出

约束（对齐 F5 需求）:
    - 不修改 EvaluationRunner / 评估指标 / ReliabilityDecision / 任何现有模块
    - 不增加额外 LLM judge / reranker / entity linker
    - Neo4j 容器不可达 → 降级为纯 LightRAG 路径，记录 neo4j_unavailable，不崩溃
    - Neo4j 证据只作为 context，不直接改变 ReliabilityDecision

用法:
    python scripts/run_graph_query.py -q "二甲双胍与2型糖尿病有什么关系？"
    python scripts/run_graph_query.py -q "高血压怎么治疗？" --output data/eval/f5_smoke_htn.json
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataclasses import asdict

from src.medical_core.safety import SafetyChecker, SafetyLevel, SYSTEM_PROMPT_SAFE
from src.rag_engine import RAGEngine, get_rag_engine
from src.neo4j_graph import Neo4jGraphRetriever
from src.evidence_fusion import (
    FusionConfig,
    NEO4J_UNAVAILABLE_KEY,
    build_fusion_prompt,
    extract_curated_mentions,
    fuse_evidence,
)
from evaluation.metrics import (
    load_entity_name_map,
    uncertainty_score_metric,
    citation_quality,
    safety_signal_presence,
    combined_reliability,
)
from evaluation.runner import detect_refusal
from evaluation.citation_verifier import analyze_citations
from evaluation.uncertainty_signals import extract_signals
from evaluation.reliability_decision import make_reliability_decision

logger = logging.getLogger(__name__)


# ============================================================
# 辅助
# ============================================================

def _safety_dict(safety) -> dict:
    return {
        "level": getattr(safety.level, "value", str(safety.level)),
        "reason": getattr(safety, "reason", ""),
        "disclaimer_needed": bool(getattr(safety, "disclaimer_needed", False)),
        "is_restricted": bool(getattr(safety, "is_restricted", False)),
    }


def _retrieve_neo4j(mentions: list, retriever=None) -> dict:
    """对每个 curated mention 做 Neo4j 只读检索（get_entity + get_neighbors both）。

    - 无 mention → 空结果 + 说明（不伪造）
    - Neo4j 不可达 → {"unavailable": True, error, ...}（调用方降级）
    """
    if not mentions:
        return {
            "nodes": [], "edges": [], "triples": [],
            "note": "问题未命中 curated 实体，Neo4j 检索返回空。",
        }
    r = retriever if retriever is not None else Neo4jGraphRetriever()
    all_nodes: list = []
    all_edges: list = []
    all_triples: list = []
    seen_node_ids: set = set()
    seen_triple_keys: set = set()

    try:
        for mention in mentions:
            g = r.get_entity(mention)
            for n in (g.get("nodes") or []):
                nid = n.get("id")
                if nid not in seen_node_ids:
                    seen_node_ids.add(nid)
                    all_nodes.append(n)

            ng = r.get_neighbors(mention, direction="both", limit=50)
            for e in (ng.get("edges") or []):
                key = (e.get("source"), e.get("relation_type"), e.get("target"))
                if key in seen_triple_keys:
                    continue
                seen_triple_keys.add(key)
                all_edges.append(e)
                all_triples.append(list(key))
            for n in (ng.get("nodes") or []):
                nid = n.get("id")
                if nid not in seen_node_ids:
                    seen_node_ids.add(nid)
                    all_nodes.append(n)
        return {"nodes": all_nodes, "edges": all_edges, "triples": all_triples}
    except Exception as exc:
        return {
            NEO4J_UNAVAILABLE_KEY: True,
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "nodes": [], "edges": [], "triples": [],
        }


def _base_safety_prompt(safety) -> str | None:
    """与 runner B2 分层一致：CAUTION/RESTRICTED 用安全 prompt；SAFE/BLOCKED 用 None。"""
    if safety.level == SafetyLevel.RESTRICTED:
        return safety.modified_prompt or SYSTEM_PROMPT_SAFE
    if safety.level == SafetyLevel.CAUTION:
        return SYSTEM_PROMPT_SAFE
    return None


# ============================================================
# 主编排
# ============================================================

async def run_fusion_query(
    question: str,
    mode: str = "mix",
    engine: RAGEngine | None = None,
    retriever=None,
    name_map: dict | None = None,
    safety_checker: SafetyChecker | None = None,
    config: FusionConfig | None = None,
) -> dict:
    """证据融合查询编排（可注入 engine/retriever 便于测试）。

    Returns:
        结构化 JSON dict（见模块 docstring）。
    """
    if not question or not isinstance(question, str):
        raise ValueError("question 不能为空")

    result: dict = {"question": question, "mode": mode}

    sc = safety_checker or SafetyChecker()
    safety = sc.check(question)
    result["safety"] = _safety_dict(safety)

    nm = name_map if isinstance(name_map, dict) else load_entity_name_map()
    mentions = extract_curated_mentions(question, nm)
    result["mentions"] = mentions

    neo4j_result = _retrieve_neo4j(mentions, retriever=retriever)
    result["neo4j_unavailable"] = bool(neo4j_result.get(NEO4J_UNAVAILABLE_KEY))

    fusion_prompt = build_fusion_prompt(
        neo4j_result,
        base_prompt=_base_safety_prompt(safety),
        config=config,
    )
    result["system_prompt"] = fusion_prompt

    llm_generation_count = 0
    if safety.level == SafetyLevel.BLOCKED:
        # 高危问题直接拒答，不调用 LLM（与 runner B2 一致）
        llm_response = safety.rejection_message
        lr_data: dict = {"entities": [], "relationships": [], "chunks": []}
    else:
        eng = engine if engine is not None else await get_rag_engine()
        qresult = await eng.query_with_context(
            question,
            mode=mode,
            system_prompt=fusion_prompt,
        )
        llm_generation_count = 1
        llm_response = (qresult or {}).get("llm_response") or ""
        lr_data = (qresult or {}).get("data") or {}

    result["llm_generation_count"] = llm_generation_count
    result["llm_response"] = llm_response

    # ---- 融合（完整上下文，供审计 / F6 使用）----
    fused = fuse_evidence(
        lightrag_data=lr_data,
        neo4j_data=neo4j_result,
        entity_name_map=nm,
        config=config,
    )
    result["fused_context"] = fused

    # ---- 信号链路（全部复用，不修改）----
    refusal = detect_refusal(llm_response, safety)
    result["refusal"] = refusal

    try:
        citation = analyze_citations(
            answer=llm_response,
            retrieved_entities=lr_data.get("entities") or [],
            retrieved_chunks=lr_data.get("chunks") or [],
            entity_name_map=nm,
        )
        citation_dict = asdict(citation)
    except Exception as exc:
        citation_dict = {"error": type(exc).__name__,
                         "message": str(exc)[:200]}
    result["citation"] = citation_dict

    try:
        uncertainty = extract_signals(
            question=question,
            retrieved_entities=lr_data.get("entities") or [],
            retrieved_chunks=lr_data.get("chunks") or [],
            retrieved_relationships=lr_data.get("relationships") or [],
            safety_check_result=safety,
            refusal_result=refusal,
            citation_analysis=citation_dict,
            metrics=None,
            entity_name_map=nm,
            answer=llm_response,
        )
        uncertainty_dict = uncertainty.to_dict()
    except Exception as exc:
        uncertainty_dict = {"error": type(exc).__name__,
                            "message": str(exc)[:200]}
    result["uncertainty"] = uncertainty_dict

    m8 = uncertainty_score_metric(uncertainty_dict)
    m9 = citation_quality(citation_dict)
    m10 = safety_signal_presence(safety, uncertainty_dict)
    m12 = combined_reliability(m8, m9, m10)
    result["metrics"] = {"M8": m8, "M9": m9, "M10": m10, "M12": m12}

    decision = make_reliability_decision(
        question=question,
        safety_check=safety,
        uncertainty_analysis=uncertainty_dict,
        citation_analysis=citation_dict,
        m8=m8,
        m9=m9,
        m10=m10,
        m12=m12,
        refusal_result=refusal,
        answer=llm_response,
    )
    result["reliability_decision"] = decision.to_dict()

    return result


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 F5 — 证据融合查询编排（LightRAG + Neo4j）",
    )
    parser.add_argument("-q", "--question", required=True, help="用户问题")
    parser.add_argument("--mode", default="mix",
                        help="LightRAG 检索模式（mix/hybrid/local/global/naive）")
    parser.add_argument("--output", default=None,
                        help="可选：将结果 JSON 写入该文件")
    parser.add_argument("--verbose", action="store_true", help="打印详细日志")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    result = asyncio.run(run_fusion_query(args.question, mode=args.mode))

    out_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out_text, encoding="utf-8")
    print(out_text)


if __name__ == "__main__":
    main()
