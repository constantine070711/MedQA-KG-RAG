# -*- coding: utf-8 -*-
"""Phase 10-B B2 — 旁路检索级 Reranker Benchmark（Baseline vs Reranker）。

设计（docs/phase10_b_reranker_design_audit.md §9）：
  - 只做 retrieval-level benchmark，不调用 LLM。
  - Baseline:  engine.retrieve(question, mode=mix, top_k=60, chunk_top_k=20)   # 当前默认
  - Experiment: engine.retrieve(question, mode=mix, top_k=60, chunk_top_k=100)  # 提高候选召回
               → rerank_chunks(question, chunks, top_m=14)                      # cross-encoder 重排
  - 指标：M1（实体召回，对照组）/ M2（chunk 相关）/ MRR（chunk 排序）/ latency。
  - Neo4j 边界：本 benchmark 只用 engine.retrieve()（纯 LightRAG），Neo4j canonical
    evidence 不进入 reranker、不参与本 benchmark（检索级）。
  - 输出独立 data/eval/reranker_benchmark/，不污染生产 cache（retrieve 只读）。

参数（对齐 Design Audit §6）：
    retrieval_top_k=60 / chunk_top_k_baseline=20 / chunk_top_k_experiment=100 /
    final_chunks=14（如实际代码约束需调整，如实记录原因，不擅自扩大范围）。

用法：
    python scripts/reranker_benchmark.py                # 需 bge-reranker-base 已下载
    python scripts/reranker_benchmark.py --no-rerank    # 仅 Baseline（对照）
    python scripts/reranker_benchmark.py --model models/bge-reranker-base
"""

import argparse
import asyncio
import io
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_engine import RAGEngine  # noqa: E402
from src.retrieval.rerank import Reranker, rerank_chunks  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    _text_contains_key_terms,
    load_entity_name_map,
    retrieval_chunk_relevance,
    retrieval_entity_recall,
)

# ---- 参数（对齐 Design Audit §6）----
RETRIEVAL_TOP_K = 60          # 实体/关系检索（Baseline 与 Experiment 相同）
CHUNK_TOP_K_BASELINE = 20     # Baseline chunk 召回（当前默认）
CHUNK_TOP_K_EXPERIMENT = 100  # Experiment chunk 召回（提高候选范围）
FINAL_CHUNKS = 14             # rerank 后进入最终 context 的 chunk 数（对齐 Fusion max_chunks=14）

PARAMS = {
    "retrieval_top_k": RETRIEVAL_TOP_K,
    "chunk_top_k_baseline": CHUNK_TOP_K_BASELINE,
    "chunk_top_k_experiment": CHUNK_TOP_K_EXPERIMENT,
    "final_chunks": FINAL_CHUNKS,
    "mode": "mix",
}


# ============================================================
# MRR（chunk 排序质量）
# ============================================================

def _mrr(chunks: list[dict], key_facts: list[str]) -> float:
    """查询级 MRR：第一个命中任一 key_fact 的 chunk 的位置倒数；无命中为 0。

    与 M2 使用同一匹配函数（_text_contains_key_terms）保证口径一致。
    """
    if not key_facts:
        return 1.0
    for i, c in enumerate(chunks, 1):
        content = str(c.get("content", "") or "")
        if any(_text_contains_key_terms(content, f) for f in key_facts):
            return round(1.0 / i, 4)
    return 0.0


# ============================================================
# 单题评估
# ============================================================

async def _evaluate_one(engine, q: dict, name_map: dict, reranker) -> dict:
    question = q["question"]
    exp_entities = q.get("expected_entities") or []
    key_facts = q.get("key_facts") or []

    # ---- Baseline ----
    t0 = time.perf_counter()
    base = await engine.retrieve(question, mode="mix",
                                 top_k=RETRIEVAL_TOP_K, chunk_top_k=CHUNK_TOP_K_BASELINE)
    t_base = time.perf_counter() - t0
    base_data = base.get("data") or {}
    base_chunks = base_data.get("chunks") or []
    base_entities = base_data.get("entities") or []

    # ---- Experiment（提高候选召回 + rerank）----
    t1 = time.perf_counter()
    exp = await engine.retrieve(question, mode="mix",
                                top_k=RETRIEVAL_TOP_K, chunk_top_k=CHUNK_TOP_K_EXPERIMENT)
    exp_data = exp.get("data") or {}
    exp_chunks_raw = exp_data.get("chunks") or []
    t_retrieve_exp = time.perf_counter() - t1

    # rerank（旁路；reranker 不可用则内部回退原始顺序 → 语义等同 Baseline 截取）
    t2 = time.perf_counter()
    exp_chunks = rerank_chunks(question, exp_chunks_raw, top_m=FINAL_CHUNKS, reranker=reranker)
    t_rerank = time.perf_counter() - t2

    # ---- 指标 ----
    m1 = retrieval_entity_recall(exp_entities, base_entities, name_map)
    m2_base = retrieval_chunk_relevance(key_facts, base_chunks)
    m2_exp = retrieval_chunk_relevance(key_facts, exp_chunks)
    mrr_base = _mrr(base_chunks, key_facts)
    mrr_exp = _mrr(exp_chunks, key_facts)

    return {
        "qid": q.get("id"),
        "category": q.get("category"),
        "question": question,
        "has_expected_entities": bool(exp_entities),
        "has_key_facts": bool(key_facts),
        "candidate_counts": {
            "baseline_chunks": len(base_chunks),
            "experiment_chunks_raw": len(exp_chunks_raw),
            "experiment_chunks_final": len(exp_chunks),
        },
        "M1": m1,
        "M2_baseline": m2_base,
        "M2_experiment": m2_exp,
        "MRR_baseline": mrr_base,
        "MRR_experiment": mrr_exp,
        "latency_s": {
            "baseline_retrieve": round(t_base, 4),
            "experiment_retrieve": round(t_retrieve_exp, 4),
            "experiment_rerank": round(t_rerank, 4),
            "experiment_total": round(t_retrieve_exp + t_rerank, 4),
        },
    }


def _mean(nums: list[float]) -> Optional[float]:
    return round(sum(nums) / len(nums), 4) if nums else None


def _agg(results: list[dict]) -> dict:
    def _col(rec, path):
        v = rec
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        return v

    def _means(key_facts_only=False):
        m1, m2b, m2e, mrrb, mrre = [], [], [], [], []
        for r in results:
            if key_facts_only and not r["has_key_facts"]:
                continue
            m1.append(r["M1"]["score"])
            m2b.append(r["M2_baseline"]["score"])
            m2e.append(r["M2_experiment"]["score"])
            mrrb.append(r["MRR_baseline"])
            mrre.append(r["MRR_experiment"])
        return {
            "n": len(m1),
            "M1": _mean(m1),
            "M2_baseline": _mean(m2b),
            "M2_experiment": _mean(m2e),
            "MRR_baseline": _mean(mrrb),
            "MRR_experiment": _mean(mrre),
        }

    lat_base = [r["latency_s"]["baseline_retrieve"] for r in results]
    lat_exp_total = [r["latency_s"]["experiment_total"] for r in results]
    lat_rerank = [r["latency_s"]["experiment_rerank"] for r in results]

    return {
        "all_28": _means(),
        "grounded_only_has_key_facts": _means(key_facts_only=True),
        "latency_s": {
            "baseline_retrieve_mean": _mean(lat_base),
            "experiment_total_mean": _mean(lat_exp_total),
            "experiment_rerank_mean": _mean(lat_rerank),
            "rerank_overhead_mean": _mean([b for b in lat_rerank]),
        },
        "candidate_summary": {
            "baseline_chunks_avg": _mean([r["candidate_counts"]["baseline_chunks"] for r in results]),
            "experiment_chunks_raw_avg": _mean([r["candidate_counts"]["experiment_chunks_raw"] for r in results]),
            "experiment_chunks_final_avg": _mean([r["candidate_counts"]["experiment_chunks_final"] for r in results]),
        },
    }


# ============================================================
# 主流程
# ============================================================

async def _run(output_dir: Path, model_path: Optional[str], use_rerank: bool) -> dict:
    # 加载 28 题
    qdata = json.loads(Path("evaluation/datasets/test_questions.json").read_text(encoding="utf-8"))
    questions = qdata["questions"]
    name_map = load_entity_name_map()

    # Reranker（旁路；不可用则回退，不影响 Benchmark 运行）
    reranker = None
    rerank_state = "disabled"
    if use_rerank:
        reranker = Reranker(model_name_or_path=model_path)
        if reranker.is_available:
            rerank_state = "active"
        else:
            # 不可用 → 记录状态；仍传入实例（is_available=False → rerank_chunks 立即 fallback，
            # 不再触发重复模型加载/网络超时）
            rerank_state = f"fallback({reranker.load_error})"

    engine = RAGEngine()
    await engine.initialize()

    results = []
    for i, q in enumerate(questions, 1):
        try:
            rec = await _evaluate_one(engine, q, name_map, reranker)
            results.append(rec)
            print(f"  [{i}/{len(questions)}] {q.get('id')} {q['category']} "
                  f"chunks_b={rec['candidate_counts']['baseline_chunks']} "
                  f"chunks_e={rec['candidate_counts']['experiment_chunks_final']} "
                  f"M2: {rec['M2_baseline']['score']}→{rec['M2_experiment']['score']} "
                  f"MRR: {rec['MRR_baseline']}→{rec['MRR_experiment']}")
        except Exception as exc:
            results.append({
                "qid": q.get("id"), "category": q.get("category"), "question": q.get("question"),
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            })
            print(f"  [{i}/{len(questions)}] {q.get('id')} ERROR {type(exc).__name__}")

    ok = [r for r in results if "error" not in r]
    agg = _agg(ok)

    # 模型版本
    model_version = "none"
    if use_rerank and reranker is not None:
        model_version = reranker.model_name_or_path
    elif use_rerank:
        model_version = rerank_state

    summary = {
        "meta": {
            "phase": "10-B-B2",
            "rerank_state": rerank_state,
            "model": model_version,
            "use_rerank": use_rerank,
            "params": PARAMS,
            "seed": None,  # 本实验无随机组件（确定性指标）
            "n_questions": len(questions),
            "n_success": len(ok),
            "n_error": len(questions) - len(ok),
            "runtime_s": round(sum(r["latency_s"]["baseline_retrieve"] + r["latency_s"]["experiment_total"]
                                   for r in ok), 2),
        },
        "aggregate": agg,
        "per_question": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "per_question"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 10-B B2 旁路检索级 Reranker Benchmark")
    parser.add_argument("--output", default="data/eval/reranker_benchmark",
                        help="输出目录（默认 data/eval/reranker_benchmark）")
    parser.add_argument("--model", default=None,
                        help="reranker 模型路径/HF 名（默认本地 models/bge-reranker-base → BAAI/bge-reranker-base）")
    parser.add_argument("--no-rerank", action="store_true",
                        help="仅 Baseline（对照）")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = Path(args.output) if Path(args.output).is_absolute() else root / args.output
    use_rerank = not args.no_rerank

    print(f"[RERANK-BENCH] params={PARAMS} use_rerank={use_rerank} output={out}")
    summary = asyncio.run(_run(out, args.model, use_rerank))

    print()
    print("=== SUMMARY ===")
    m = summary["meta"]
    print(f"  rerank_state: {m['rerank_state']} | model: {m['model']}")
    print(f"  n_success: {m['n_success']}/{m['n_questions']} | errors: {m['n_error']}")
    a = summary["aggregate"]
    print(f"  [all_28]     M1={a['all_28']['M1']} M2={a['all_28']['M2_baseline']}→{a['all_28']['M2_experiment']} "
          f"MRR={a['all_28']['MRR_baseline']}→{a['all_28']['MRR_experiment']}")
    g = a["grounded_only_has_key_facts"]
    print(f"  [grounded]   n={g['n']} M2={g['M2_baseline']}→{g['M2_experiment']} "
          f"MRR={g['MRR_baseline']}→{g['MRR_experiment']}")
    print(f"  latency_s: baseline_retrieve={a['latency_s']['baseline_retrieve_mean']} "
          f"experiment_total={a['latency_s']['experiment_total_mean']} "
          f"rerank_overhead={a['latency_s']['experiment_rerank_mean']}")
    print(f"[OK] 产物已写入: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
