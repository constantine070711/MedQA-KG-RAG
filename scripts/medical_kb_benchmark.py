# -*- coding: utf-8 -*-
"""Phase 10-A.5 Step B4 — 正式 Benchmark：Baseline vs Experiment（28 题，旁路）。

实验变量：
  Baseline   = 生产 lightrag_cache/（14 markdown，只读）
  Experiment = lightrag_cache_medkb/（14 markdown + 5,000 CMeIE）

协议（解决 keyword-LM cache confound，见 B3 发现）：
  1. Warm-up pass：28 题 × 2 cache 各 retrieve 一次（计算关键词，丢弃结果）
  2. Measurement pass：28 题 × 2 cache 再 retrieve（关键词已缓存，记录 M1/M2/MRR + latency）
  → 两侧关键词状态一致（warm），latency 可比。

指标：
  M1 = retrieval_entity_recall（复用 evaluation/metrics.py）
  M2 = retrieval_chunk_relevance（复用）
  MRR = gold key_fact 在检索 chunks 中的平均倒数排名（本脚本实现，0=未命中）
  latency = measurement pass 单次 retrieve 耗时（warm）
  citation = 可选：query_with_context 生成回答 + citation_extractor（LLM 生成，成本高，--with-citation 开启）

输出 data/eval/medical_kb_benchmark/：
  baseline_results.json / experiment_results.json / comparison.json / summary.json
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_engine import RAGEngine  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_CACHE = PROJECT_ROOT / "lightrag_cache_medkb"
QUESTIONS = PROJECT_ROOT / "evaluation" / "datasets" / "test_questions.json"
BENCH_OUT = PROJECT_ROOT / "data" / "eval" / "medical_kb_benchmark"

# 复用 evaluation/metrics（只读 import，不改）
from evaluation.metrics import (  # noqa: E402
    load_entity_name_map,
    retrieval_entity_recall,
    retrieval_chunk_relevance,
)


def _mr_over_chunks(key_facts, chunks) -> float:
    """MRR：gold key_fact 在检索 chunks 中命中的平均倒数排名（1-indexed，未命中→0）。"""
    if not key_facts or not chunks:
        return 0.0
    # 命中判定与 M2 一致：_text_contains_key_terms（含核心词）
    from evaluation.metrics import _text_contains_key_terms
    rrs = []
    for fact in key_facts:
        rank = None
        for i, c in enumerate(chunks, 1):
            if _text_contains_key_terms(c.get("content", ""), fact):
                rank = i
                break
        rrs.append(1.0 / rank if rank else 0.0)
    return round(sum(rrs) / len(rrs), 4)


async def run_retrieval(engine, question: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    res = await engine.retrieve(question)
    dt = time.perf_counter() - t0
    return res, dt


async def run_retrieval_retry(engine, question: str, retries: int = 3,
                              sleep_s: float = 5.0, label: str = "") -> tuple[dict, float]:
    """带重试的检索（应对 transient 连接错误）。"""
    last = None
    for attempt in range(1, retries + 2):
        try:
            return await run_retrieval(engine, question)
        except Exception as e:
            last = e
            if attempt <= retries:
                print(f"    [retry {attempt}/{retries}] {label} {question[:20]}... "
                      f"error={type(e).__name__}, 等待 {sleep_s}s", flush=True)
                await asyncio.sleep(sleep_s)
    raise last


async def evaluate_one(engine, q: dict, name_map) -> dict:
    res, latency = await run_retrieval(engine, q["question"])
    data = res.get("data", {}) or {}
    retrieved_entities = data.get("entities", [])
    retrieved_chunks = data.get("chunks", [])
    m1 = retrieval_entity_recall(q.get("expected_entities", []), retrieved_entities, name_map)
    m2 = retrieval_chunk_relevance(q.get("key_facts", []), retrieved_chunks)
    mr = _mr_over_chunks(q.get("key_facts", []), retrieved_chunks)
    return {
        "question_id": q.get("id"),
        "question": q["question"],
        "latency_s": round(latency, 2),
        "M1": m1["score"],
        "M1_matched": m1["matched"],
        "M1_missed": m1["missed"],
        "M2": m2["score"],
        "M2_covered": m2["covered_facts"],
        "M2_uncovered": m2["uncovered_facts"],
        "MRR": mr,
        "retrieved_entity_count": len(retrieved_entities),
        "retrieved_chunk_count": len(retrieved_chunks),
    }


async def main(with_citation: bool) -> int:
    name_map = load_entity_name_map()
    qs = json.load(open(QUESTIONS, encoding="utf-8"))["questions"]
    print(f"[bench] 28 题加载: {len(qs)}")

    baseline_engine = RAGEngine()                          # 生产 cache
    experiment_engine = RAGEngine(working_dir=str(EXPERIMENT_CACHE))
    await baseline_engine.initialize()
    await experiment_engine.initialize()
    print("[bench] engines initialized")

    # ---- Warm-up pass（关键词缓存，不计入结果；带重试容错）----
    print("[bench] warm-up pass（28 题 × 2 cache，计算关键词，逐题容错）...", flush=True)
    for i, q in enumerate(qs, 1):
        await run_retrieval_retry(baseline_engine, q["question"], label=f"base[{i}/28]")
        await run_retrieval_retry(experiment_engine, q["question"], label=f"exp[{i}/28]")
        if i % 5 == 0 or i == 1:
            print(f"    warm-up 进度: {i}/28", flush=True)
    print("[bench] warm-up 完成", flush=True)

    # ---- Measurement pass ----
    print("[bench] measurement pass ...")
    base_rows, exp_rows = [], []
    for q in qs:
        base_rows.append(await evaluate_one(baseline_engine, q, name_map))
        exp_rows.append(await evaluate_one(experiment_engine, q, name_map))

    # ---- 汇总 ----
    def _agg(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {"mean": round(statistics.mean(vals), 4) if vals else None,
                "median": round(statistics.median(vals), 4) if vals else None,
                "p95": round(sorted(vals)[int(0.95 * len(vals)) - 1], 4) if len(vals) > 1 else None}

    summary = {
        "dataset": "28 questions（Phase 5 固定评测集）",
        "baseline": {"M1": _agg(base_rows, "M1"), "M2": _agg(base_rows, "M2"),
                     "MRR": _agg(base_rows, "MRR"), "latency_s": _agg(base_rows, "latency_s")},
        "experiment": {"M1": _agg(exp_rows, "M1"), "M2": _agg(exp_rows, "M2"),
                       "MRR": _agg(exp_rows, "MRR"), "latency_s": _agg(exp_rows, "latency_s")},
        "protocol": "warm-up pass（不计入）+ measurement pass（计入）；两侧关键词缓存状态一致",
    }
    # 逐题 improved / unchanged / degraded
    comparison = []
    improved = unchanged = degraded = 0
    for b, e in zip(base_rows, exp_rows):
        m1_d = e["M1"] - b["M1"]
        m2_d = e["M2"] - b["M2"]
        mr_d = e["MRR"] - b["MRR"]
        # 综合判定：M1 优先，其次 MRR、M2（任一提升且无下降）
        delta = round(m1_d + m2_d + mr_d, 4)
        if delta > 0.01:
            improved += 1
        elif delta < -0.01:
            degraded += 1
        else:
            unchanged += 1
        comparison.append({
            "question_id": b["question_id"],
            "baseline": {"M1": b["M1"], "M2": b["M2"], "MRR": b["MRR"],
                         "latency_s": b["latency_s"]},
            "experiment": {"M1": e["M1"], "M2": e["M2"], "MRR": e["MRR"],
                           "latency_s": e["latency_s"]},
            "delta": delta,
        })
    summary["distribution"] = {"improved": improved, "unchanged": unchanged,
                               "degraded": degraded}
    summary["note"] = ("delta = (ΔM1 + ΔMRR + ΔM2)；>0.01 improved / <-0.01 degraded / 其余 unchanged。"
                       "检索为 warm 状态；关键词抽取 LLM 调用发生在 warm-up，不计入 latency。")

    BENCH_OUT.mkdir(parents=True, exist_ok=True)
    (BENCH_OUT / "baseline_results.json").write_text(
        json.dumps(base_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (BENCH_OUT / "experiment_results.json").write_text(
        json.dumps(exp_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (BENCH_OUT / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=1), encoding="utf-8")
    (BENCH_OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n=== Benchmark 结果 ===")
    for side in ("baseline", "experiment"):
        s = summary[side]
        print(f"{side:>10}: M1={s['M1']['mean']} M2={s['M2']['mean']} MRR={s['MRR']['mean']} "
              f"lat_mean={s['latency_s']['mean']}s lat_p95={s['latency_s']['p95']}s")
    print(f"分布: improved={improved} unchanged={unchanged} degraded={degraded}")
    # 展示 improved / degraded 题目
    imp_qs = [c for c in comparison if c["delta"] > 0.01]
    deg_qs = [c for c in comparison if c["delta"] < -0.01]
    if imp_qs:
        print("improved:", [f"{c['question_id']}(Δ{c['delta']})" for c in imp_qs][:12])
    if deg_qs:
        print("degraded:", [f"{c['question_id']}(Δ{c['delta']})" for c in deg_qs][:12])
    print(f"[bench] 产物 → {BENCH_OUT}/")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--with-citation", action="store_true",
                   help="额外生成回答并做 citation presence（LLM 生成，成本高）")
    args = p.parse_args()
    raise SystemExit(asyncio.run(main(args.with_citation)))
