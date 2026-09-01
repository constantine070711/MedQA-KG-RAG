# -*- coding: utf-8 -*-
"""Phase 10-A.5 Step B3 — 100 条 CMeIE 正式小规模 ingest + Baseline/Experiment 小对照（旁路）。

实验变量：
  Baseline  = 生产 lightrag_cache/（14 markdown，只读）
  Experiment = lightrag_cache_medkb/ = 14 markdown（复用生产 chunk+图，只读拷贝）+ 100 条 CMeIE 结构化知识

零 LLM：整个 ingest 仅 ainsert_custom_kg（本地 embedding + 图/向量写入），不触发任何 LLM 抽取。
  14 markdown 的 chunks/entities/relationships 从生产 storage 只读复用（与生产逐字节一致，
  保证唯一实验变量 = CMeIE 增量）。

流程：
  1. 读生产 kv_store_text_chunks.json → 14 markdown chunks
  2. 读生产 graphml → 380 entities / 422 relationships
  3. medkb_custom_kg 构建 100 条 CMeIE custom_kg
  4. 合并（实体按名去重、markdown 优先；关系按 (src,tgt) 聚合多 predicate 防覆盖）
  5. ainsert_custom_kg → lightrag_cache_medkb（验证 ingest LLM 调用 = 0）
  6. 检索冒烟 + 小规模 Baseline vs Experiment 对照（同一组问题，retrieve() 两侧）
  7. 全量测试

禁止：改生产 cache / 改生产代码 / monkey-patch / 用 ainsert（触发 LLM）。
"""

import argparse
import asyncio
import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightrag import LightRAG, QueryParam  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

from src.config import get_config  # noqa: E402
from src.data_adapter.medkb_custom_kg import build_custom_kg_from_file  # noqa: E402
from src.embedder import get_embedder  # noqa: E402
from src.llm_client import get_llm_client  # noqa: E402
from src.rag_engine import RAGEngine  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_CACHE = PROJECT_ROOT / "lightrag_cache"
EXPERIMENT_CACHE = PROJECT_ROOT / "lightrag_cache_medkb"
TRAIN_SRC = PROJECT_ROOT / "data" / "external" / "medical_kb" / "CMeIE" / "CMeIE_train.jsonl"
BENCH_OUT = PROJECT_ROOT / "data" / "eval" / "medical_kb_benchmark"


class LLMCounter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    async def __call__(self, prompt, system_prompt=None, history_messages=None, **kw):
        self.count += 1
        return await self.func(prompt, system_prompt=system_prompt,
                               history_messages=history_messages, **kw)


# ============================================================
# 1. 读生产 markdown chunks（只读复用）
# ============================================================

def load_production_chunks() -> list[dict]:
    """读生产 kv_store_text_chunks.json → custom_kg chunks（source_id=生产 chunk_id）。"""
    d = json.load(open(PRODUCTION_CACHE / "kv_store_text_chunks.json", encoding="utf-8"))
    chunks = []
    for chunk_id, v in d.items():
        chunks.append({
            "content": v["content"],
            "source_id": chunk_id,           # 与生产一致，graphml 实体 source_id 可直接映射
            "file_path": v.get("file_path", "unknown_source"),
            "chunk_order_index": 0,
        })
    return chunks


# ============================================================
# 2. 读生产 graphml → 实体 + 关系（只读复用）
# ============================================================

def load_production_graph() -> tuple[list[dict], list[dict]]:
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    tree = ET.parse(PRODUCTION_CACHE / "graph_chunk_entity_relation.graphml")
    root = tree.getroot()

    entities = []
    for node in root.findall(".//g:node", ns):
        nid = node.get("id")
        attrs = {d.get("key"): d.text or "" for d in node.findall("g:data", ns)}
        entities.append({
            "entity_name": nid,
            "entity_type": attrs.get("d1", "UNKNOWN"),
            "description": attrs.get("d2", ""),
            "source_id": attrs.get("d3", "UNKNOWN"),
            "file_path": attrs.get("d4", "unknown_source"),
        })

    relationships = []
    for edge in root.findall(".//g:edge", ns):
        src = edge.get("source")
        tgt = edge.get("target")
        attrs = {d.get("key"): d.text or "" for d in edge.findall("g:data", ns)}
        relationships.append({
            "src_id": src,
            "tgt_id": tgt,
            "description": attrs.get("d8", ""),
            "keywords": attrs.get("d9", ""),
            "weight": float(attrs.get("d7", "1.0") or 1.0),
            "source_id": attrs.get("d10", "UNKNOWN"),
            "file_path": attrs.get("d11", "unknown_source"),
        })
    return entities, relationships


# ============================================================
# 4. 合并（多 predicate 防覆盖 + 实体按名去重）
# ============================================================

def merge_kg(md_chunks, md_entities, md_relations, cmeie):
    """合并 markdown（生产复用）与 CMeIE custom_kg。

    实体：按 entity_name 去重，markdown 优先（保持生产一致性）。
    关系：按 (src_id, tgt_id) 聚合，全部 predicate/relation_type 合并进
          keywords / description（防覆盖，不丢失）。
    """
    chunks = md_chunks + cmeie["chunks"]

    # 实体：markdown 优先（先放 CMeIE，后放 markdown 覆盖同名校，保持生产一致性）
    entities: OrderedDict[str, dict] = OrderedDict()
    for e in cmeie["entities"]:
        entities[e["entity_name"]] = e
    for e in md_entities:
        entities[e["entity_name"]] = e

    # 关系：按 (src,tgt) 聚合多 predicate
    pair: OrderedDict[tuple[str, str], dict] = OrderedDict()
    for r in md_relations + cmeie["relationships"]:
        key = (r["src_id"], r["tgt_id"])
        info = pair.setdefault(key, {"preds": set(), "kws": set(),
                                     "source_id": r.get("source_id", "UNKNOWN")})
        if r["description"]:
            for p in str(r["description"]).replace("；", ";").split(";"):
                p = p.strip()
                if p:
                    info["preds"].add(p)
        for kw in str(r["keywords"]).replace("，", ",").split(","):
            kw = kw.strip()
            if kw:
                info["kws"].add(kw)
    relationships = []
    multi = 0
    for (src, tgt), info in pair.items():
        if len(info["preds"]) > 1 or len(info["kws"]) > 1:
            multi += 1
        relationships.append({
            "src_id": src,
            "tgt_id": tgt,
            "description": "; ".join(sorted(info["preds"])) or "N/A",
            "keywords": ", ".join(sorted(info["kws"])) or "N/A",
            "weight": 1.0,
            "source_id": info["source_id"],
            "file_path": "merged",
        })
    return chunks, list(entities.values()), relationships, multi


# ============================================================
# 5. 小规模 Baseline vs Experiment 对照
# ============================================================

async def compare(baseline_engine, experiment_engine, questions: list[str]) -> list[dict]:
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        b = await baseline_engine.retrieve(q)
        b_dt = time.perf_counter() - t0
        t1 = time.perf_counter()
        e = await experiment_engine.retrieve(q)
        e_dt = time.perf_counter() - t1
        rows.append({
            "question": q,
            "baseline": {"entities": [x.get("entity_name") for x in b.get("data", {}).get("entities", [])],
                         "chunks": len(b.get("data", {}).get("chunks", [])),
                         "latency_s": round(b_dt, 2)},
            "experiment": {"entities": [x.get("entity_name") for x in e.get("data", {}).get("entities", [])],
                           "chunks": len(e.get("data", {}).get("chunks", [])),
                           "latency_s": round(e_dt, 2)},
        })
    return rows


async def main(limit: int) -> int:
    print(f"[B3] 实验：14 markdown（生产复用）+ {limit} 条 CMeIE → lightrag_cache_medkb")

    # 1-2. 生产 markdown（只读）
    md_chunks = load_production_chunks()
    md_entities, md_relations = load_production_graph()
    print(f"[B3] 生产 markdown 复用: chunks={len(md_chunks)} entities={len(md_entities)} "
          f"relationships={len(md_relations)}")

    # 3. CMeIE custom_kg
    cmeie = build_custom_kg_from_file(TRAIN_SRC, limit=limit)
    print(f"[B3] CMeIE custom_kg: records={cmeie['stats']['records']} "
          f"chunks={cmeie['stats']['chunks']} entities={cmeie['stats']['entities']} "
          f"relationships={cmeie['stats']['relationships']}")

    # 4. 合并
    chunks, entities, relationships, multi = merge_kg(
        md_chunks, md_entities, md_relations, cmeie)
    print(f"[B3] 合并后: chunks={len(chunks)} entities={len(entities)} "
          f"relationships={len(relationships)} 多predicate边={multi}")

    # 5. ainsert_custom_kg（零 LLM）
    import shutil
    if EXPERIMENT_CACHE.exists():
        shutil.rmtree(EXPERIMENT_CACHE)
    EXPERIMENT_CACHE.mkdir(parents=True, exist_ok=True)

    cfg = get_config()
    embedder = get_embedder()
    llm = get_llm_client()
    counter = LLMCounter(llm.llm_model_func)
    rag = LightRAG(
        working_dir=str(EXPERIMENT_CACHE),
        embedding_func=EmbeddingFunc(
            embedding_dim=embedder.dim,
            max_token_size=cfg.embedding_max_token_size,
            func=embedder.embedding_func,
        ),
        llm_model_func=counter,
    )

    async def _run():
        await rag.initialize_storages()
        t1 = time.perf_counter()
        await rag.ainsert_custom_kg({"chunks": chunks, "entities": entities,
                                     "relationships": relationships})
        ingest_time = time.perf_counter() - t1
        ingest_llm = counter.count
        # 检索冒烟
        t2 = time.perf_counter()
        q = await rag.aquery_data("高血压有哪些常见并发症？", param=QueryParam(mode="mix", top_k=20, chunk_top_k=5))
        query_time = time.perf_counter() - t2
        query_llm = counter.count - ingest_llm
        return ingest_time, ingest_llm, query_time, query_llm, q

    ingest_time, ingest_llm, query_time, query_llm, q = await _run()
    print(f"[B3] ingest 耗时={ingest_time:.1f}s | ingest LLM 调用={ingest_llm}（必须为 0）")
    print(f"[B3] 检索冒烟: {query_time:.1f}s（含关键词LLM={query_llm}）| "
          f"entities={len(q.get('data',{}).get('entities',[]))} chunks={len(q.get('data',{}).get('chunks',[]))}")
    if ingest_llm > 0:
        print("[STOP-1] ingest 触发 LLM！中止。")
        return 1

    # 6. 小规模 Baseline vs Experiment 对照
    questions = [
        "高血压有哪些常见并发症？",
        "二甲双胍用于治疗什么疾病？",
        "高血压和脑卒中有什么关系？",
    ]
    baseline_engine = RAGEngine()                      # 生产 cache
    experiment_engine = RAGEngine(working_dir=str(EXPERIMENT_CACHE))
    await baseline_engine.initialize()
    await experiment_engine.initialize()
    rows = await compare(baseline_engine, experiment_engine, questions)
    print("\n=== Baseline vs Experiment 小对照 ===")
    for r in rows:
        b_ents = r["baseline"]["entities"][:6]
        e_ents = r["experiment"]["entities"][:6]
        print(f"\nQ: {r['question']}")
        print(f"  Baseline  : entities={b_ents} chunks={r['baseline']['chunks']} "
              f"lat={r['baseline']['latency_s']}s")
        print(f"  Experiment: entities={e_ents} chunks={r['experiment']['chunks']} "
              f"lat={r['experiment']['latency_s']}s")

    # 汇总落盘
    BENCH_OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "limit": limit,
        "ingest_time_s": round(ingest_time, 2),
        "ingest_llm_calls": ingest_llm,
        "chunks": len(chunks), "entities": len(entities), "relationships": len(relationships),
        "multi_predicate_edges": multi,
        "comparison": rows,
        "note": "ingest 仅 ainsert_custom_kg，零 LLM 抽取；检索含关键词抽取 LLM（两侧一致）",
    }
    (BENCH_OUT / "b3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[B3] summary → {BENCH_OUT / 'b3_summary.json'}")
    print("[B3] 完成。STOP，等待批准进入 5000 条。")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args()
    raise SystemExit(asyncio.run(main(args.limit)))
