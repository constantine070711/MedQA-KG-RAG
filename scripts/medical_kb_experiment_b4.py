# -*- coding: utf-8 -*-
"""Phase 10-A.5 Step B4 — 5,000 条 CMeIE 正式结构化 ingest + 完整性检查（旁路，零 LLM）。

复用 B3 验证通过的 custom_kg 方案（唯一 ingest 路径），不切换回 ainsert()。

流程：
  1. Preflight（输入 SHA1 / 生产隔离 / 磁盘空间 / experiment cache 状态）
  2. 构建 custom_kg：14 markdown（生产 chunk+图只读复用）+ 5,000 CMeIE
  3. 合并（多 predicate 防覆盖聚合）
  4. ainsert_custom_kg → lightrag_cache_medkb（验证 ingest LLM 调用 = 0）
  5. 记录真实指标（ingest 时间 / LLM / 规模 / 磁盘 / 多 predicate / provenance）
  6. 完整性：chunks 数量 / 多 predicate lost_predicates=0 / provenance 随机抽样
  7. 输出 b4_ingest_summary.json

禁止：改生产 cache / 改生产代码 / monkey-patch / 触发 LLM 抽取 / 伪造指标。
"""

import argparse
import asyncio
import hashlib
import json
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightrag import LightRAG  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

from src.config import get_config  # noqa: E402
from src.data_adapter.medkb_custom_kg import build_custom_kg_from_file  # noqa: E402
from src.embedder import get_embedder  # noqa: E402
from src.llm_client import get_llm_client  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_CACHE = PROJECT_ROOT / "lightrag_cache"
EXPERIMENT_CACHE = PROJECT_ROOT / "lightrag_cache_medkb"
TRAIN_SRC = PROJECT_ROOT / "data" / "external" / "medical_kb" / "CMeIE" / "CMeIE_train.jsonl"
DOCS_SRC = PROJECT_ROOT / "data" / "processed" / "medical_kb_experiment" / "documents.jsonl"
BENCH_OUT = PROJECT_ROOT / "data" / "eval" / "medical_kb_benchmark"
EXPECTED_LIMIT = 5000


class LLMCounter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    async def __call__(self, prompt, system_prompt=None, history_messages=None, **kw):
        self.count += 1
        return await self.func(prompt, system_prompt=system_prompt,
                               history_messages=history_messages, **kw)


# ============================================================
# 生产 markdown 只读复用（与 B3 一致）
# ============================================================

def load_production_chunks() -> list[dict]:
    d = json.load(open(PRODUCTION_CACHE / "kv_store_text_chunks.json", encoding="utf-8"))
    return [{"content": v["content"], "source_id": k,
             "file_path": v.get("file_path", "unknown_source"), "chunk_order_index": 0}
            for k, v in d.items()]


def load_production_graph() -> tuple[list[dict], list[dict]]:
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    root = ET.parse(PRODUCTION_CACHE / "graph_chunk_entity_relation.graphml").getroot()
    entities = []
    for node in root.findall(".//g:node", ns):
        nid = node.get("id")
        a = {d.get("key"): d.text or "" for d in node.findall("g:data", ns)}
        entities.append({"entity_name": nid, "entity_type": a.get("d1", "UNKNOWN"),
                         "description": a.get("d2", ""), "source_id": a.get("d3", "UNKNOWN"),
                         "file_path": a.get("d4", "unknown_source")})
    relationships = []
    for edge in root.findall(".//g:edge", ns):
        a = {d.get("key"): d.text or "" for d in edge.findall("g:data", ns)}
        relationships.append({"src_id": edge.get("source"), "tgt_id": edge.get("target"),
                              "description": a.get("d8", ""), "keywords": a.get("d9", ""),
                              "weight": float(a.get("d7", "1.0") or 1.0),
                              "source_id": a.get("d10", "UNKNOWN"),
                              "file_path": a.get("d11", "unknown_source")})
    return entities, relationships


def merge_kg(md_chunks, md_entities, md_relations, cmeie):
    chunks = md_chunks + cmeie["chunks"]
    entities: OrderedDict[str, dict] = OrderedDict()
    for e in cmeie["entities"]:
        entities[e["entity_name"]] = e
    for e in md_entities:
        entities[e["entity_name"]] = e
    pair: OrderedDict[tuple[str, str], dict] = OrderedDict()
    for r in md_relations + cmeie["relationships"]:
        key = (r["src_id"], r["tgt_id"])
        info = pair.setdefault(key, {"preds": set(), "kws": set(),
                                     "source_id": r.get("source_id", "UNKNOWN")})
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
        relationships.append({"src_id": src, "tgt_id": tgt,
                              "description": "; ".join(sorted(info["preds"])) or "N/A",
                              "keywords": ", ".join(sorted(info["kws"])) or "N/A",
                              "weight": 1.0, "source_id": info["source_id"],
                              "file_path": "merged"})
    return chunks, list(entities.values()), relationships, multi


# ============================================================
# 完整性检查
# ============================================================

def check_multi_predicate_lost(records: list[dict], relationships: list[dict]) -> dict:
    """验证每个 CMeIE 三元组的 predicate 是否出现在合并边的 description 中。

    lost_predicates = 输入三元组 predicate 中，未在其 (subj,obj) 合并边中出现者。
    要求 = 0。
    """
    rel_by_pair: dict[tuple[str, str], str] = {}
    for r in relationships:
        rel_by_pair[(r["src_id"], r["tgt_id"])] = r["description"]
    lost = 0
    checked = 0
    for rec in records:
        for spo in rec.get("spo_list", []) or []:
            if not isinstance(spo, dict):
                continue
            from src.data_adapter.medkb_custom_kg import _unwrap_value
            subj = _unwrap_value(spo.get("subject"))
            obj = _unwrap_value(spo.get("object"))
            pred = str(spo.get("predicate", "") or "").strip()
            if not subj or not obj or not pred:
                continue
            checked += 1
            desc = rel_by_pair.get((subj, obj), "")
            if pred not in desc:
                lost += 1
    return {"triples_checked": checked, "lost_predicates": lost}


def sample_provenance(relationships: list[dict], chunk_to_sample: dict, n: int = 30) -> dict:
    """随机抽样 n 条关系，验证可追溯到 CMeIE sample。"""
    random.seed(42)
    sample = random.sample(relationships, min(n, len(relationships)))
    traceable = 0
    detail = []
    for r in sample:
        sid = chunk_to_sample.get(r["source_id"])
        ok = sid is not None
        if ok:
            traceable += 1
        detail.append({"src": r["src_id"], "tgt": r["tgt_id"],
                       "predicates": r["description"],
                       "source_id": r["source_id"], "sample_id": sid,
                       "traceable": ok})
    return {"sampled": len(sample), "traceable": traceable, "detail": detail[:8]}


def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def disk_usage(path: Path) -> float:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return round(total / (1024 * 1024), 2)


# ============================================================
# 主流程
# ============================================================

async def main(limit: int) -> int:
    # ---- 1. Preflight ----
    print("[B4] ===== Preflight =====")
    train_sha = sha1_file(TRAIN_SRC)
    print(f"[B4] 输入 CMeIE_train.jsonl SHA1[:16] = {train_sha[:16]}")
    n_recs = sum(1 for _ in open(TRAIN_SRC, encoding="utf-8") if _.strip())
    print(f"[B4] train 总行数 = {n_recs}（应为 14,339）")
    prod_chunk_sha = sha1_file(PRODUCTION_CACHE / "kv_store_text_chunks.json")
    prod_graph_sha = sha1_file(PRODUCTION_CACHE / "graph_chunk_entity_relation.graphml")
    print(f"[B4] 生产 text_chunks SHA1[:16] = {prod_chunk_sha[:16]}（只读）")
    print(f"[B4] 生产 graphml SHA1[:16] = {prod_graph_sha[:16]}（只读）")
    import shutil as _sh
    _sh.disk_usage  # noqa（确认可用）
    du = _sh.disk_usage(str(PROJECT_ROOT))
    print(f"[B4] 磁盘可用 = {du.free // (1024**3)} GB")
    if EXPERIMENT_CACHE.exists():
        print(f"[B4] 旧 experiment cache 存在（B3 产物），将清理重建（不影响生产）: "
              f"{disk_usage(EXPERIMENT_CACHE)} MB")
    ok = n_recs == 14339 and train_sha[:16] != ""
    print(f"[B4] Preflight: {'PASS' if ok else 'CHECK'}（输入=14339 行；生产只读）")

    # ---- 2. 构建 custom_kg ----
    print(f"\n[B4] ===== 构建 custom_kg（limit={limit}）=====")
    md_chunks = load_production_chunks()
    md_entities, md_relations = load_production_graph()
    print(f"[B4] 生产 markdown 复用: chunks={len(md_chunks)} entities={len(md_entities)} "
          f"relationships={len(md_relations)}")
    t0 = time.perf_counter()
    cmeie = build_custom_kg_from_file(TRAIN_SRC, limit=limit)
    build_s = time.perf_counter() - t0
    cs = cmeie["stats"]
    print(f"[B4] CMeIE custom_kg: records={cs['records']} chunks={cs['chunks']} "
          f"entities={cs['entities']} relationships={cs['relationships']} "
          f"triples_input={cs['triples_input']} 多predicate对={cs['multi_predicate_pairs']} "
          f"构建耗时={build_s:.2f}s")

    chunks, entities, relationships, multi = merge_kg(
        md_chunks, md_entities, md_relations, cmeie)
    print(f"[B4] 合并后: chunks={len(chunks)} entities={len(entities)} "
          f"relationships={len(relationships)} 多predicate边={multi}")

    # ---- 3. ingest（零 LLM）----
    print(f"\n[B4] ===== ainsert_custom_kg =====")
    disk_before = disk_usage(EXPERIMENT_CACHE) if EXPERIMENT_CACHE.exists() else 0.0
    if EXPERIMENT_CACHE.exists():
        shutil.rmtree(EXPERIMENT_CACHE)
    EXPERIMENT_CACHE.mkdir(parents=True, exist_ok=True)

    cfg = get_config()
    embedder = get_embedder()
    llm = get_llm_client()
    counter = LLMCounter(llm.llm_model_func)
    rag = LightRAG(working_dir=str(EXPERIMENT_CACHE),
                   embedding_func=EmbeddingFunc(embedding_dim=embedder.dim,
                                                max_token_size=cfg.embedding_max_token_size,
                                                func=embedder.embedding_func),
                   llm_model_func=counter)
    t1 = time.perf_counter()
    await rag.initialize_storages()
    t2 = time.perf_counter()
    await rag.ainsert_custom_kg({"chunks": chunks, "entities": entities,
                                 "relationships": relationships})
    t3 = time.perf_counter()
    ingest_llm = counter.count
    disk_after = disk_usage(EXPERIMENT_CACHE)
    print(f"[B4] init 耗时={t2-t1:.1f}s | ingest 耗时={t3-t2:.1f}s | "
          f"ingest LLM 调用={ingest_llm}（必须为 0）| 磁盘 {disk_before}→{disk_after} MB")
    if ingest_llm > 0:
        print("[STOP-1] ingest 触发 LLM！中止。")
        return 1

    # ---- 4. 完整性检查 ----
    print(f"\n[B4] ===== 完整性检查 =====")
    # 4a. 多 predicate lost
    records = []
    with open(TRAIN_SRC, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    lost_res = check_multi_predicate_lost(records, relationships)
    print(f"[B4] 多predicate: 检查三元组={lost_res['triples_checked']} "
          f"lost_predicates={lost_res['lost_predicates']}（要求=0）")
    # 4b. provenance 抽样
    chunk_to_sample = {c["source_id"]: int(c["source_id"].split("_")[-1])
                       if c["source_id"].startswith("medkb_cmeie_") else None
                       for c in chunks}
    prov_res = sample_provenance(relationships, chunk_to_sample, n=30)
    print(f"[B4] provenance 抽样: {prov_res['sampled']} 条, 可追溯 {prov_res['traceable']}")
    for d in prov_res["detail"]:
        print(f"    {d['src']}--{d['predicates'][:30]}-->{d['tgt']} "
              f"→ sample {d['sample_id']} traceable={d['traceable']}")

    # ---- 5. 落盘 summary ----
    BENCH_OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": {"chunks": len(chunks), "entities": len(entities),
                       "relationships": len(relationships),
                       "multi_predicate_edges": multi,
                       "disk_usage_before_mb": disk_before,
                       "disk_usage_after_mb": disk_after},
        "cmeie_stats": cs,
        "ingest": {"build_s": round(build_s, 2), "init_s": round(t2 - t1, 2),
                   "ingest_s": round(t3 - t2, 2),
                   "ingest_llm_calls": ingest_llm, "api_cost": 0.0},
        "integrity": {"multi_predicate": lost_res, "provenance": prov_res},
        "preflight": {"train_sha1": train_sha[:16], "train_lines": n_recs,
                      "prod_text_chunks_sha1": prod_chunk_sha[:16],
                      "prod_graphml_sha1": prod_graph_sha[:16]},
        "notes": "ingest 仅 ainsert_custom_kg，零 LLM 抽取，API cost=0",
    }
    (BENCH_OUT / "b4_ingest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[B4] summary → {BENCH_OUT / 'b4_ingest_summary.json'}")
    print("[B4] 5,000 条 ingest + 完整性检查完成。")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=EXPECTED_LIMIT)
    args = p.parse_args()
    raise SystemExit(asyncio.run(main(args.limit)))
