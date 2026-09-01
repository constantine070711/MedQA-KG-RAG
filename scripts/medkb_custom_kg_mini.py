# -*- coding: utf-8 -*-
"""Phase 10-A.5 Performance Audit — 最小实验：Candidate（custom_kg 结构化路径）验证。

对比基线（已实测）：
  Baseline = 原始 text → LightRAG 正常 LLM extraction：5 条文档 301.9s / 10 次 LLM 调用（30s/次）

Candidate = CMeIE 已有三元组 → ainsert_custom_kg（零 LLM 抽取，仅本地 embedding）。

本脚本：
  1. 取前 N 条 CMeIE 记录（默认 10）
  2. 构建 custom_kg = {chunks(text), entities(subject/object), relationships(triples)}
  3. ainsert_custom_kg 插入全新实验缓存
  4. 记录：耗时 / LLM 调用数（应为 0）/ 图规模
  5. 检索冒烟：aquery_data 一道题，确认检索可用

用法：
  python scripts/medkb_custom_kg_mini.py [--n 10]

隔离：仅写 lightrag_cache_medkb（可丢弃实验目录），不碰生产。
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightrag import LightRAG  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

from src.config import get_config  # noqa: E402
from src.embedder import get_embedder  # noqa: E402
from src.llm_client import get_llm_client  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SRC = PROJECT_ROOT / "data" / "external" / "medical_kb" / "CMeIE" / "CMeIE_train.jsonl"
EXPERIMENT_CACHE = PROJECT_ROOT / "lightrag_cache_medkb"

# 复用 medical_kb_adapter 的类型/关系映射
sys.path.insert(0, str(PROJECT_ROOT))
from src.data_adapter.medical_kb_adapter import (  # noqa: E402
    _unwrap_value,
    CMEIE_TYPE_MAP,
    MEDKB_RELATION_MAP,
)


def unwrap(v):
    return _unwrap_value(v)


def build_custom_kg(records: list[dict]) -> dict:
    """从 CMeIE 记录构建 custom_kg（chunks + entities + relationships）。"""
    chunks = []
    entities: dict[str, dict] = {}
    relationships = []

    for i, rec in enumerate(records):
        text = str(rec.get("text", "") or "").strip()
        source_id = f"medkb_cmeie_{i:05d}"
        chunks.append({
            "content": text,
            "source_id": source_id,
            "file_path": "cmeie",
            "chunk_order_index": 0,
        })
        for spo in rec.get("spo_list", []) or []:
            subj = unwrap(spo.get("subject"))
            obj = unwrap(spo.get("object"))
            pred = str(spo.get("predicate", "") or "").strip()
            if not subj or not obj or not pred:
                continue
            # 实体
            for name, raw_type in [(subj, spo.get("subject_type")),
                                   (obj, unwrap(spo.get("object_type")))]:
                if not name:
                    continue
                etype = CMEIE_TYPE_MAP.get(str(raw_type or "").strip(), "UNKNOWN")
                if name not in entities:
                    entities[name] = {
                        "entity_name": name,
                        "entity_type": etype,
                        "description": f"CMeIE 实体：{name}（记录 {source_id}）",
                        "source_id": source_id,
                        "file_path": "cmeie",
                    }
            # 关系（无向边；keywords=映射后 RelationType 或原 predicate）
            rel_type = MEDKB_RELATION_MAP.get(pred, "UNMAPPED")
            relationships.append({
                "src_id": subj,
                "tgt_id": obj,
                "description": pred,
                "keywords": rel_type if rel_type != "UNMAPPED" else pred,
                "weight": 1.0,
                "source_id": source_id,
                "file_path": "cmeie",
            })

    return {"chunks": chunks, "entities": list(entities.values()),
            "relationships": relationships}


class LLMCounter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    async def __call__(self, prompt, system_prompt=None, history_messages=None, **kw):
        self.count += 1
        return await self.func(prompt, system_prompt=system_prompt,
                               history_messages=history_messages, **kw)


async def main(n: int) -> int:
    records = []
    with open(TRAIN_SRC, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[mini] 取前 {len(records)} 条 CMeIE 记录")

    custom_kg = build_custom_kg(records)
    n_chunks = len(custom_kg["chunks"])
    n_ent = len(custom_kg["entities"])
    n_rel = len(custom_kg["relationships"])
    print(f"[mini] custom_kg: chunks={n_chunks} entities={n_ent} relationships={n_rel}")

    # 清空实验缓存（可丢弃）
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

    await rag.initialize_storages()
    t0 = time.perf_counter()
    await rag.ainsert_custom_kg(custom_kg)
    elapsed = time.perf_counter() - t0
    print(f"[mini] ainsert_custom_kg 完成：{elapsed:.1f}s | LLM 调用数 = {counter.count}")

    # 检索冒烟
    from lightrag import QueryParam
    t1 = time.perf_counter()
    res = await rag.aquery_data("产后抑郁症和轻度情绪失调是什么关系？", param=QueryParam(mode="mix", top_k=20, chunk_top_k=5))
    dt = time.perf_counter() - t1
    ents = res.get("data", {}).get("entities", [])
    chunks_r = res.get("data", {}).get("chunks", [])
    print(f"[mini] 检索冒烟：{dt:.1f}s | entities={len(ents)} chunks={len(chunks_r)} | "
          f"命中实体样例={[e.get('entity_name') for e in ents[:5]]}")

    # 图规模
    import os
    graphml = EXPERIMENT_CACHE / "graph_chunk_entity_relation.graphml"
    print(f"[mini] 图文件: {graphml.name} ({graphml.stat().st_size if graphml.exists() else 0} bytes)")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    raise SystemExit(asyncio.run(main(args.n)))
