# -*- coding: utf-8 -*-
"""Phase 10-A.5 Step B1 — CMeIE custom_kg 构建 + dry run / insert（旁路，零 LLM 抽取）。

用法：
  python scripts/medical_kb_custom_kg.py --limit 10 --dry-run   # 只构建+统计，不写图
  python scripts/medical_kb_custom_kg.py --limit 10 --insert    # 构建 + ainsert_custom_kg + 零 LLM 验证 + 检索冒烟

说明：
  - 输入：data/external/medical_kb/CMeIE/CMeIE_train.jsonl（原始，只读）
  - 输出：data/processed/medical_kb_experiment/custom_kg_{limit}.json（dry-run 产物）
         data/processed/medical_kb_experiment/custom_kg_stats_{limit}.json
  - insert 模式写入 lightrag_cache_medkb/（独立实验目录），验证 LLM extraction calls == 0
  - 不改生产 / 不改 LightRAG 源码 / 不 monkey-patch
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightrag import LightRAG, QueryParam  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

from src.config import get_config  # noqa: E402
from src.data_adapter.medkb_custom_kg import build_custom_kg_from_file  # noqa: E402
from src.embedder import get_embedder  # noqa: E402
from src.llm_client import get_llm_client  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SRC = PROJECT_ROOT / "data" / "external" / "medical_kb" / "CMeIE" / "CMeIE_train.jsonl"
EXPERIMENT_CACHE = PROJECT_ROOT / "lightrag_cache_medkb"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "medical_kb_experiment"


class LLMCounter:
    """验证零 LLM 抽取：若 ainsert_custom_kg 触发任何 LLM 调用则计数 > 0。"""

    def __init__(self, func):
        self.func = func
        self.count = 0
        self.calls: list[str] = []

    async def __call__(self, prompt, system_prompt=None, history_messages=None, **kw):
        self.count += 1
        self.calls.append((prompt or "")[:80])
        return await self.func(prompt, system_prompt=system_prompt,
                               history_messages=history_messages, **kw)


def main() -> int:
    parser = argparse.ArgumentParser(description="CMeIE custom_kg 构建（零 LLM 抽取）")
    parser.add_argument("--limit", type=int, default=10, help="取前 N 条（默认 10）")
    parser.add_argument("--dry-run", action="store_true", help="只构建+统计，不写图")
    parser.add_argument("--insert", action="store_true", help="构建 + ainsert_custom_kg + 验证")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result = build_custom_kg_from_file(TRAIN_SRC, limit=args.limit)
    build_time = time.perf_counter() - t0
    stats = result["stats"]
    print("=" * 64)
    print(f"custom_kg 构建（limit={args.limit}）")
    for k, v in stats.items():
        print(f"  {k} = {v}")
    print(f"  构建耗时 = {build_time:.3f}s")
    print(f"  多 predicate 对 = {stats['multi_predicate_pairs']} "
          f"（最大 {stats['max_predicates_per_pair']} 个 predicate/对）")
    print("=" * 64)

    # dry-run：输出产物，不写图
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.dry_run or not args.insert:
        (OUT_DIR / f"custom_kg_{args.limit}.json").write_text(
            json.dumps({k: result[k] for k in ("chunks", "entities", "relationships")},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        (OUT_DIR / f"custom_kg_stats_{args.limit}.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
        (OUT_DIR / f"custom_kg_provenance_{args.limit}.json").write_text(
            json.dumps(result["provenance_manifest"], ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"[dry-run] 产物已写：custom_kg_{args.limit}.json + stats + provenance")
        print("[dry-run] 未写图、未调用 LLM。")
        return 0

    # ---- insert 模式 ----
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
        await rag.ainsert_custom_kg({
            "chunks": result["chunks"],
            "entities": result["entities"],
            "relationships": result["relationships"],
        })
        ingest_time = time.perf_counter() - t1
        ingest_llm_calls = counter.count        # 仅 ingest 阶段的 LLM 调用（应为 0）
        # 检索冒烟（aquery_data 每次查询含 1 次关键词抽取 LLM 调用——预期、非 ingest）
        t2 = time.perf_counter()
        q = await rag.aquery_data("高血压的并发症有哪些？", param=QueryParam(mode="mix", top_k=20, chunk_top_k=5))
        query_time = time.perf_counter() - t2
        query_llm_calls = counter.count - ingest_llm_calls
        return ingest_time, query_time, q, ingest_llm_calls, query_llm_calls

    ingest_time, query_time, q, ingest_llm_calls, query_llm_calls = asyncio.run(_run())

    print("[insert] ainsert_custom_kg 完成")
    print(f"  ingest 耗时 = {ingest_time:.1f}s")
    print(f"  ingest 阶段 LLM 调用 = {ingest_llm_calls}  (必须为 0)")
    print(f"  检索冒烟阶段 LLM 调用 = {query_llm_calls}  (关键词抽取，预期 1/查询)")
    if ingest_llm_calls > 0:
        print("[STOP-1] custom_kg ingest 触发了 LLM 调用！立即中止。")
        print(f"  调用样例: {counter.calls[:3]}")
        return 1
    ents = q.get("data", {}).get("entities", [])
    chk = q.get("data", {}).get("chunks", [])
    print(f"  检索冒烟: {query_time:.1f}s | entities={len(ents)} chunks={len(chk)} "
          f"命中样例={[e.get('entity_name') for e in ents[:5]]}")
    graphml = EXPERIMENT_CACHE / "graph_chunk_entity_relation.graphml"
    print(f"  图文件: {graphml.name} ({graphml.stat().st_size if graphml.exists() else 0} bytes)")
    print("[insert] PASS：零 LLM 抽取 + 检索可用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
