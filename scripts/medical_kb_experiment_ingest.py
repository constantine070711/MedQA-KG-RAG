# -*- coding: utf-8 -*-
"""Phase 10-A.5 Step 10-B — Experiment LightRAG KB 真实 ingest（旁路）。

用法：
  python scripts/medical_kb_experiment_ingest.py --preflight   # 只做 preflight，不 ingest
  python scripts/medical_kb_experiment_ingest.py --pilot 20    # 14 markdown + 前 20 条 CMeIE，测速后停止
  python scripts/medical_kb_experiment_ingest.py               # 全量（14 markdown + 5000 条，幂等续跑）

设计：
  - working_dir = lightrag_cache_medkb/（全新，与生产 lightrag_cache/ 完全隔离）
  - 复用 get_embedder() / get_llm_client() / 现有 LightRAG 版本，不改任何生产组件
  - llm_model_func 包装计数器：精确统计 LLM 调用次数 / 耗时 / 失败（token usage 由
    openai_complete_if_cache 返回字符串，不暴露 → 如实记 "usage unavailable"）
  - STOP 条件：LLM 调用数 > 10000（≈2×dry-run 预估）→ 立即中止
  - 幂等：LightRAG 按内容 md5 去重，重复 insert 自动跳过 → 续跑/重跑安全
  - 输出：data/eval/medical_kb_benchmark/experiment_ingest_summary.json
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightrag import LightRAG  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

from src.config import get_config  # noqa: E402
from src.data_loader import DataLoader  # noqa: E402
from src.embedder import get_embedder  # noqa: E402
from src.llm_client import get_llm_client  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_CACHE = PROJECT_ROOT / "lightrag_cache"
EXPERIMENT_CACHE = PROJECT_ROOT / "lightrag_cache_medkb"
DOCS_SRC = PROJECT_ROOT / "data" / "processed" / "medical_kb_experiment" / "documents.jsonl"
BENCH_OUT = PROJECT_ROOT / "data" / "eval" / "medical_kb_benchmark"

MAX_LLM_CALLS = 10000          # STOP 阈值：≈2× dry-run 预估（5000）
MAX_FAIL_RATE = 0.05           # 失败率阈值：>5% 中止


class LLMCounter:
    """包装 llm_model_func：精确统计调用/耗时/失败。"""

    def __init__(self, func):
        self.func = func
        self.count = 0
        self.errors = 0
        self.total_seconds = 0.0

    async def __call__(self, prompt, system_prompt=None, history_messages=None, **kwargs):
        t0 = time.perf_counter()
        self.count += 1
        try:
            return await self.func(prompt, system_prompt=system_prompt,
                                   history_messages=history_messages, **kwargs)
        except Exception:
            self.errors += 1
            raise
        finally:
            self.total_seconds += time.perf_counter() - t0


def load_14_markdown() -> list[dict]:
    """加载 14 个 markdown（只读），返回 [{name, text}]。"""
    loader = DataLoader()
    docs = loader.load_all()
    out = []
    for d in docs:
        text = d.content if d.format == "md" else d.plain_text
        out.append({"name": d.name, "text": text})
    return out


def load_cmeie_docs(limit: Optional[int] = None) -> list[dict]:
    """读取实验语料 documents.jsonl。"""
    docs = []
    with open(DOCS_SRC, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
            if limit is not None and len(docs) >= limit:
                break
    return docs


def preflight() -> dict:
    cfg = get_config()
    md_docs = load_14_markdown()
    cmeie_docs = load_cmeie_docs()
    total_chars = sum(len(d["text"]) for d in cmeie_docs) + sum(len(d["text"]) for d in md_docs)

    # 检查 ainsert 是否支持 file_path
    import inspect
    ainsert_sig = inspect.signature(LightRAG.ainsert)

    info = {
        "experiment_document_count": len(cmeie_docs),
        "markdown_14_count": len(md_docs),
        "total_documents": len(md_docs) + len(cmeie_docs),
        "experiment_chars": sum(len(d["text"]) for d in cmeie_docs),
        "markdown_chars": sum(len(d["text"]) for d in md_docs),
        "total_chars": total_chars,
        "est_chunks": sum(max(1, (len(d["text"]) + 1099) // 1100) for d in cmeie_docs) + len(md_docs),
        "working_dir": str(EXPERIMENT_CACHE),
        "working_dir_is_production": str(EXPERIMENT_CACHE) == str(PRODUCTION_CACHE),
        "production_working_dir": str(PRODUCTION_CACHE),
        "experiment_cache_exists": EXPERIMENT_CACHE.exists(),
        "production_cache_exists": PRODUCTION_CACHE.exists(),
        "llm_provider": "DeepSeek",
        "llm_model": cfg.deepseek_model,
        "llm_base_url": cfg.deepseek_base_url,
        "api_key_configured": bool(cfg.deepseek_api_key and "your-deepseek" not in cfg.deepseek_api_key),
        "embedding_model": cfg.embedding_model_name,
        "embedding_local_path": cfg.embedding_local_path,
        "embedding_dim": cfg.embedding_dim,
        "chunk_token_size": cfg.chunk_token_size,
        "chunk_overlap_token_size": cfg.chunk_overlap_token_size,
        "calls_entity_relation_extraction": True,
        "ainsert_supports_file_path": "file_path" in ainsert_sig.parameters,
        "duplicate_text_in_experiment": len(cmeie_docs) - len({d["text"] for d in cmeie_docs}),
        "stops": {"max_llm_calls": MAX_LLM_CALLS, "max_fail_rate": MAX_FAIL_RATE},
    }
    return info


def print_preflight(info: dict) -> None:
    print("=" * 64)
    print("PREFLIGHT — Experiment LightRAG Ingest")
    print("=" * 64)
    for k, v in info.items():
        print(f"  {k} = {v}")
    ok = (
        info["working_dir_is_production"] is False
        and info["api_key_configured"] is True
        and info["experiment_cache_exists"] is False
    )
    print("-" * 64)
    print(f"  working_dir != production: {not info['working_dir_is_production']}")
    print(f"  api_key_configured: {info['api_key_configured']}")
    print(f"  experiment_cache fresh (不存在): {not info['experiment_cache_exists']}")
    print(f"  PREFLIGHT: {'PASS' if ok else 'CHECK — 存在需注意项，见上'}")


async def run_ingest(limit_cmeie: Optional[int], counter: LLMCounter,
                     rag: LightRAG, md_docs, cmeie_docs, verbose: bool = True) -> dict:
    """摄入 14 markdown + CMeIE 文档（幂等，LightRAG 按 md5 去重）。"""
    inserted = {"markdown": 0, "cmeie": 0, "skipped_dup": 0}
    t_start = time.perf_counter()

    # 14 markdown（LightRAG 1.5.4 ainsert 不支持 file_path，沿用生产 index_docs 调用方式；
    # 内容按 md5 去重，distinct 文本不会误去重）
    for i, d in enumerate(md_docs, 1):
        try:
            await rag.ainsert(d["text"])
            inserted["markdown"] += 1
            if verbose:
                print(f"  [md {i}/{len(md_docs)}] {d['name']} ({len(d['text'])}字符)")
        except Exception as e:
            if verbose:
                print(f"  [md {i}] FAILED {d['name']}: {e}")
            raise

    # CMeIE 文档（默认前 limit_cmeie 条，None=全部）
    total = len(cmeie_docs)
    for i, d in enumerate(cmeie_docs, 1):
        if counter.count > MAX_LLM_CALLS:
            print(f"[STOP] LLM 调用数 {counter.count} 超过阈值 {MAX_LLM_CALLS}，中止 ingest。")
            break
        try:
            await rag.ainsert(d["text"])
            inserted["cmeie"] += 1
        except Exception as e:
            if verbose:
                print(f"  [cmeie {i}] FAILED {d['doc_id']}: {e}")
            raise
        if verbose and (i % 200 == 0 or i == 1 or i == total):
            rate = counter.total_seconds / max(1, counter.count)
            print(f"  [cmeie {i}/{total}] LLM调用={counter.count} "
                  f"avg={rate:.2f}s/调用 (累计 {counter.total_seconds:.0f}s)")

    elapsed = time.perf_counter() - t_start
    return {
        "started_at": None,  # 由外层补时间戳
        "elapsed_seconds": round(elapsed, 2),
        "inserted_markdown": inserted["markdown"],
        "inserted_cmeie": inserted["cmeie"],
        "llm_call_count": counter.count,
        "llm_error_count": counter.errors,
        "llm_total_seconds": round(counter.total_seconds, 2),
        "llm_avg_seconds_per_call": round(counter.total_seconds / max(1, counter.count), 2),
        "documents_requested_cmeie": min(limit_cmeie, total) if limit_cmeie else total,
        "documents_total_requested": len(md_docs) + (min(limit_cmeie, total) if limit_cmeie else total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment LightRAG KB ingest（旁路）")
    parser.add_argument("--preflight", action="store_true", help="只做 preflight，不 ingest")
    parser.add_argument("--pilot", type=int, default=None,
                        help="只摄入前 N 条 CMeIE（测速），然后停止")
    args = parser.parse_args()

    info = preflight()
    print_preflight(info)

    if args.preflight:
        return 0

    cfg = get_config()
    if info["working_dir_is_production"]:
        print("[STOP] working_dir 与生产一致，拒绝继续。")
        return 1
    if not info["api_key_configured"]:
        print("[STOP] API Key 未配置。")
        return 1
    if info["experiment_cache_exists"]:
        print("[注意] experiment cache 已存在，将幂等续跑（重复内容自动跳过）。")

    md_docs = load_14_markdown()
    cmeie_docs = load_cmeie_docs(limit=args.pilot)
    print(f"[ingest] markdown={len(md_docs)} cmeie={len(cmeie_docs)} (pilot={args.pilot})")

    embedder = get_embedder()
    llm_client = get_llm_client()
    counter = LLMCounter(llm_client.llm_model_func)

    EXPERIMENT_CACHE.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(EXPERIMENT_CACHE),
        embedding_func=EmbeddingFunc(
            embedding_dim=embedder.dim,
            max_token_size=cfg.embedding_max_token_size,
            func=embedder.embedding_func,
        ),
        llm_model_func=counter,
    )

    import datetime
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def _run_all():
        await rag.initialize_storages()
        return await run_ingest(args.pilot, counter, rag, md_docs, cmeie_docs)

    result = asyncio.run(_run_all())
    result["started_at"] = stamp
    result["mode"] = "pilot" if args.pilot else "full"
    result["usage_unavailable_reason"] = (
        "openai_complete_if_cache 返回字符串，不暴露 token usage → usage unavailable（不伪造）")

    BENCH_OUT.mkdir(parents=True, exist_ok=True)
    (BENCH_OUT / "experiment_ingest_summary.json").write_text(
        json.dumps({**info, **result}, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=" * 64)
    print("INGEST 结果:")
    for k, v in result.items():
        print(f"  {k} = {v}")
    print(f"  完整 summary: {BENCH_OUT / 'experiment_ingest_summary.json'}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
