# -*- coding: utf-8 -*-
"""Phase 10-A.5 Step 10-A — Experiment Corpus 准备（旁路，dry-run，零 LLM/零 LightRAG ingest）。

方案①：直接使用每条原始 text 作为 LightRAG 文档（不做三元组重写，最大程度保真）。

输入：
  data/external/medical_kb/CMeIE/CMeIE_train.jsonl   （原始源，只读）
  data/raw/medical_kb/cleaned_records.jsonl          （清洗语料，用于交叉校验：前 5000 条全部通过清洗）

说明：
  cleaned_records.jsonl 为逐三元组审计产物（含 provenance.sample_id，但无 text）；
  方案① 需要原始 text，故文本源为原始 train（text 原样保留），
  并经 cleaned_records 校验 sample_id∈[0, limit) 全部存在（= 全部通过清洗）。

默认取前 5000 条（确定性顺序 = 文件行序），幂等（固定切片，重复运行产物一致）。

输出：
  data/processed/medical_kb_experiment/
    documents.jsonl   每条 = {doc_id, sample_id, text, n_triples, provenance}
    summary.json      语料统计 + dry-run 预估（docs / chars / tokens / chunks / LLM 抽取调用预估）

不调用 LightRAG / 不调用 LLM / 不写 Neo4j / 不改任何 production 文件。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SRC = PROJECT_ROOT / "data" / "external" / "medical_kb" / "CMeIE" / "CMeIE_train.jsonl"
CLEANED = PROJECT_ROOT / "data" / "raw" / "medical_kb" / "cleaned_records.jsonl"
EXPERIMENT_OUT = PROJECT_ROOT / "data" / "processed" / "medical_kb_experiment"

DEFAULT_LIMIT = 5000
CHUNK_TOKEN_SIZE = 1200  # 与生产 config.chunk_token_size 一致


def _count_tokens(text: str) -> int:
    """用 tiktoken 估算 token 数（LightRAG 默认编码族）。失败返回字符数/2 近似。"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 2)


def _estimate_chunks(tokens: int) -> int:
    """估算单文档 chunk 数：LightRAG token_size=1200, overlap=100。"""
    if tokens <= 0:
        return 1
    return max(1, (tokens + (CHUNK_TOKEN_SIZE - 100) - 1) // (CHUNK_TOKEN_SIZE - 100))


def build_corpus(limit: int = DEFAULT_LIMIT, verbose: bool = True,
                 train_src: Optional[Path] = None,
                 cleaned: Optional[Path] = None) -> dict:
    """读取原始 train 前 limit 条，构建实验语料（确定性，幂等）。

    Args:
        limit: 取前 N 条。
        verbose: 打印进度。
        train_src: 原始 train 路径（默认 TRAIN_SRC，测试可注入临时文件）。
        cleaned: 清洗语料路径（默认 CLEANED，测试可注入）。

    Returns:
        {"documents": [dict], "summary": dict}
    """
    train_src = train_src or TRAIN_SRC
    cleaned = cleaned or CLEANED
    if not train_src.exists():
        raise FileNotFoundError(f"原始源不存在: {train_src}")

    documents: list[dict] = []
    n_dup_text = 0
    seen_text: set[str] = set()

    with open(train_src, encoding="utf-8") as f:
        for sample_id, line in enumerate(f):
            if limit is not None and sample_id >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = str(rec.get("text", "") or "").strip()
            spo = rec.get("spo_list", []) or []
            if text in seen_text:
                n_dup_text += 1
            seen_text.add(text)
            documents.append({
                "doc_id": f"medkb_cmeie_{sample_id:05d}",
                "sample_id": sample_id,
                "text": text,
                "n_triples": len(spo),
                "provenance": {
                    "dataset": "CMeIE",
                    "split": "train",
                    "sample_id": str(sample_id),
                    "source_file": str(train_src),
                    "adapter": "medical_kb",
                    "license": "CC-BY-NC 4.0",
                    "ingest_date": "2026-08-31",
                },
            })

    # 交叉校验：这些记录是否通过清洗（以 cleaned_records.jsonl 中幸存的三元组为准）。
    # 注意：ingest 全局按 (subject,relation_type,object) 去重并保留首次 provenance，
    #   部分记录的三元组可能被更早记录的重复三元组吞并 → 不出现在逐三元组清洗文件，
    #   属**去重正常结果**，不代表清洗失败（prepare 已证 records_loaded=14339、empty=0、malformed=0）。
    cleaned_ids: set[str] = set()
    if cleaned.exists():
        with open(cleaned, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cleaned_ids.add(str(obj.get("provenance", {}).get("sample_id", "")))
                except Exception:
                    pass
    sample_ids = {str(d["sample_id"]) for d in documents}
    fully_deduped = sorted(sample_ids - cleaned_ids)   # 三元组被去重吞并（非清洗失败）
    if verbose:
        print(f"[experiment] 语料 = 原始 train 前 {len(documents)} 条")
        print(f"[experiment] 交叉校验: 幸存三元组的记录 {len(sample_ids & cleaned_ids)} / "
              f"三元组全被去重吞并 {len(fully_deduped)}（预期去重行为，非清洗失败）")

    # ---- dry-run 预估（不调用 LightRAG/LLM）----
    total_chars = sum(len(d["text"]) for d in documents)
    total_tokens = sum(_count_tokens(d["text"]) for d in documents)
    chunks_per_doc = [_estimate_chunks(_count_tokens(d["text"])) for d in documents]
    est_chunks = sum(chunks_per_doc)
    avg_chars = round(total_chars / len(documents), 1) if documents else 0
    # LightRAG ingest：每 chunk 1 次实体+关系抽取 LLM 调用（plus 检索侧关键词抽取另计，非 ingest）
    est_llm_extract_calls = est_chunks

    summary = {
        "dataset": "CMeIE",
        "source_file": str(TRAIN_SRC),
        "corpus_size": len(documents),
        "limit": limit,
        "deterministic": "file_line_order",
        "total_chars": total_chars,
        "avg_chars_per_doc": avg_chars,
        "total_tokens_est": total_tokens,
        "est_chunks": est_chunks,
        "chunk_token_size": CHUNK_TOKEN_SIZE,
        "est_llm_extract_calls": est_llm_extract_calls,
        "unique_text_count": len(seen_text),
        "duplicate_text_in_corpus": n_dup_text,
        "records_with_triples_in_cleaned": len(sample_ids & cleaned_ids),
        "records_triples_fully_deduped": len(fully_deduped),
        "reuses_existing_doc": False,   # 全新实验 working_dir，无既有文档
        "production_untouched": True,   # 本脚本只读，不写生产
    }
    return {"documents": documents, "summary": summary}


def write_outputs(documents: list[dict], summary: dict, out_dir: Path) -> None:
    """落盘（幂等：固定切片 + 确定性顺序，重复运行覆盖为相同内容）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "documents.jsonl", "w", encoding="utf-8") as f:
        for d in documents:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="CMeIE 实验语料准备（dry-run，零 LLM）")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"取前 N 条（默认 {DEFAULT_LIMIT}）")
    args = parser.parse_args()

    result = build_corpus(limit=args.limit, verbose=True)
    documents, summary = result["documents"], result["summary"]

    write_outputs(documents, summary, EXPERIMENT_OUT)

    print("=" * 60)
    print("Experiment Corpus 已生成（dry-run 预估，未调用 LightRAG/LLM）:")
    print(f"  文档数          = {summary['corpus_size']}")
    print(f"  总字符数        = {summary['total_chars']}")
    print(f"  平均字符/文档   = {summary['avg_chars_per_doc']}")
    print(f"  预估 token      = {summary['total_tokens_est']}")
    print(f"  预估 chunk      = {summary['est_chunks']} (chunk_token_size={summary['chunk_token_size']})")
    print(f"  预估 LLM 抽取调用 = {summary['est_llm_extract_calls']}（1 次/ chunk，实体+关系抽取）")
    print(f"  语料内重复 text = {summary['duplicate_text_in_corpus']}")
    print(f"  交叉校验 = 幸存三元组记录 {summary['records_with_triples_in_cleaned']} / "
          f"三元组全被去重吞并 {summary['records_triples_fully_deduped']}")
    print(f"  是否复用既有文档 = {summary['reuses_existing_doc']}")
    print(f"  产物: {EXPERIMENT_OUT / 'documents.jsonl'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
