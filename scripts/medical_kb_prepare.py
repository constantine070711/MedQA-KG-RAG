# -*- coding: utf-8 -*-
"""Phase 10-A.5 — 真实医学知识库（CMeIE）预处理：清洗 + 转换 + Entity Linking + 产物落盘。

数据流：
  data/external/medical_kb/CMeIE/CMeIE_train.jsonl（原始，只读）
    → ingest（解析 + 注入 record_idx + 基础统计 + 去重）
    → convert（统一 schema + 校验回写 quality）
    → Entity Linking（L0-L3 复用 cblue + L4 相似度候选）
    → 写产物：
        data/raw/medical_kb/cleaned_records.jsonl      （清洗中间态，可审计）
        data/processed/medical_kb/entities.json
        data/processed/medical_kb/relations.json
        data/processed/medical_kb/triples.json
        data/processed/medical_kb/provenance.json
        data/processed/medical_kb/entity_linking.json
        data/processed/medical_kb/summary.json

用法：
  python scripts/medical_kb_prepare.py [--limit N] [--input PATH]

只读原始数据，不改生产核心 / 不写 Neo4j / 不进生产 LightRAG。
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_adapter.medical_kb_adapter import (  # noqa: E402
    DATASET_NAME,
    convert,
    ingest,
    link_entities,
    link_entities_l4,
    load_production_codes,
    load_production_name_map,
    summarize,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "external" / "medical_kb" / "CMeIE"
RAW_OUT = PROJECT_ROOT / "data" / "raw" / "medical_kb"
PROCESSED_OUT = PROJECT_ROOT / "data" / "processed" / "medical_kb"


def main() -> int:
    parser = argparse.ArgumentParser(description="CMeIE 真实医学知识库预处理（旁路）")
    parser.add_argument("--limit", type=int, default=None,
                        help="最多处理记录数（None=全部，默认全部 14,339 条）")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="输入文件或目录（默认 CMeIE 数据目录，优先 *train*.jsonl）")
    args = parser.parse_args()

    RAW_OUT.mkdir(parents=True, exist_ok=True)
    PROCESSED_OUT.mkdir(parents=True, exist_ok=True)

    print(f"[medical_kb] 输入: {args.input}  limit={args.limit}")
    ingest_result = ingest(args.input, split="train", limit=args.limit)
    stats = ingest_result["stats"]
    print(f"[medical_kb] 读取: {stats['records_loaded']} 条记录 / "
          f"原始实体 {stats['entities_raw']} / 原始关系 {stats['relations_raw']}")

    converted = convert(ingest_result)
    entities, relations = converted["entities"], converted["relations"]
    print(f"[medical_kb] 转换后: 实体 {len(entities)} / 关系 {len(relations)}")

    # ---- Entity Linking（L0-L3 复用 cblue + L4 相似度候选）----
    prod_name_map = load_production_name_map()
    prod_codes = load_production_codes()
    linking = link_entities(entities, prod_name_map, prod_codes)
    # L4：对 unmatched 做字符相似度候选，命中的标记 match_level=="L4" 并入 candidate
    l4_results = link_entities_l4(linking.get("unmatched", []), prod_name_map)
    l4_hits = [c for c in l4_results if c.get("match_level") == "L4"]
    linking["unmatched"] = [u for u in l4_results if u.get("match_level") != "L4"]
    linking["candidate"] = linking.get("candidate", []) + l4_hits
    linking["link_summary"] = {
        "linked": len(linking.get("linked", [])),
        "candidate": len(linking.get("candidate", [])),
        "unmatched": len(linking.get("unmatched", [])),
        "l4_similarity_hits": len(l4_hits),
    }
    print(f"[medical_kb] Entity Linking: linked={linking['link_summary']['linked']} "
          f"candidate={linking['link_summary']['candidate']} "
          f"unmatched={linking['link_summary']['unmatched']} "
          f"(L4 相似度 {linking['link_summary']['l4_similarity_hits']})")

    summary = summarize(converted, ingest_result["stats"])

    # ---- 落盘 ----
    def _dump(name: str, data) -> None:
        (PROCESSED_OUT / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    _dump("entities.json", entities)
    _dump("relations.json", relations)
    _dump("triples.json", [{
        "subject": r["subject"], "predicate": r["relation"],
        "object": r["object"], "relation_type": r["relation_type"],
        "provenance": r["provenance"]} for r in relations])
    _dump("provenance.json", {
        "entities": [e["provenance"] for e in entities],
        "relations": [r["provenance"] for r in relations],
    })
    _dump("entity_linking.json", linking)
    _dump("summary.json", summary)

    # 清洗中间态（可审计）：每条记录 → 清洗后三元组
    cleaned_records = []
    for r in relations:
        cleaned_records.append({
            "subject": r["subject"],
            "predicate": r["relation"],
            "object": r["object"],
            "relation_type": r["relation_type"],
            "provenance": r["provenance"],
        })
    (RAW_OUT / "cleaned_records.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cleaned_records), encoding="utf-8")

    print(f"[medical_kb] 产物已写入: {PROCESSED_OUT}")
    print(f"[medical_kb] summary: 实体={summary['entities']} 关系={summary['relations']} "
          f"三元组去重前={summary['triples_input']} 去重后={summary['triples_unique']} "
          f"未映射关系={summary['unmapped_relations']} ({summary['unmapped_relation_ratio']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
