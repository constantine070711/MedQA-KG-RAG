# -*- coding: utf-8 -*-
"""Phase 10-A — CBLUE 数据转换（幂等）：external/cblue → processed/cblue。

用法:
    python scripts/cblue_prepare.py                       # 全部子集，无限制
    python scripts/cblue_prepare.py --subsets cmeie_v2    # 仅 CMeIE-V2
    python scripts/cblue_prepare.py --limit 5000          # 前 5000 条（真实数据）
    python scripts/cblue_prepare.py --input <dir> --output <dir>

产物（data/processed/cblue/）:
    entities.json          # 统一实体（带 provenance/quality）
    relations.json         # 统一关系（带 provenance/quality/relation_type）
    triples.json           # 简化三元组
    provenance.json        # provenance 索引（逐条 cblue 溯源）
    entity_linking.json    # Entity Linking 候选 {linked, candidate, unmatched}
    summary.json           # 运行统计（输入/输出/质量）

原则：
  - 只处理小规模样本（--limit 控制）；不下载、不进生产 LightRAG、不写 Neo4j
  - 幂等：基于 sample_id/entity_id 去重；重复运行产物稳定
  - 不修改生产 entities.json / Neo4j / Agent / evaluation
"""

import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_adapter import cblue_adapter as ca  # noqa: E402


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="CBLUE 数据转换（幂等，独立实验旁路）")
    parser.add_argument("--input", default="data/external/cblue",
                        help="CBLUE 原始/样本目录（默认 data/external/cblue）")
    parser.add_argument("--output", default="data/processed/cblue",
                        help="产物目录（默认 data/processed/cblue）")
    parser.add_argument("--subsets", default="cmeee_v2,cmeie_v2",
                        help="子集列表，逗号分隔（默认 cmeee_v2,cmeie_v2）")
    parser.add_argument("--split", default="train", help="数据划分（默认 train）")
    parser.add_argument("--limit", type=int, default=None,
                        help="每子集最多处理记录数（默认全部）")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    input_path = Path(args.input) if Path(args.input).is_absolute() else root / args.input
    out_path = Path(args.output) if Path(args.output).is_absolute() else root / args.output

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    all_entities: list[dict] = []
    all_relations: list[dict] = []
    provenance_index: dict[str, dict] = {}
    per_subset: dict[str, dict] = {}

    print(f"[CBLUE-PREPARE] input={input_path} output={out_path} subsets={subsets} split={args.split} limit={args.limit}")
    print(f"[CBLUE-PREPARE] license: {ca.CBLUE_LICENSE}（学术非商业，须署名，同方式共享）")
    print()

    for subset in subsets:
        res = ca.ingest(input_path, subset, split=args.split, limit=args.limit)
        conv = ca.convert(res)
        per_subset[subset] = {
            "records_loaded": res["stats"]["records_loaded"],
            "entities_raw": res["stats"]["entities_raw"],
            "relations_raw": res["stats"]["relations_raw"],
            "entities": len(conv["entities"]),
            "relations": len(conv["relations"]),
            "skipped": len(res["skipped"]),
        }
        all_entities.extend(conv["entities"])
        all_relations.extend(conv["relations"])
        for e in conv["entities"]:
            if e.get("provenance"):
                provenance_index[e["entity_id"]] = e["provenance"]
        for r in conv["relations"]:
            key = f"rel:{r['subject']}:{r['relation_type']}:{r['object']}"
            provenance_index[key] = r["provenance"]
        print(f"  [{subset}] records={res['stats']['records_loaded']} "
              f"entities_raw={res['stats']['entities_raw']}->{len(conv['entities'])} "
              f"relations_raw={res['stats']['relations_raw']}->{len(conv['relations'])} "
              f"skipped={len(res['skipped'])}")

    # 汇总统计（unresolved / duplicate / missing provenance）
    ingest_stats = {
        "entities_raw": sum(v["entities_raw"] for v in per_subset.values()),
        "relations_raw": sum(v["relations_raw"] for v in per_subset.values()),
    }
    summary = ca.summarize({"entities": all_entities, "relations": all_relations},
                           ingest_stats)
    summary["per_subset"] = per_subset
    summary["split"] = args.split

    # Entity Linking 候选（只读生产 entities.json，不修改）
    prod_name_map = ca.load_production_name_map()
    prod_codes = ca.load_production_codes()
    # link_entities 需要 CBLUEEntity 或 dict；这里传 dict 列表
    linking = ca.link_entities(all_entities, prod_name_map, prod_codes)

    _write_json(out_path / "entities.json", {"generated_at": ca.INGEST_DATE,
                                              "total": len(all_entities),
                                              "license": ca.CBLUE_LICENSE,
                                              "entities": all_entities})
    _write_json(out_path / "relations.json", {"generated_at": ca.INGEST_DATE,
                                               "total": len(all_relations),
                                               "license": ca.CBLUE_LICENSE,
                                               "relations": all_relations})
    triples = []
    for r in all_relations:
        triples.append({"subject": r["subject"], "predicate": r["relation"],
                        "object": r["object"], "relation_type": r["relation_type"],
                        "provenance": r["provenance"]})
    _write_json(out_path / "triples.json", {"generated_at": ca.INGEST_DATE,
                                             "total": len(triples),
                                             "triples": triples})
    _write_json(out_path / "provenance.json", {"generated_at": ca.INGEST_DATE,
                                                "total": len(provenance_index),
                                                "provenance": provenance_index})
    _write_json(out_path / "entity_linking.json", linking)
    _write_json(out_path / "summary.json", summary)

    print()
    print("=== SUMMARY ===")
    print(f"  entities:            {summary['entities']}")
    print(f"  relations:           {summary['relations']}")
    print(f"  triples:             {summary['triples']}")
    print(f"  unmapped_relations:  {summary['unmapped_relations']}")
    print(f"  unknown_type_ents:   {summary['unknown_type_entities']}")
    print(f"  dup_merged_entities: {summary['duplicates_merged_entities']}")
    print(f"  dup_merged_rels:     {summary['duplicates_merged_relations']}")
    print(f"  missing_provenance:  {summary['missing_provenance']}")
    print(f"  quality_fail_ent/rel:{summary['quality_fail_entities']}/{summary['quality_fail_relations']}")
    print(f"  entity_linking:      linked={len(linking['linked'])} candidate={len(linking['candidate'])} unmatched={len(linking['unmatched'])}")
    print(f"[OK] 产物已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
