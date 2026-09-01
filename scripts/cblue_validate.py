# -*- coding: utf-8 -*-
"""Phase 10-A — CBLUE 产物质量校验：data/processed/cblue/ → 可读质量报告。

用法:
    python scripts/cblue_validate.py
    python scripts/cblue_validate.py --input data/processed/cblue

校验内容:
    - schema 字段完整性（entities / relations / triples）
    - unresolved（未映射关系 / 未知类型实体）
    - duplicate（重复实体/关系，应已在 prepare 阶段去重）
    - missing provenance（溯源缺失）
    - quality level（pass/warn/fail 分布）

只读校验，不修改任何产物。
"""

import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="CBLUE 产物质量校验（只读）")
    parser.add_argument("--input", default="data/processed/cblue",
                        help="产物目录（默认 data/processed/cblue）")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    in_path = Path(args.input) if Path(args.input).is_absolute() else root / args.input

    missing_files = []
    for fn in ("entities.json", "relations.json", "triples.json",
               "provenance.json", "entity_linking.json", "summary.json"):
        if not (in_path / fn).exists():
            missing_files.append(fn)
    if missing_files:
        print(f"[VALIDATE] 缺失产物: {missing_files}")
        print(f"[VALIDATE] 请先运行 python scripts/cblue_prepare.py")
        return 1

    ents = _load(in_path / "entities.json").get("entities", [])
    rels = _load(in_path / "relations.json").get("relations", [])
    triples = _load(in_path / "triples.json").get("triples", [])
    prov = _load(in_path / "provenance.json").get("provenance", {})
    linking = _load(in_path / "entity_linking.json")
    summary = _load(in_path / "summary.json")

    print("=== CBLUE 产物质量校验 ===")
    print(f"输入目录: {in_path}")

    # 1. schema 完整性
    ent_required = ("entity_id", "name", "entity_type", "provenance", "quality")
    rel_required = ("subject", "relation", "object", "relation_type", "provenance")
    ent_missing = sum(1 for e in ents if any(k not in e for k in ent_required))
    rel_missing = sum(1 for r in rels if any(k not in r for k in rel_required))
    tri_missing = sum(1 for t in triples if any(k not in t for k in ("subject", "predicate", "object")))
    print(f"[schema] entities={len(ents)} missing_field={ent_missing}")
    print(f"[schema] relations={len(rels)} missing_field={rel_missing}")
    print(f"[schema] triples={len(triples)} missing_field={tri_missing}")

    # 2. provenance 完整性
    prov_missing_ent = sum(1 for e in ents if not e.get("provenance"))
    prov_missing_rel = sum(1 for r in rels if not r.get("provenance"))
    print(f"[provenance] index_entries={len(prov)} missing_ent={prov_missing_ent} missing_rel={prov_missing_rel}")
    lic = {p.get("license") for p in prov.values()}
    print(f"[provenance] licenses={sorted(x for x in lic if x)}")

    # 3. unresolved
    unmapped = [r for r in rels if r.get("relation_type") == "UNMAPPED"]
    unk_type = [e for e in ents if e.get("entity_type") in ("unknown", "anatomy", "biomarker")]
    print(f"[unresolved] unmapped_relations={len(unmapped)} unknown_type_entities={len(unk_type)}")
    if unmapped:
        samples = sorted({r["relation"] for r in unmapped})[:10]
        print(f"[unresolved] unmapped predicate 样本: {samples}")

    # 4. duplicate
    eids = [e["entity_id"] for e in ents]
    rkeys = [(r["subject"], r["relation_type"], r["object"]) for r in rels]
    dup_ent = len(eids) - len(set(eids))
    dup_rel = len(rkeys) - len(set(rkeys))
    print(f"[duplicate] entities={dup_ent} relations={dup_rel}")

    # 5. quality level 分布
    from collections import Counter
    ent_lv = Counter(e.get("quality", {}).get("level") for e in ents)
    rel_lv = Counter(r.get("quality", {}).get("level") for r in rels)
    print(f"[quality] entity levels={dict(ent_lv)}")
    print(f"[quality] relation levels={dict(rel_lv)}")

    # 6. entity linking
    print(f"[linking] linked={len(linking.get('linked', []))} "
          f"candidate={len(linking.get('candidate', []))} "
          f"unmatched={len(linking.get('unmatched', []))}")

    # 7. summary 对照
    s = summary
    print(f"[summary] 输入 records: {sum(v['records_loaded'] for v in s.get('per_subset', {}).values())}")
    print(f"[summary] entities={s.get('entities')} relations={s.get('relations')} triples={s.get('triples')}")

    # 结论
    hard_fail = ent_missing + rel_missing + prov_missing_ent + prov_missing_rel + dup_ent + dup_rel
    print()
    if hard_fail == 0:
        print(f"[VALIDATE] PASS — 无 schema/provenance/duplicate 硬错误（unresolved 为如实记录，非错误）")
        return 0
    print(f"[VALIDATE] FAIL — 存在 {hard_fail} 处硬错误，需检查")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
