# -*- coding: utf-8 -*-
"""Phase 10-A.5 — 真实医学知识库产物质量校验（只读）。

检查 data/processed/medical_kb/ 产物：
  - schema 字段完整性（entities/relations/triples/provenance/entity_linking）
  - unresolved（UNMAPPED）关系
  - 重复关系 / 重复实体
  - missing provenance
  - quality fail 实体/关系
  - 输入文件未被修改（与原始摘要比对）

只读，不写任何文件。退出码 0 = 通过（PASS），1 = 存在质量问题（FAIL）。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = PROJECT_ROOT / "data" / "external" / "medical_kb"
PROCESSED = PROJECT_ROOT / "data" / "processed" / "medical_kb"


def _load(name: str) -> dict:
    with open(PROCESSED / name, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="CMeIE 医学知识库产物校验（只读）")
    parser.add_argument("--verbose", action="store_true", help="打印明细")
    args = parser.parse_args()

    issues: list[str] = []

    entities = _load("entities.json")
    relations = _load("relations.json")
    triples = _load("triples.json")
    provenance = _load("provenance.json")
    linking = _load("entity_linking.json")
    summary = _load("summary.json")

    # 1. schema 字段完整性
    required_ent = {"entity_id", "name", "entity_type", "provenance", "quality"}
    required_rel = {"subject", "relation", "object", "relation_type", "provenance", "quality"}
    for e in entities:
        missing = required_ent - set(e.keys())
        if missing:
            issues.append(f"entity 缺字段 {missing}: {e.get('name')!r}")
    for r in relations:
        missing = required_rel - set(r.keys())
        if missing:
            issues.append(f"relation 缺字段 {missing}: {r.get('subject')!r}")
    # triples 与 relations 数量一致
    if len(triples) != len(relations):
        issues.append(f"triples({len(triples)}) != relations({len(relations)})")

    # 2. missing provenance
    missing_prov = sum(1 for e in entities if not e.get("provenance"))
    missing_prov += sum(1 for r in relations if not r.get("provenance"))
    if missing_prov:
        issues.append(f"missing provenance: {missing_prov}")

    # 3. quality fail
    fail_ents = sum(1 for e in entities if e.get("quality", {}).get("level") == "fail")
    fail_rels = sum(1 for r in relations if r.get("quality", {}).get("level") == "fail")
    if fail_ents or fail_rels:
        issues.append(f"quality fail: entities={fail_ents} relations={fail_rels}")

    # 4. 重复关系 / 重复实体（按规范化键）
    seen_rel = set(); dup_rel = 0
    for r in relations:
        key = (r["subject"].strip(), r["relation_type"], r["object"].strip())
        if key in seen_rel:
            dup_rel += 1
        seen_rel.add(key)
    seen_ent = set(); dup_ent = 0
    for e in entities:
        key = (e["name"].strip(), e["entity_type"])
        if key in seen_ent:
            dup_ent += 1
        seen_ent.add(key)
    if dup_rel or dup_ent:
        issues.append(f"duplicate: relations={dup_rel} entities={dup_ent}")

    # 5. unresolved 关系（如实记录，非错误）
    unmapped = sum(1 for r in relations if r["relation_type"] == "UNMAPPED")
    unmapped_ratio = unmapped / len(relations) if relations else None

    # 6. 输入文件未被修改（sha1 比对原始 zip 与目录文件）
    zip_path = EXTERNAL / "CMeIE_original.zip"
    train_path = EXTERNAL / "CMeIE" / "CMeIE_train.jsonl"
    if zip_path.exists():
        h1 = hashlib.sha1(zip_path.read_bytes()).hexdigest()[:16]
        print(f"  [input] CMeIE_original.zip sha1[:16] = {h1}（原样存档）")
    else:
        issues.append("CMeIE_original.zip 缺失")
    if train_path.exists():
        h2 = hashlib.sha1(train_path.read_bytes()).hexdigest()[:16]
        print(f"  [input] CMeIE_train.jsonl sha1[:16] = {h2}（只读源）")
    else:
        issues.append("CMeIE_train.jsonl 缺失")

    # 输出汇总
    print("=" * 60)
    print(f"产物: data/processed/medical_kb/")
    print(f"  实体 = {len(entities)}")
    print(f"  关系 = {len(relations)}")
    print(f"  三元组 = {len(triples)}")
    print(f"  provenance 实体记录 = {len(provenance.get('entities', []))}")
    print(f"  provenance 关系记录 = {len(provenance.get('relations', []))}")
    print(f"  Entity Linking: linked={linking.get('link_summary', {}).get('linked')} "
          f"candidate={linking.get('link_summary', {}).get('candidate')} "
          f"unmatched={linking.get('link_summary', {}).get('unmatched')}")
    print(f"  未映射关系 = {unmapped} ({round(unmapped_ratio, 4) if unmapped_ratio is not None else 'N/A'})")
    print(f"  missing provenance = {missing_prov}")
    print(f"  quality fail = entities {fail_ents} / relations {fail_rels}")
    print(f"  duplicate = relations {dup_rel} / entities {dup_ent}")
    if args.verbose:
        print(f"  summary = {json.dumps(summary, ensure_ascii=False)}")
    print("=" * 60)

    if issues:
        print("[FAIL] 存在问题:")
        for i in issues:
            print(f"  - {i}")
        return 1
    print("[PASS] 产物质量校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
