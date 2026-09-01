#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 F3 — curated 临床关系重建 + 写入 Neo4j

职责:
    - 解析 curated .md 结构化章节 → 候选临床关系（D4 解析两端）
    - 参数化 UNWIND MERGE 写入（幂等）
    - 报告: 候选/解析/unresolved/占位符跳过/自环跳过/实际写入/按类型分布
    - 一致性: 关系端点存在性 / 占位符检查 / 重复检查 / F2 原始层不受影响

用法:
    python scripts/rebuild_curated_relations.py
    python scripts/rebuild_curated_relations.py --no-verify
    python scripts/rebuild_curated_relations.py --verbose

退出码:
    0 成功 / 1 失败
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.neo4j_utils import get_neo4j_connection
from src.relation_rebuild import (
    parse_curated_relations,
    import_relations,
    count_relations_by_type,
)
from src.graph_import import (
    node_label_distribution,
    relationship_type_distribution,
    duplicate_check,
)

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_ENTITIES = PROJECT_ROOT / "data" / "entities" / "entities.json"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 6 F3 — curated 临床关系重建 + 写入 Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="curated .md 目录")
    p.add_argument("--entities", default=str(DEFAULT_ENTITIES), help="entities.json 路径")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--no-verify", action="store_true", help="跳过导入后校验")
    p.add_argument("--verbose", action="store_true", help="详细日志")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    conn = None
    try:
        print("== Rebuild curated clinical relations → Neo4j ==")
        parsed = parse_curated_relations(args.raw_dir, args.entities)
        st = parsed["stats"]
        print(f"源 .md 文件: {st['source_md_files']}")
        print(f"候选关系(含重复来源): {st['candidate_relations']}")
        print(f"解析成功(去重后): {st['resolved_relations']}")
        print(f"unresolved 提及: {st['unresolved_mentions']}")
        print(f"占位符跳过: {st['placeholder_skipped']}")
        print(f"自环跳过: {st['self_skipped']}")
        print(f"D4 解析成功/失败: {st['d4_resolve_success']}/{st['d4_resolve_fail']}")
        print(f"按关系类型: {st['by_rel_type']}")

        if parsed["self_skipped"]:
            print(f"  [warn] 自环剔除: {parsed['self_skipped']}")
        if parsed["placeholder_skipped"]:
            print(f"  [warn] 占位符跳过: {parsed['placeholder_skipped']}")

        if not parsed["relations"]:
            print("[warn] 无关系可写入（如实报告：数据仅支持少量关系）")
            return 0

        # ---- 写入 ----
        conn = get_neo4j_connection()
        conn.connect()
        results = import_relations(conn, parsed["relations"], args.batch_size)
        total_ok = 0
        for rel_type, res in results.items():
            print(f"写入 {rel_type}: {res['ok']}/{res['total']}, 错误 {len(res['errors'])}")
            total_ok += res["ok"]
            if res["errors"]:
                print(f"  {res['errors'][:3]}")
        print(f"实际写入关系: {total_ok}")

        if any(res["errors"] for res in results.values()):
            print("[FAIL] 写入存在错误")
            return 1

        # ---- 校验 ----
        if not args.no_verify:
            print("\n-- 校验 --")
            print(f"节点标签分布: {node_label_distribution(conn)}")
            print(f"关系类型分布: {relationship_type_distribution(conn)}")
            for rel_type, res in results.items():
                n = count_relations_by_type(conn, rel_type)
                print(f"  {rel_type}: {n} 条 (本次写入 {res['ok']})")
            dup = conn.execute(
                "MATCH (a)-[r]->(b) WITH type(r) AS t, a.name AS s, b.name AS o, count(*) AS c "
                "WHERE c > 1 RETURN t, s, o, c"
            )
            print(f"重复关系检查: {len(dup)} 处 {dup[:5]}")

        print("\n[OK] curated 临床关系重建完成。")
        return 0

    except Exception as exc:
        print(f"[FAIL] 关系重建失败: {type(exc).__name__}: {exc}")
        return 1

    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
