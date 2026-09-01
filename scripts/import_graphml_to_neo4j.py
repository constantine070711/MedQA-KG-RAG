#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 F2 — 导入 LightRAG graphml 原始层到 Neo4j

职责:
    - 解析 graphml（通用 Entity 节点 + RELATED_TO 关系）
    - 确保 Entity.entity_id 唯一约束（幂等）
    - 参数化 UNWIND 批量导入（不拼接数据到 Cypher）
    - 报告导入统计 / 缺失字段 / 校验数量

用法:
    python scripts/import_graphml_to_neo4j.py
    python scripts/import_graphml_to_neo4j.py --graphml lightrag_cache/graph_chunk_entity_relation.graphml
    python scripts/import_graphml_to_neo4j.py --batch-size 500 --verbose

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
from src.graph_import import (
    parse_graphml,
    import_entity_nodes,
    import_related_edges,
    ensure_entity_id_constraint,
    count_entities,
    count_related_edges,
    node_label_distribution,
    relationship_type_distribution,
    duplicate_check,
)

DEFAULT_GRAPHML = PROJECT_ROOT / "lightrag_cache" / "graph_chunk_entity_relation.graphml"
DEFAULT_BATCH = 500


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 6 F2 — 导入 LightRAG graphml 原始层到 Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--graphml", default=str(DEFAULT_GRAPHML), help="graphml 文件路径")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="UNWIND 批量大小")
    p.add_argument("--no-verify", action="store_true", help="跳过导入后计数校验")
    p.add_argument("--verbose", action="store_true", help="详细日志")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    graphml_path = Path(args.graphml)
    if not graphml_path.exists():
        print(f"[FAIL] graphml 不存在: {graphml_path}")
        return 1

    conn = None
    try:
        # ---- 1. 解析 ----
        print("== Import LightRAG graphml → Neo4j (raw layer) ==")
        parsed = parse_graphml(graphml_path)
        st = parsed["stats"]
        print(f"源文件: {graphml_path.name}")
        print(f"源节点 {st['source_nodes']} / 解析 {st['parsed_nodes']} / 跳过 {st['skipped_nodes']}")
        print(f"源边   {st['source_edges']} / 解析 {st['parsed_edges']} / 跳过 {st['skipped_edges']}")
        print(f"节点类型分布: {st['node_type_distribution']}")
        print(f"属性完整性: {st['attribute_completeness']}")
        if parsed["skipped_nodes"]:
            print(f"  [warn] 跳过节点 {len(parsed['skipped_nodes'])}: {parsed['skipped_nodes'][:5]}")
        if parsed["skipped_edges"]:
            print(f"  [warn] 跳过边 {len(parsed['skipped_edges'])}: {parsed['skipped_edges'][:5]}")

        if not parsed["nodes"]:
            print("[FAIL] 无节点可导入")
            return 1

        # ---- 2. 连接 + 约束 ----
        conn = get_neo4j_connection()
        conn.connect()
        cst = ensure_entity_id_constraint(conn)
        if not cst["ok"]:
            print(f"[FAIL] Entity.entity_id 约束创建失败: {cst['error']}")
            return 1
        print("Entity.entity_id 唯一约束: OK")

        # ---- 3. 导入 ----
        r_nodes = import_entity_nodes(conn, parsed["nodes"], args.batch_size)
        print(f"\n节点导入: {r_nodes['ok']}/{r_nodes['total']} 成功, 错误 {len(r_nodes['errors'])}")
        if r_nodes["errors"]:
            print(f"  {r_nodes['errors'][:3]}")

        r_edges = import_related_edges(conn, parsed["edges"], args.batch_size)
        print(f"边导入:   {r_edges['ok']}/{r_edges['total']} 成功, 错误 {len(r_edges['errors'])}")
        if r_edges["errors"]:
            print(f"  {r_edges['errors'][:3]}")

        if r_nodes["errors"] or r_edges["errors"]:
            print("[FAIL] 导入存在错误")
            return 1

        # ---- 4. 校验 ----
        if not args.no_verify:
            print("\n-- 校验 --")
            print(f"Neo4j Entity 节点数: {count_entities(conn, 'Entity')} (源 {st['source_nodes']})")
            print(f"Neo4j RELATED_TO 关系数: {count_related_edges(conn)} (源 {st['source_edges']})")
            print(f"节点标签分布: {node_label_distribution(conn)}")
            print(f"关系类型分布: {relationship_type_distribution(conn)}")
            dup = duplicate_check(conn, "Entity", "entity_id")
            print(f"Entity.entity_id 重复检查: {len(dup)} 处重复 {dup[:5]}")

        print("\n[OK] graphml 原始层导入完成。")
        return 0

    except Exception as exc:
        print(f"[FAIL] 导入失败: {type(exc).__name__}: {exc}")
        return 1

    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
