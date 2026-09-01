#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 F2 — 导入 curated 临床层到 Neo4j

职责:
    - 解析 data/entities/entities.json（14 curated 实体）
    - 按 entity_type 映射到 Disease/Drug 等标签（受控映射）
    - 参数化 UNWIND 批量导入（不拼接数据到 Cypher）
    - **不导入占位符关系**（object="见正文…章节"）；真实关系重建属 F3
    - 报告类型分布 / 标准编码覆盖 / 关系缺口

用法:
    python scripts/import_curated_to_neo4j.py
    python scripts/import_curated_to_neo4j.py --entities data/entities/entities.json
    python scripts/import_curated_to_neo4j.py --verbose

退出码:
    0 成功 / 1 失败
"""

import argparse
import logging
import sys
from collections import OrderedDict
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
    parse_curated,
    import_curated_nodes,
    count_entities,
    node_label_distribution,
    duplicate_check,
)

DEFAULT_ENTITIES = PROJECT_ROOT / "data" / "entities" / "entities.json"
DEFAULT_BATCH = 200


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 6 F2 — 导入 curated 临床层到 Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--entities", default=str(DEFAULT_ENTITIES), help="entities.json 路径")
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

    entities_path = Path(args.entities)
    if not entities_path.exists():
        print(f"[FAIL] entities.json 不存在: {entities_path}")
        return 1

    conn = None
    try:
        # ---- 1. 解析 ----
        print("== Import curated 临床层 → Neo4j ==")
        parsed = parse_curated(entities_path)
        st = parsed["stats"]
        print(f"源文件: {entities_path.name}")
        print(f"源实体 {st['source_entities']} / 可导入 {st['imported_nodes']} / 跳过 {st['skipped_entities']}")
        print(f"类型分布: {st['node_type_distribution']}")
        print(f"标准编码覆盖: {st['standard_code_coverage']}/{st['imported_nodes']}")
        print(f"别名覆盖: {st['alias_coverage']}/{st['imported_nodes']}")
        if parsed["skipped"]:
            print(f"  [warn] 跳过实体: {parsed['skipped'][:5]}")
        print(f"源关系 {st['source_relations']} 条, 其中占位符 {st['placeholder_relations']} 条")
        print(f"  [info] F2 不导入 curated 关系（全部为占位符）；真实关系重建属 F3 → 导入关系 0")

        if not parsed["nodes"]:
            print("[FAIL] 无实体可导入")
            return 1

        # ---- 2. 连接 ----
        conn = get_neo4j_connection()
        conn.connect()

        # ---- 3. 按标签分组导入 ----
        by_label: "OrderedDict[str, list]" = OrderedDict()
        for n in parsed["nodes"]:
            by_label.setdefault(n["label"], []).append(n)

        total_ok = 0
        all_errors = []
        for label, items in by_label.items():
            r = import_curated_nodes(conn, label, items, args.batch_size)
            print(f"{label} 节点导入: {r['ok']}/{r['total']} 成功, 错误 {len(r['errors'])}")
            total_ok += r["ok"]
            all_errors.extend(r["errors"])

        if all_errors:
            print(f"[FAIL] 导入存在错误: {all_errors[:3]}")
            return 1

        # ---- 4. 校验 ----
        if not args.no_verify:
            print("\n-- 校验 --")
            print(f"节点标签分布: {node_label_distribution(conn)}")
            for label in by_label:
                n = count_entities(conn, label)
                print(f"  {label}: {n} 个 (期望 {len(by_label[label])})")
                dup = duplicate_check(conn, label, "name")
                print(f"  {label}.name 重复检查: {len(dup)} 处重复 {dup[:3]}")

        print("\n[OK] curated 临床层导入完成。")
        return 0

    except Exception as exc:
        print(f"[FAIL] 导入失败: {type(exc).__name__}: {exc}")
        return 1

    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
