#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 F1 — Neo4j Schema bootstrap

职责:
    - 从项目 config（.env）读取 Neo4j 连接信息
    - 连接 Neo4j（Bolt）
    - 执行现有 Schema DDL（Neo4jSchema.bootstrap(): 约束 + 索引，幂等）
    - 校验实际创建的约束/索引（SHOW CONSTRAINTS / SHOW INDEXES）
    - 输出清晰的成功/失败信息
    - 正常关闭连接

约束（F1）:
    - 不导入 graphml
    - 不导入 curated entities
    - 不创建关系
    - 不实现 GraphRetriever
    - 不改动现有 Schema 定义

用法:
    python scripts/neo4j_bootstrap.py            # 默认连接 + bootstrap + 校验
    python scripts/neo4j_bootstrap.py --no-verify # 跳过 SHOW 校验
    python scripts/neo4j_bootstrap.py --verbose   # 详细日志

退出码:
    0 全部成功 / 1 连接或 DDL 失败
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

# 强制 UTF-8 输出 (Windows GBK 兼容)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.neo4j_utils import get_neo4j_connection, Neo4jSchema

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 6 F1 — Neo4j Schema bootstrap（约束 + 索引，幂等）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--no-verify", action="store_true",
                   help="跳过 SHOW CONSTRAINTS / SHOW INDEXES 校验")
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
        # ---- 1. 连接 ----
        print("== Neo4j Schema bootstrap ==")
        conn = get_neo4j_connection()
        conn.connect()
        print(f"连接成功: {conn.uri} (user={conn.user})")

        # ---- 2. 执行 Schema DDL ----
        schema = Neo4jSchema()
        result = schema.bootstrap(conn)
        print(f"\n-- Schema DDL --")
        print(f"约束: {len(result['constraints'])} 条, "
              f"索引: {len(result['indexes'])} 条, 失败: {result['failed']}")
        for c in result["constraints"]:
            print(f"  constraint {c['name']}: {'OK' if c['ok'] else 'FAIL ' + (c['error'] or '')}")
        for i in result["indexes"]:
            print(f"  index {i['name']}: {'OK' if i['ok'] else 'FAIL ' + (i['error'] or '')}")

        if result["failed"] > 0:
            print("\n[FAIL] 部分 Schema DDL 执行失败，请检查上述错误。")
            return 1

        # ---- 3. 校验（可选） ----
        if not args.no_verify:
            print("\n-- Schema 校验 (SHOW) --")
            verify = schema.verify_schema(conn)
            print(f"约束 ({len(verify['constraints']) if isinstance(verify['constraints'], list) else 'err'}):")
            for c in verify["constraints"] if isinstance(verify["constraints"], list) else []:
                print(f"  {c.get('name')}  [{c.get('type')}] on {c.get('entityType')}")
            print(f"索引 ({len(verify['indexes']) if isinstance(verify['indexes'], list) else 'err'}):")
            for i in verify["indexes"] if isinstance(verify["indexes"], list) else []:
                print(f"  {i.get('name')}  [{i.get('type')}] on {i.get('labelsOrTypes')}")

        print("\n[OK] Neo4j Schema bootstrap 完成。")
        return 0

    except Exception as exc:
        print(f"\n[FAIL] Neo4j Schema bootstrap 失败: {type(exc).__name__}: {exc}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return 1

    finally:
        if conn is not None:
            conn.close()
            print("连接已关闭。")


if __name__ == "__main__":
    sys.exit(main())
