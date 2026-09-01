"""
批量索引脚本 (Phase 3) — 将 data/raw/ 下所有医疗文档导入 LightRAG

支持 Phase 3 目录结构:
    data/raw/
    ├── diseases/     — 疾病知识 (.md)
    ├── drugs/        — 药物知识 (.md)
    ├── symptoms/     — 症状知识 (预留)
    └── guidelines/   — 临床指南 (预留)

用法:
    cd MedQA-GitHub
    venv\Scripts\activate
    python scripts/index_docs.py

选项:
    --clear      清除现有缓存后重新索引
    --stats      仅显示数据目录统计，不索引
    --metadata   导出元数据索引到 data/metadata/index.json
    --type TYPE  仅索引指定类型 (diseases/drugs/symptoms/guidelines)
"""

import os
import sys
import io
import asyncio
import argparse

# 强制 UTF-8 输出（Windows GBK 兼容）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config
from src.data_loader import DataLoader
from src.rag_engine import RAGEngine


async def main(
    clear_cache: bool = False,
    stats_only: bool = False,
    export_metadata: bool = False,
    entity_type: str = None,
):
    """主流程"""
    cfg = get_config()

    # 验证配置
    errors = cfg.validate()
    if errors:
        print("[ERROR] 配置验证失败:")
        for e in errors:
            print(f"  - {e}")
        return

    print(cfg.summary())
    print()

    # --- 数据加载和统计 ---
    loader = DataLoader()

    # 按类型加载
    entity_types = [entity_type] if entity_type else None
    documents = loader.load_all(entity_types=entity_types)
    stats = loader.get_stats()

    print(f"[DATA] 数据目录: {stats['data_dir']}")
    print(f"   文档总数: {stats['total_documents']}")
    print(f"   总字符数: {stats['total_characters']}")
    print(f"   元数据记录: {stats['metadata_records']}")
    print()

    if stats["by_entity_type"]:
        print("[DATA] 按实体类型分布:")
        for etype, count in stats["by_entity_type"].items():
            print(f"   {etype}: {count} 个文档")
        print()

    if stats["by_format"]:
        print("[DATA] 按格式分布:")
        for fmt, count in stats["by_format"].items():
            print(f"   {fmt}: {count} 个文档")
        print()

    if stats_only:
        print("[DATA] 文档列表:")
        for doc in stats["documents"]:
            print(f"   [{doc['entity_type'] or '?'}] {doc['entity_name']}"
                  f" — {doc['name']} ({doc['char_count']} 字符)")
            if doc["source"]:
                print(f"     来源: {doc['source']}")
            if doc["license"]:
                print(f"     许可证: {doc['license']}")
        return

    # --- 导出元数据 ---
    if export_metadata:
        metadata_path = str(project_root / "data" / "metadata" / "index.json")
        loader.build_metadata_index(output_path=metadata_path)
        print(f"[METADATA] 元数据索引已导出: {metadata_path}")
        print()

    if not documents:
        print("[ERROR] 没有找到可索引的文档")
        return

    # --- 初始化引擎 ---
    engine = RAGEngine()

    if clear_cache:
        print("[CACHE] 清除现有缓存...")
        engine.clear_cache()

    await engine.initialize()

    cache_info = engine.get_cache_info()
    if cache_info['file_count'] > 0:
        print(f"[CACHE] 已有缓存: {cache_info['file_count']} 个文件 "
              f"({cache_info['total_size_mb']} MB)")
        print(f"[CACHE] 将增量添加新文档（已索引的不会重复）")
    else:
        print("[CACHE] 缓存为空，将进行首次索引")
    print()

    # --- 索引文档 ---
    print(f"[INDEX] 开始索引 {len(documents)} 个文档...")
    print()

    for i, doc in enumerate(documents, 1):
        type_label = f"[{doc.entity_type}]" if doc.entity_type else ""
        print(f"  [{i}/{len(documents)}] {type_label} {doc.entity_name} "
              f"({doc.char_count} 字符)...")
        await engine.insert_document(doc.content)
        print(f"     [OK] 索引完成")

    print()
    print(f"[OK] 全部索引完成! 共处理 {len(documents)} 个文档")

    # --- 索引后统计 ---
    cache_info = engine.get_cache_info()
    print(f"[CACHE] 缓存状态: {cache_info['file_count']} 个文件 "
          f"({cache_info['total_size_mb']} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量索引医疗文档到 LightRAG (Phase 3)")
    parser.add_argument("--clear", action="store_true", help="清除现有缓存后重新索引")
    parser.add_argument("--stats", action="store_true", help="仅显示数据目录统计")
    parser.add_argument("--metadata", action="store_true", help="导出元数据索引到 data/metadata/index.json")
    parser.add_argument("--type", type=str, choices=["diseases", "drugs", "symptoms", "guidelines"],
                        help="仅索引指定类型的文档")
    args = parser.parse_args()

    asyncio.run(main(
        clear_cache=args.clear,
        stats_only=args.stats,
        export_metadata=args.metadata,
        entity_type=args.type,
    ))
