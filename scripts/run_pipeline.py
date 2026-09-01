"""
Phase 4 数据管道执行脚本

运行完整的数据导入流程，将 data/raw/ 下的 14 个 .md 文件
通过 Adapter → Transformer → Validator → Writer 转换为标准化实体。

流程:
    1. MarkdownAdapter 解析 14 个 .md → RawEntity dicts
    2. TextCleaner 清洗文本
    3. EntityNormalizer 标准化 → UnifiedEntity
    4. EntityDeduplicator 去重
    5. DataValidator 质量校验
    6. EntityWriter → EntityRegistry + entities.json
    7. MetadataWriter → metadata/pipeline_index.json

输出:
    data/entities/entities.json    — 标准化实体 + 关系数据
    data/entities/relations.csv    — 关系 CSV (Neo4j 导入用)
    data/metadata/pipeline_index.json — Pipeline 元数据索引

用法:
    cd MedQA-GitHub
    venv\Scripts\activate
    python scripts/run_pipeline.py

选项:
    --source TYPE    数据源类型 (默认: markdown)
    --output DIR     输出目录 (默认: data/entities)
    --stats          仅显示统计信息，不运行 Pipeline
    --verbose        详细日志
"""

import os
import sys
import io
import argparse
import logging

# 强制 UTF-8 输出（Windows GBK 兼容）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.orchestrator import PipelineOrchestrator


def main(
    source: str = "markdown",
    output_dir: str = None,
    verbose: bool = False,
):
    """运行数据管道"""
    print("=" * 60)
    print("  Phase 4 医疗知识数据管道 (Medical Knowledge Pipeline)")
    print(f"  数据源: {source}")
    print("=" * 60)
    print()

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s: %(message)s",
        )

    # 运行 Pipeline
    orch = PipelineOrchestrator(project_root=str(project_root))
    report = orch.run(
        source=source,
        output_dir=output_dir,
    )

    # 输出报告
    print()
    print("=" * 60)
    print("  Pipeline 运行报告")
    print("=" * 60)
    print()
    print(report.summary())
    print()

    # 输出详情
    if report.total_input > 0:
        print("实体类型分布:")
        for etype, count in sorted(report.by_entity_type.items()):
            print(f"  {etype}: {count}")

        print()
        pass_rate = report.total_valid / max(report.total_input, 1) * 100
        print(f"通过率: {pass_rate:.0f}% ({report.total_valid}/{report.total_input})")
        if report.total_duplicates_merged > 0:
            print(f"合并重复: {report.total_duplicates_merged} 组")
        if report.total_invalid > 0:
            print(f"⚠ 未通过校验: {report.total_invalid} 个实体")

    # 质量问题
    if report.quality_issues:
        print()
        print(f"质量问题 ({len(report.quality_issues)} 条):")
        for issue in report.quality_issues[:15]:
            print(f"  - {issue}")
        if len(report.quality_issues) > 15:
            print(f"  ... 共 {len(report.quality_issues)} 条")

    # 输出文件
    print()
    print("输出文件:")
    entities_json = project_root / "data" / "entities" / "entities.json"
    relations_csv = project_root / "data" / "entities" / "relations.csv"
    metadata_json = project_root / "data" / "metadata" / "pipeline_index.json"

    for p in [entities_json, relations_csv, metadata_json]:
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  ✅ {p} ({size_kb:.1f} KB)")
        else:
            print(f"  ❌ {p} (不存在)")

    print()
    print("=" * 60)
    print("  Pipeline 执行完成")
    print("=" * 60)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4 医疗知识数据管道"
    )
    parser.add_argument(
        "--source", type=str, default="markdown",
        help="数据源类型 (默认: markdown)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出目录 (默认: data/entities)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="仅显示统计信息（使用现有的 DataLoader）",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="详细日志",
    )
    args = parser.parse_args()

    if args.stats:
        # 仅显示统计
        from src.data_loader import DataLoader
        loader = DataLoader()
        stats = loader.get_stats()
        print(f"数据目录: {stats['data_dir']}")
        print(f"文档总数: {stats['total_documents']}")
        print(f"总字符数: {stats['total_characters']}")
        print(f"元数据记录: {stats['metadata_records']}")
        print()
        print("按实体类型:")
        for etype, count in stats["by_entity_type"].items():
            print(f"  {etype}: {count}")
        print()
        print("文档列表:")
        for doc in stats["documents"]:
            print(f"  [{doc['entity_type']}] {doc['entity_name']} ({doc['char_count']} 字符)")
    else:
        main(
            source=args.source,
            output_dir=args.output,
            verbose=args.verbose,
        )
