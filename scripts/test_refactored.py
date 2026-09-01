"""
Phase 2 重构后闭环验证脚本

使用模块化 src/ 代码，验证 MVP 的 4 个医疗问答仍然通过。

与 Phase 1 test_mvp.py 的区别:
    - 使用 src/ 模块代替内联代码
    - 增加 medical_core 安全检查
    - 更清晰的输出格式

用法:
    cd MedQA-GitHub
    venv\Scripts\activate
    python scripts/test_refactored.py
"""

import os
import sys
import io
import asyncio

# 强制 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config
from src.data_loader import DataLoader
from src.rag_engine import RAGEngine
from src.medical_core.safety import SafetyChecker, MEDICAL_DISCLAIMER


# ============================================================
# 测试问题（与 Phase 1 相同）
# ============================================================
TEST_QUESTIONS = [
    {
        "question": "2型糖尿病有哪些常见症状？",
        "expected_keywords": ["多饮", "多尿", "多食", "体重下降", "三多一少"],
    },
    {
        "question": "高血压的治疗方法是什么？",
        "expected_keywords": ["生活方式", "药物", "ACEI", "钙通道阻滞剂"],
    },
    {
        "question": "二甲双胍有什么不良反应？",
        "expected_keywords": ["胃肠道", "乳酸", "二甲双胍"],
    },
    {
        "question": "感冒了应该怎么治疗？",
        "expected_keywords": ["对症", "解热镇痛", "抗生素", "休息"],
    },
]


async def main():
    """主验证流程"""
    print("=" * 60)
    print("  Phase 2 模块化重构 — 闭环验证测试")
    print("  LightRAG + DeepSeek + bge-small-zh-v1.5")
    print("=" * 60)
    print()

    # --- Step 1: 配置验证 ---
    print("[STEP 1/5] 验证配置...")
    cfg = get_config()

    errors = cfg.validate()
    if errors:
        print("[ERROR] 配置验证失败:")
        for e in errors:
            print(f"  - {e}")
        return False

    print(cfg.summary())
    print("   [OK] 配置验证通过")
    print()

    # --- Step 2: 数据检查 ---
    print("[STEP 2/5] 检查医疗数据...")
    loader = DataLoader()
    documents = loader.load_all()

    if not documents:
        print("[ERROR] data/raw/ 目录下没有有效文档")
        return False

    print(f"   发现 {len(documents)} 个文档, 共 {sum(d.char_count for d in documents)} 字符:")
    for doc in documents:
        source_info = f" (来源: {doc.metadata.source})" if doc.metadata.source else ""
        entity_type = f"[{doc.entity_type}]" if doc.entity_type else ""
        print(f"     - {entity_type} {doc.entity_name}: {doc.char_count} 字符{source_info}")
    print()

    # --- Step 3: 初始化 RAG 引擎 ---
    print("[STEP 3/5] 初始化 RAG 引擎...")
    engine = RAGEngine()
    await engine.initialize()

    cache_info = engine.get_cache_info()
    if cache_info['file_count'] > 0:
        print(f"   缓存状态: {cache_info['file_count']} 个文件 ({cache_info['total_size_mb']} MB)")
        print("   使用现有缓存（跳过重复索引）")
    else:
        print("   缓存为空，开始索引文档...")
        await engine.insert_documents([d.content for d in documents])
        print(f"   索引完成: {len(documents)} 个文档")

    print("   [OK] RAG 引擎就绪")
    print()

    # --- Step 4: 安全检查器 ---
    print("[STEP 4/5] 初始化安全检查器...")
    checker = SafetyChecker()
    print("   [OK] 安全检查器就绪")
    print()

    # --- Step 5: 测试问答 ---
    print("[STEP 5/5] 执行测试问答 (mix 模式)...")
    print("=" * 60)

    results = []
    for i, test in enumerate(TEST_QUESTIONS, 1):
        q = test["question"]
        expected = test["expected_keywords"]

        # 安全检查
        safety = checker.check(q)
        safety_label = {
            "safe": "✓",
            "caution": "⚠",
            "restricted": "🔒",
            "blocked": "🚫",
        }.get(safety.level.value, "?")

        print(f"\n{'─' * 50}")
        print(f"[TEST {i}/4] {safety_label} {q}")
        print(f"{'─' * 50}")

        # 使用带医疗安全 prompt 的查询
        system_prompt = checker.get_safe_system_prompt(safety)
        try:
            answer = await engine.query(q, mode="mix", system_prompt=system_prompt)
        except Exception as e:
            print(f"[ERROR] 查询失败: {e}")
            answer = ""

        if not answer:
            print("[ERROR] 未能获取回答（API 可能出错）")
            results.append({
                "question": q,
                "coverage": 0,
                "passed": False,
                "safety_level": safety.level.value,
            })
            continue

        # 检查关键词覆盖
        found_keywords = [kw for kw in expected if kw in answer]
        coverage = len(found_keywords) / len(expected) * 100 if expected else 100

        print(f"[ANSWER] {answer[:300]}{'...' if len(answer) > 300 else ''}")
        print()
        print(f"   预期关键词: {', '.join(expected)}")
        print(f"   命中关键词: {', '.join(found_keywords) if found_keywords else '无'}")
        print(f"   覆盖率: {coverage:.0f}% ({len(found_keywords)}/{len(expected)})")

        results.append({
            "question": q,
            "coverage": coverage,
            "passed": coverage >= 50,  # 至少 50% 关键词命中
            "safety_level": safety.level.value,
        })

    # --- 汇总 ---
    print()
    print("=" * 60)
    print("[SUMMARY] 测试汇总")
    print("=" * 60)

    all_passed = all(r["passed"] for r in results)
    for i, r in enumerate(results, 1):
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"  [{i}/4] {status} — {r['question']} (覆盖率: {r['coverage']:.0f}%)")

    print()
    if all_passed:
        print("[OK] 全部 4/4 测试通过! Phase 2 模块化重构验证成功。")
    else:
        failed = sum(1 for r in results if not r["passed"])
        print(f"[WARN] {failed}/4 测试未通过，请检查。")

    print()
    print(f"  缓存目录: {engine.working_dir}")
    print(f"  下次运行将复用缓存，如需重建请删除该目录")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
