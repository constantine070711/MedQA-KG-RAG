"""
交互式查询脚本 — 使用模块化 RAG 引擎查询医疗知识

用法:
    cd MedQA-GitHub
    venv\Scripts\activate
    python scripts/run_query.py

    # 或直接指定问题
    python scripts/run_query.py --question "糖尿病有哪些症状?"

选项:
    --question / -q    直接指定问题（非交互模式）
    --mode / -m        检索模式: mix/local/global/hybrid/naive (默认 mix)
    --safe             启用医疗安全检查
    --info             显示引擎状态信息
"""

import os
import sys
import io
import asyncio
import argparse

# 强制 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config
from src.rag_engine import RAGEngine
from src.medical_core.safety import SafetyChecker, detect_emergency, get_emergency_response


async def run_query(
    question: str,
    mode: str = "mix",
    safe_mode: bool = False,
) -> str:
    """执行单次查询"""
    engine = RAGEngine()
    await engine.initialize()

    if safe_mode:
        # 安全检查
        checker = SafetyChecker()
        safety_result = checker.check(question)

        if detect_emergency(question):
            return get_emergency_response()

        if not safety_result.is_allowed:
            return safety_result.rejection_message

        system_prompt = checker.get_safe_system_prompt(safety_result)
        answer = await engine.query(question, mode=mode, system_prompt=system_prompt)
        return checker.format_answer(answer, safety_result)
    else:
        return await engine.query(question, mode=mode)


async def interactive(mode: str = "mix", safe_mode: bool = False):
    """交互式循环"""
    print("=" * 60)
    print("  医疗知识问答系统 (MedQA) — 交互模式")
    print("  LightRAG + DeepSeek + bge-small-zh-v1.5")
    print("=" * 60)
    print()
    print(f"  检索模式: {mode}")
    print(f"  安全模式: {'开启' if safe_mode else '关闭'}")
    print()
    print("  输入问题开始对话，输入 'quit' 退出")
    print("  输入 'mode <name>' 切换检索模式")
    print("  输入 'safe' 切换安全模式")
    print()

    while True:
        try:
            user_input = input("[Q] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT] 再见!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("[EXIT] 再见!")
            break

        if user_input.lower().startswith("mode "):
            new_mode = user_input.split()[1]
            valid_modes = {"mix", "local", "global", "hybrid", "naive"}
            if new_mode in valid_modes:
                mode = new_mode
                print(f"[MODE] 切换到: {mode}")
            else:
                print(f"[ERROR] 无效模式: {new_mode}，可选: {', '.join(valid_modes)}")
            continue

        if user_input.lower() == "safe":
            safe_mode = not safe_mode
            print(f"[SAFE] 安全模式: {'开启' if safe_mode else '关闭'}")
            continue

        print()
        answer = await run_query(user_input, mode=mode, safe_mode=safe_mode)
        print(f"[A] {answer}")
        print()


async def show_info():
    """显示引擎信息"""
    engine = RAGEngine()
    await engine.initialize()
    info = engine.get_info()
    print("[INFO] RAG Engine 状态:")
    for key, value in info.items():
        print(f"  {key}: {value}")


async def main(
    question: str = None,
    mode: str = "mix",
    safe_mode: bool = False,
    show_info_only: bool = False,
):
    """主入口"""
    if show_info_only:
        await show_info()
        return

    if question:
        # 单次查询模式
        answer = await run_query(question, mode=mode, safe_mode=safe_mode)
        print(answer)
    else:
        # 交互模式
        await interactive(mode=mode, safe_mode=safe_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="医疗知识问答查询工具")
    parser.add_argument("-q", "--question", type=str, help="直接指定问题")
    parser.add_argument("-m", "--mode", type=str, default="mix",
                        choices=["mix", "local", "global", "hybrid", "naive"],
                        help="检索模式 (默认 mix)")
    parser.add_argument("--safe", action="store_true", help="启用医疗安全检查")
    parser.add_argument("--info", action="store_true", help="显示引擎状态信息")
    args = parser.parse_args()

    asyncio.run(main(
        question=args.question,
        mode=args.mode,
        safe_mode=args.safe,
        show_info_only=args.info,
    ))
