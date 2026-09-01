#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 5A A7 — Evaluation CLI (统一命令行入口)

串联 A4/A5 (EvaluationRunner) + A6 (ModeComparator) 的单一入口。
不重新实现 M1-M7 / refusal_accuracy / 模式对比，全部复用现有模块:
    - evaluation.runner.EvaluationRunner      (A4/A5 单模式评估)
    - evaluation.mode_comparator.ModeComparator (A6 四模式对比)
    - evaluation.metrics                      (M1-M7 指标)

用法:
    python scripts/eval_retrieval.py --help
    python scripts/eval_retrieval.py --compare-modes --output data/eval/modes.json
    python scripts/eval_retrieval.py --full --mode mix --no-llm-judge
    python scripts/eval_retrieval.py --refusal-only
    python scripts/eval_retrieval.py --mode naive

动作 (互斥, 只能指定一个):
    --full            完整评估: 单模式 baseline (A4/A5) + 四模式对比 (A6)
    --compare-modes   只执行 A6 四模式检索对比 (naive/local/global/mix × 28, 不调用 LLM Judge)
    --refusal-only    只执行拒答相关评估 (M7, 不调用 LLM Judge)
    --mode <m>        单模式评估 (naive/local/global/mix); 无动作 flag 时单独使用

LLM Judge (A5):
    --full / --mode 默认启用 LLM-as-judge; --no-llm-judge 可禁用以节省 API 费用。
    --compare-modes / --refusal-only 永不调用 LLM Judge。

输出:
    默认输出目录 data/eval/ (不存在则自动创建)。
    各动作生成固定主报告文件:
        full         data/eval/full_report.json
        compare      data/eval/mode_comparison.json
        refusal      data/eval/refusal_report.json
        single       data/eval/single_mode_<mode>.json
    --output <PATH> 覆盖主报告路径; 若为目录则写入该目录下的默认文件名。
    组件产物 (phase5a_a4_results.json / phase5a_a4_summary.json /
              phase5a_a6_mode_comparison.json) 始终写入输出目录,
    不会覆盖 evaluation/reports/ 下已有的 A4/A5/A6 报告。

退出码:
    0 成功 / 2 参数错误 (argparse) / 1 运行时致命错误
    单题失败由底层 EvaluationRunner / ModeComparator 自行记录, 不会静默吞掉。
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import traceback
from pathlib import Path

# 强制 UTF-8 输出 (Windows GBK 兼容)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"
MODES = ["naive", "local", "global", "mix"]

logger = logging.getLogger(__name__)


# ============================================================
# argparse
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 5A A7 — Evaluation CLI (统一命令行入口)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/eval_retrieval.py --compare-modes --output data/eval/modes.json\n"
            "  python scripts/eval_retrieval.py --full --mode mix --no-llm-judge\n"
            "  python scripts/eval_retrieval.py --refusal-only\n"
            "  python scripts/eval_retrieval.py --mode naive"
        ),
    )

    # ---- 动作 (互斥) ----
    p.add_argument(
        "--full",
        action="store_true",
        help="完整评估: 单模式 baseline (A4/A5) + 四模式对比 (A6)",
    )
    p.add_argument(
        "--compare-modes",
        action="store_true",
        help="只执行 A6 四模式检索对比 (naive/local/global/mix, 不调用 LLM Judge)",
    )
    p.add_argument(
        "--refusal-only",
        action="store_true",
        help="只执行拒答相关评估 (M7, 不调用 LLM Judge)",
    )

    # ---- 参数 ----
    p.add_argument(
        "--mode",
        choices=MODES,
        default=None,
        help="检索模式 (默认 mix); 无动作 flag 时单独使用表示单模式评估",
    )
    p.add_argument(
        "--output",
        default=None,
        help="主报告输出文件路径; 若为目录则写入其下默认文件名 (默认 data/eval/)",
    )
    p.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="禁用 LLM-as-judge (A5), 节省 API 费用",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="详细日志输出",
    )
    return p


def _resolve_action(args, parser: argparse.ArgumentParser) -> str:
    """
    解析主动作。

    Returns:
        动作字符串: "full" / "compare_modes" / "refusal_only" / "single"
    """
    flags = [
        f for f in ("full", "compare_modes", "refusal_only")
        if getattr(args, f)
    ]
    if len(flags) > 1:
        names = {
            "full": "--full",
            "compare_modes": "--compare-modes",
            "refusal_only": "--refusal-only",
        }
        parser.error("动作冲突: 只能指定一个 " + " / ".join(names[f] for f in flags))

    if flags:
        return flags[0]

    # 无动作 flag
    if args.mode is not None:
        return "single"

    # 完全无参数 → 打印 help
    parser.print_help()
    parser.exit(0)


def _output_paths(args, action: str) -> tuple[Path, Path]:
    """解析输出目录与主报告文件路径"""
    default_names = {
        "full": "full_report.json",
        "compare_modes": "mode_comparison.json",
        "refusal_only": "refusal_report.json",
        "single": f"single_mode_{args.mode or 'mix'}.json",
    }
    default_name = default_names[action]

    if args.output:
        op = Path(args.output)
        if op.suffix.lower() == ".json":
            output_dir = op.parent
            main_output = op
        else:
            output_dir = op
            main_output = op / default_name
    else:
        output_dir = DEFAULT_OUTPUT_DIR
        main_output = DEFAULT_OUTPUT_DIR / default_name

    output_dir.mkdir(parents=True, exist_ok=True)
    main_output.parent.mkdir(parents=True, exist_ok=True)
    return output_dir, main_output


# ============================================================
# 组件执行 (全部复用现有模块)
# ============================================================

async def _run_runner(mode: str, enable_llm_judge: bool, output_dir: Path) -> dict:
    """复用 EvaluationRunner 执行单模式评估, 返回 runner.summary"""
    from evaluation.runner import EvaluationRunner

    runner = EvaluationRunner(
        mode=mode,
        output_dir=str(output_dir),
        enable_llm_judge=enable_llm_judge,
    )
    await runner.initialize()
    await runner.run()
    results_path, summary_path = runner.generate_reports()
    logger.info(
        f"runner 产物: {results_path.name} / {summary_path.name} "
        f"(mode={mode}, llm_judge={enable_llm_judge})"
    )
    return runner.summary


async def _run_compare_modes(output_dir: Path) -> dict:
    """复用 ModeComparator 执行 A6 四模式对比, 返回 comparator.summary"""
    from evaluation.mode_comparator import ModeComparator

    comparator = ModeComparator(output_dir=str(output_dir))
    await comparator.initialize()
    await comparator.run()
    report_path = comparator.generate_report()
    logger.info(f"comparator 产物: {report_path.name} (naive/local/global/mix × 28)")
    return comparator.summary


def _now() -> str:
    """当前时间戳"""
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# 动作实现
# ============================================================

async def _action_compare_modes(output_dir: Path) -> dict:
    """--compare-modes: 只执行 A6, 永不调用 LLM Judge"""
    summary = await _run_compare_modes(output_dir)
    return {
        "_meta": {
            "phase": "5A",
            "step": "A7",
            "action": "compare_modes",
            "llm_judge": False,
            "generated_at": _now(),
        },
        "summary": summary,
    }


async def _action_full(args, output_dir: Path) -> dict:
    """--full: 单模式 baseline (A4/A5) + 四模式对比 (A6)"""
    mode = args.mode or "mix"
    enable_llm_judge = not args.no_llm_judge

    runner_summary = await _run_runner(mode, enable_llm_judge, output_dir)
    comparator_summary = await _run_compare_modes(output_dir)

    return {
        "_meta": {
            "phase": "5A",
            "step": "A7",
            "action": "full",
            "mode": mode,
            "llm_judge": enable_llm_judge,
            "generated_at": _now(),
        },
        "mode": mode,
        "llm_judge": enable_llm_judge,
        "runner_summary": runner_summary,
        "mode_comparison_summary": comparator_summary,
    }


async def _action_refusal_only(args, output_dir: Path) -> dict:
    """--refusal-only: 只输出拒答相关评估 (M7), 不调用 LLM Judge"""
    mode = args.mode or "mix"

    # 拒答检测 (runner.detect_refusal) 是规则化启发式, 不需要 LLM Judge
    runner_summary = await _run_runner(mode, False, output_dir)

    refusal_report = {
        "M7_refusal_accuracy": runner_summary.get("M7_refusal_accuracy"),
        "by_category": runner_summary.get("by_category"),
        "special_categories": runner_summary.get("special_categories"),
        "by_safety_level": runner_summary.get("by_safety_level"),
        "failed_questions": runner_summary.get("failed_questions"),
    }

    return {
        "_meta": {
            "phase": "5A",
            "step": "A7",
            "action": "refusal_only",
            "mode": mode,
            "llm_judge": False,
            "generated_at": _now(),
        },
        "mode": mode,
        "refusal_report": refusal_report,
    }


async def _action_single(args, output_dir: Path) -> dict:
    """--mode <m>: 单模式评估 (A4/A5)"""
    mode = args.mode or "mix"
    enable_llm_judge = not args.no_llm_judge

    summary = await _run_runner(mode, enable_llm_judge, output_dir)

    return {
        "_meta": {
            "phase": "5A",
            "step": "A7",
            "action": "single",
            "mode": mode,
            "llm_judge": enable_llm_judge,
            "generated_at": _now(),
        },
        "mode": mode,
        "llm_judge": enable_llm_judge,
        "summary": summary,
    }


# ============================================================
# CLI 入口
# ============================================================

async def main():
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        # 动作解析 (冲突时 parser.error → exit 2)
        action = _resolve_action(args, parser)

        # compare_modes 固定对比 4 种模式, --mode 无意义 → 明确报错
        if action == "compare_modes" and args.mode is not None:
            parser.error(
                "--compare-modes 固定对比 naive/local/global/mix 4 种模式, "
                "不能同时指定 --mode"
            )

        output_dir, main_output = _output_paths(args, action)
        logger.info(f"动作: {action} | 输出目录: {output_dir} | 主报告: {main_output}")

        if action == "full":
            result = await _action_full(args, output_dir)
        elif action == "compare_modes":
            result = await _action_compare_modes(output_dir)
        elif action == "refusal_only":
            result = await _action_refusal_only(args, output_dir)
        else:
            result = await _action_single(args, output_dir)

        with open(main_output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[OK] 主报告已写入: {main_output}")

    except Exception as exc:
        logger.error(f"CLI 执行失败: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
