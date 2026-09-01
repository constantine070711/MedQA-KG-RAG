"""
Phase 5A A6 — Mode Comparator (4 种检索模式对比)

目标:
    客观对比 LightRAG 四种检索模式 (naive / local / global / mix)
    在相同 28 道测试问题上的"检索效果"。

范围:
    - 只评估检索效果 (M1 Retrieval Entity Recall / M2 Chunk Relevance / M4 Context Precision)
    - 不调用 LLM，不生成回答，不计算 M5 (Faithfulness) / M6 (Citation)
    - 只调用 RAGEngine.retrieve()，零 API 费用
    - 不依赖 data.references (D2)

复用:
    - 初始化逻辑与 evaluation/runner.py 保持一致 (RAGEngine + entity_name_map)
    - M1/M2/M4 完全复用 evaluation/metrics.py 现有定义，不修改指标
    - 空 expected_entities / key_facts 的处理与 runner.py 一致:
        M1/M2 记为 None (不适用，不参与平均)；M4 无条件计算

运行方式:
    python -m evaluation.mode_comparator
    python -m evaluation.mode_comparator --output-dir evaluation/reports
    python -m evaluation.mode_comparator --top-k 60 --chunk-top-k 20

输出:
    evaluation/reports/phase5a_a6_mode_comparison.json
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.metrics import (
    load_entity_name_map,
    retrieval_entity_recall,
    retrieval_chunk_relevance,
    context_precision,
)

logger = logging.getLogger(__name__)

# 参与对比的 4 种检索模式 (与 ARCHITECTURE.md 一致)
MODES: list[str] = ["naive", "local", "global", "mix"]


# ============================================================
# 安全提取辅助函数 (与 runner.py 保持一致)
# ============================================================

def _safe_get_data(result: dict) -> dict:
    """安全提取 result['data']，处理各种异常格式"""
    if not isinstance(result, dict):
        return {}
    data = result.get("data", {})
    return data if isinstance(data, dict) else {}


def _safe_get_list(data: dict, key: str) -> list:
    """安全提取列表字段"""
    val = data.get(key, [])
    return val if isinstance(val, list) else []


def _safe_score(metrics: dict, key: str) -> Optional[float]:
    """安全提取指标分数"""
    m = metrics.get(key, {})
    if isinstance(m, dict):
        return m.get("score")
    return None


def _fmt(value) -> str:
    """格式化分数输出，None → 'N/A'"""
    return "N/A" if value is None else str(value)


# ============================================================
# Mode Comparator
# ============================================================

class ModeComparator:
    """
    Phase 5A A6 — 4 种检索模式对比器。

    流程:
        1. 初始化 RAGEngine + 实体名称映射表
        2. 加载 28 条测试问题
        3. 对每个 mode 分别执行全部 28 题 retrieve()
        4. 计算 M1 / M2 / M4
        5. 汇总 overall + by_category + per_question
        6. 生成 JSON 报告

    设计原则:
        - 只评估检索效果，不调用 LLM
        - 完全复用 metrics.py 中 M1/M2/M4 定义，不修改指标
        - 单题/单模式失败不中断整体评估
        - 4 个 mode 使用完全相同的测试集和指标
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        top_k: int = 60,
        chunk_top_k: int = 20,
    ):
        self.modes: list[str] = list(MODES)
        self.top_k = top_k
        self.chunk_top_k = chunk_top_k

        # 输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = _PROJECT_ROOT / "evaluation" / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 状态
        self.engine = None
        self.entity_name_map: dict[str, str] = {}
        self.questions: list[dict] = []
        self.results: list[dict] = []          # 每题每 mode 一条记录
        self.summary: dict = {}

        # 统计
        self.success_count = 0
        self.failure_count = 0
        self.per_mode_stats: dict[str, dict] = {
            m: {"success": 0, "failed": 0} for m in self.modes
        }

    # ----------------------------------------------------------
    # 初始化
    # ----------------------------------------------------------

    async def initialize(self) -> None:
        """初始化 RAGEngine 并加载实体名称映射表 (复用 runner.py 逻辑)"""
        logger.info("正在初始化 ModeComparator...")

        # 加载实体名称映射表 (D4: 中英文别名归一化)
        entities_path = _PROJECT_ROOT / "data" / "entities" / "entities.json"
        if entities_path.exists():
            self.entity_name_map = load_entity_name_map(str(entities_path))
            logger.info(
                f"实体名称映射表加载完成: {len(self.entity_name_map)} 个名称变体"
            )
        else:
            logger.warning(f"entities.json 不存在: {entities_path}，将使用空映射表")

        # 初始化 RAGEngine
        from src.rag_engine import RAGEngine

        self.engine = RAGEngine()
        await self.engine.initialize()
        logger.info("RAGEngine 初始化完成")

        # 加载测试问题
        questions_path = (
            _PROJECT_ROOT / "evaluation" / "datasets" / "test_questions.json"
        )
        with open(questions_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        self.questions = dataset.get("questions", [])
        logger.info(f"加载 {len(self.questions)} 条测试问题")

    # ----------------------------------------------------------
    # 单题单模式检索评估
    # ----------------------------------------------------------

    async def _evaluate_one_mode(self, q: dict, mode: str) -> dict:
        """
        对单条问题在指定 mode 下执行检索并计算 M1/M2/M4。

        Args:
            q: 测试问题 dict
            mode: 检索模式

        Returns:
            该题在该 mode 下的评估记录
        """
        question_id = q.get("id", "?")
        question_text = q.get("question", "")
        expected_entities = q.get("expected_entities", [])
        key_facts = q.get("key_facts", [])

        record = {
            "question_id": question_id,
            "question": question_text[:200],
            "category": q.get("category", ""),
            "mode": mode,
            "status": "pending",
            "error": None,
            "retrieved_entity_count": 0,
            "retrieved_chunk_count": 0,
            "retrieved_relation_count": 0,
            "metrics": {},
            "duration_ms": 0,
        }

        t_start = time.time()

        try:
            # ---- 纯检索 (不调用 LLM) ----
            retrieve_result = await self.engine.retrieve(
                question_text,
                mode=mode,
                top_k=self.top_k,
                chunk_top_k=self.chunk_top_k,
            )

            status = retrieve_result.get("status", "unknown")
            if status != "success":
                raise RuntimeError(
                    f"retrieve() 返回 status={status}, "
                    f"message={retrieve_result.get('message', '')}"
                )

            retrieve_data = _safe_get_data(retrieve_result)
            retrieved_entities = _safe_get_list(retrieve_data, "entities")
            retrieved_chunks = _safe_get_list(retrieve_data, "chunks")
            retrieved_relations = _safe_get_list(retrieve_data, "relationships")

            record["retrieved_entity_count"] = len(retrieved_entities)
            record["retrieved_chunk_count"] = len(retrieved_chunks)
            record["retrieved_relation_count"] = len(retrieved_relations)

            # ---- M1 / M2 / M4 ----
            metrics = {}

            # M1: 仅当有期望实体时计算 (空则不适用, 与 runner.py 一致)
            if expected_entities:
                metrics["M1"] = retrieval_entity_recall(
                    expected_entities, retrieved_entities, self.entity_name_map
                )
            else:
                metrics["M1"] = {
                    "score": None,
                    "note": "expected_entities 为空，不适用",
                }

            # M2: 仅当有关键事实时计算
            if key_facts:
                metrics["M2"] = retrieval_chunk_relevance(
                    key_facts, retrieved_chunks
                )
            else:
                metrics["M2"] = {
                    "score": None,
                    "note": "key_facts 为空，不适用",
                }

            # M4: 无条件计算 (与 runner.py 一致)
            #     注意: expected_entities 为空时相关实体数为 0，score 可能偏低
            metrics["M4"] = context_precision(
                retrieved_entities, expected_entities, self.entity_name_map
            )

            record["metrics"] = metrics
            record["status"] = "success"

        except Exception as exc:
            record["status"] = "error"
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
            logger.error(
                f"[{question_id}] mode={mode} 检索失败: "
                f"{type(exc).__name__}: {exc}"
            )

        record["duration_ms"] = round((time.time() - t_start) * 1000)
        return record

    # ----------------------------------------------------------
    # 全量执行
    # ----------------------------------------------------------

    async def run(self) -> None:
        """对 4 种模式 × 全部题目执行检索评估"""
        if self.engine is None:
            raise RuntimeError("请先调用 await comparator.initialize()")

        total_questions = len(self.questions)
        total_calls = total_questions * len(self.modes)
        logger.info(
            f"开始模式对比: {len(self.modes)} modes × {total_questions} 题 "
            f"= {total_calls} 次检索"
        )
        t_start = time.time()

        self.results = []
        self.success_count = 0
        self.failure_count = 0
        for m in self.per_mode_stats:
            self.per_mode_stats[m] = {"success": 0, "failed": 0}

        call_index = 0
        for i, q in enumerate(self.questions):
            qid = q.get("id", f"Q{id(q)}")
            cat = q.get("category", "?")
            for mode in self.modes:
                call_index += 1
                logger.info(
                    f"[{call_index}/{total_calls}] {qid} mode={mode} "
                    f"(cat={cat})"
                )

                record = await self._evaluate_one_mode(q, mode)
                self.results.append(record)

                if record["status"] == "success":
                    self.success_count += 1
                    self.per_mode_stats[mode]["success"] += 1
                else:
                    self.failure_count += 1
                    self.per_mode_stats[mode]["failed"] += 1

        total_duration = round((time.time() - t_start) * 1000)
        logger.info(
            f"模式对比完成: {self.success_count}/{total_calls} 成功, "
            f"{self.failure_count} 失败, 总耗时 {total_duration}ms"
        )

        self._build_summary(total_duration)

    # ----------------------------------------------------------
    # 汇总
    # ----------------------------------------------------------

    def _build_summary(self, total_duration_ms: int) -> None:
        """基于所有结果构建对比汇总"""

        def _avg(scores: list[float]) -> Optional[float]:
            if not scores:
                return None
            return round(sum(scores) / len(scores), 4)

        # ---- overall: 每个 mode 的 M1/M2/M4 平均 ----
        overall = {}
        for mode in self.modes:
            m1, m2, m4 = [], [], []
            for r in self.results:
                if r["mode"] != mode or r["status"] != "success":
                    continue
                s1 = _safe_score(r["metrics"], "M1")
                if s1 is not None:
                    m1.append(s1)
                s2 = _safe_score(r["metrics"], "M2")
                if s2 is not None:
                    m2.append(s2)
                s4 = _safe_score(r["metrics"], "M4")
                if s4 is not None:
                    m4.append(s4)
            overall[mode] = {
                "m1_avg": _avg(m1),
                "m2_avg": _avg(m2),
                "m4_avg": _avg(m4),
                "m1_count": len(m1),
                "m2_count": len(m2),
                "m4_count": len(m4),
                "success": self.per_mode_stats[mode]["success"],
                "failed": self.per_mode_stats[mode]["failed"],
            }

        # ---- by_category: category × mode ----
        categories = [
            "basic_fact",
            "cross_entity",
            "multi_hop",
            "out_of_kb",
            "hallucination_prone",
            "safety_sensitive",
        ]
        by_category = {}
        for cat in categories:
            by_category[cat] = {}
            for mode in self.modes:
                m1, m2, m4 = [], [], []
                success = 0
                failed = 0
                for r in self.results:
                    if r["mode"] != mode or r["category"] != cat:
                        continue
                    if r["status"] == "success":
                        success += 1
                        s1 = _safe_score(r["metrics"], "M1")
                        if s1 is not None:
                            m1.append(s1)
                        s2 = _safe_score(r["metrics"], "M2")
                        if s2 is not None:
                            m2.append(s2)
                        s4 = _safe_score(r["metrics"], "M4")
                        if s4 is not None:
                            m4.append(s4)
                    else:
                        failed += 1
                by_category[cat][mode] = {
                    "m1_avg": _avg(m1),
                    "m2_avg": _avg(m2),
                    "m4_avg": _avg(m4),
                    "success": success,
                    "failed": failed,
                }

        # ---- per_question: 嵌套 mode 结果 (保持题序) ----
        per_question = []
        for q in self.questions:
            qid = q.get("id", "?")
            modes_map = {}
            for mode in self.modes:
                rec = next(
                    (r for r in self.results
                     if r["question_id"] == qid and r["mode"] == mode),
                    None,
                )
                if rec is None:
                    continue
                metrics = rec.get("metrics", {})
                modes_map[mode] = {
                    "status": rec["status"],
                    "error": rec.get("error"),
                    "m1": _safe_score(metrics, "M1"),
                    "m2": _safe_score(metrics, "M2"),
                    "m4": _safe_score(metrics, "M4"),
                    "retrieved_entity_count": rec.get("retrieved_entity_count", 0),
                    "retrieved_chunk_count": rec.get("retrieved_chunk_count", 0),
                    "retrieved_relation_count": rec.get("retrieved_relation_count", 0),
                    "duration_ms": rec.get("duration_ms", 0),
                }
            per_question.append({
                "id": qid,
                "question": q.get("question", "")[:200],
                "category": q.get("category", ""),
                "expected_entities": q.get("expected_entities", []),
                "modes": modes_map,
            })

        # ---- 失败详情 ----
        failed_details = [
            {
                "question_id": r["question_id"],
                "mode": r["mode"],
                "category": r["category"],
                "error": r.get("error", {}),
            }
            for r in self.results
            if r["status"] == "error"
        ]

        # ---- 每题最佳模式分析 (M1→M2→M4 依次比较, None 视为 -1) ----
        best_mode_analysis = []
        for pq in per_question:
            candidates = []
            for mode, data in pq["modes"].items():
                if data["status"] != "success":
                    continue
                scores = (data.get("m1"), data.get("m2"), data.get("m4"))
                if all(s is None for s in scores):
                    continue
                candidates.append((mode, scores))
            if not candidates:
                best_mode_analysis.append({
                    "question_id": pq["id"],
                    "category": pq["category"],
                    "best_mode": None,
                    "scores": None,
                })
                continue

            best_mode, best_scores = candidates[0]
            for mode, scores in candidates[1:]:
                if self._scores_better(scores, best_scores):
                    best_mode, best_scores = mode, scores
            best_mode_analysis.append({
                "question_id": pq["id"],
                "category": pq["category"],
                "best_mode": best_mode,
                "scores": {
                    "m1": pq["modes"][best_mode].get("m1"),
                    "m2": pq["modes"][best_mode].get("m2"),
                    "m4": pq["modes"][best_mode].get("m4"),
                },
            })

        # ---- 最佳模式分布 ----
        best_mode_distribution: dict[str, int] = {}
        for bm in best_mode_analysis:
            m = bm["best_mode"]
            if m is not None:
                best_mode_distribution[m] = best_mode_distribution.get(m, 0) + 1

        # ---- 组装 summary ----
        self.summary = {
            "_meta": {
                "phase": "5A",
                "step": "A6",
                "modes": self.modes,
                "total_questions": len(self.questions),
                "total_calls": len(self.results),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "execution": {
                "modes": self.modes,
                "total_questions": len(self.questions),
                "expected_calls": len(self.questions) * len(self.modes),
                "successful": self.success_count,
                "failed": self.failure_count,
                "per_mode": self.per_mode_stats,
                "total_duration_ms": total_duration_ms,
            },
            "overall": overall,
            "by_category": by_category,
            "per_question": per_question,
            "best_mode_analysis": best_mode_analysis,
            "best_mode_distribution": best_mode_distribution,
            "failed_details": failed_details,
        }

    @staticmethod
    def _scores_better(a: tuple, b: tuple) -> bool:
        """
        比较两个 score 三元组 (m1, m2, m4) 的优劣。

        依次比较 m1 → m2 → m4，None 视为 -1.0。
        """
        for x, y in zip(a, b):
            xv = -1.0 if x is None else x
            yv = -1.0 if y is None else y
            if xv != yv:
                return xv > yv
        return False

    # ----------------------------------------------------------
    # 报告生成
    # ----------------------------------------------------------

    def generate_report(self) -> Path:
        """生成对比报告 JSON 文件"""
        path = self.output_dir / "phase5a_a6_mode_comparison.json"
        report = {
            "summary": self.summary,
            "results": self.results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"模式对比报告已写入: {path}")
        return path

    # ----------------------------------------------------------
    # 打印摘要
    # ----------------------------------------------------------

    def print_summary(self) -> None:
        """在控制台打印对比矩阵"""
        s = self.summary
        exec_info = s.get("execution", {})

        print("\n" + "=" * 78)
        print("  Phase 5A A6 — Mode Comparator 模式对比摘要")
        print("=" * 78)
        print(f"  模式: {', '.join(self.modes)}")
        print(f"  题目: {exec_info.get('total_questions', '?')} 题")
        print(
            f"  检索次数: {exec_info.get('successful', '?')}/"
            f"{exec_info.get('expected_calls', '?')} 成功"
        )
        per_mode = exec_info.get("per_mode", {})
        for m in self.modes:
            st = per_mode.get(m, {})
            print(f"    - {m:8s}: success={st.get('success', 0)}  "
                  f"failed={st.get('failed', 0)}")
        print(f"  总耗时: {exec_info.get('total_duration_ms', '?')}ms")
        print("-" * 78)

        # ---- Overall 对比矩阵 ----
        print("  Overall 对比矩阵 (平均分):")
        print(f"    {'mode':10s} {'M1':>8s} {'M2':>8s} {'M4':>8s}")
        print("    " + "-" * 40)
        overall = s.get("overall", {})
        for m in self.modes:
            o = overall.get(m, {})
            print(
                f"    {m:10s} "
                f"{_fmt(o.get('m1_avg')):>8} "
                f"{_fmt(o.get('m2_avg')):>8} "
                f"{_fmt(o.get('m4_avg')):>8}"
            )
        print("-" * 78)

        # ---- By Category ----
        print("  By Category (平均分):")
        print(f"    {'category':20s} {'mode':8s} {'M1':>8s} {'M2':>8s} {'M4':>8s}")
        print("    " + "-" * 50)
        by_cat = s.get("by_category", {})
        for cat, modes in by_cat.items():
            for m in self.modes:
                d = modes.get(m, {})
                print(
                    f"    {cat:20s} {m:8s} "
                    f"{_fmt(d.get('m1_avg')):>8} "
                    f"{_fmt(d.get('m2_avg')):>8} "
                    f"{_fmt(d.get('m4_avg')):>8}"
                )
        print("-" * 78)

        # ---- 最佳模式分布 ----
        print("  Best Mode 分布 (按 M1→M2→M4 依次比较):")
        dist = s.get("best_mode_distribution", {})
        for m in self.modes:
            print(f"    {m:10s}: {dist.get(m, 0):2d} 题")
        print("-" * 78)

        # ---- 失败详情 ----
        failed = s.get("failed_details", [])
        if failed:
            print(f"  [WARN] 失败 {len(failed)} 次:")
            for f in failed:
                err = f.get("error", {})
                print(
                    f"    {f['question_id']} [{f['mode']}]: "
                    f"{err.get('type', '?')}: {err.get('message', '')[:120]}"
                )
        else:
            print("  [OK] 无失败，全部检索成功")
        print("=" * 78)


# ============================================================
# CLI 入口
# ============================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Phase 5A A6 — Mode Comparator (4 种检索模式对比)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="报告输出目录 (默认: evaluation/reports/)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=60,
        help="实体/关系检索数量 (默认: 60)",
    )
    parser.add_argument(
        "--chunk-top-k",
        type=int,
        default=20,
        help="文本块检索数量 (默认: 20)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细日志输出",
    )
    args = parser.parse_args()

    # Windows 控制台编码处理 (GBK 下避免 UnicodeEncodeError)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    comparator = ModeComparator(
        output_dir=args.output_dir,
        top_k=args.top_k,
        chunk_top_k=args.chunk_top_k,
    )

    try:
        await comparator.initialize()
        await comparator.run()
        report_path = comparator.generate_report()
        comparator.print_summary()

        print(f"\n[OK] 模式对比报告已生成: {report_path}")

    except Exception as exc:
        logger.error(f"ModeComparator 执行失败: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
