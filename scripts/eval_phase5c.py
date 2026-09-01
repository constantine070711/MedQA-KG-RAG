#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 5C C4 — 完整评估（含 LLM Judge）+ Phase 5C 最终报告生成器

复用 EvaluationRunner（decision layer ON，Step E5）执行 mix 模式完整评估，
将 M1-M12 + citation_analysis + uncertainty_analysis + safety + reliability_decision
组装为 `data/eval/phase5c_report.json`。

本脚本只做「评估 + 固化结果」，不修改任何指标定义 / 不修复 Q016 / 不修改 R5 /
不修改 decision layer。

用法:
    python scripts/eval_phase5c.py                  # mix, 含 LLM Judge (~8-12min)
    python scripts/eval_phase5c.py --no-llm-judge   # 跳过 LLM Judge (M5 llm_judge=None)
    python scripts/eval_phase5c.py --output data/eval/phase5c_report.json
    python scripts/eval_phase5c.py --verbose

退出码:
    0 成功 / 1 运行时致命错误
"""

import argparse
import asyncio
import json
import logging
import platform
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

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "phase5c_report.json"
DEFAULT_RUNNER_DIR = PROJECT_ROOT / "data" / "eval"
A8_BASELINE_PATH = PROJECT_ROOT / "data" / "eval" / "full_report.json"
B6_BASELINE_PATH = PROJECT_ROOT / "data" / "eval" / "phase5b_report.json"

logger = logging.getLogger(__name__)


# ============================================================
# 参数
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 5C C4 — 完整评估 + Phase 5C 最终报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/eval_phase5c.py\n"
            "  python scripts/eval_phase5c.py --no-llm-judge\n"
            "  python scripts/eval_phase5c.py --output data/eval/phase5c_report.json"
        ),
    )
    p.add_argument("--mode", default="mix",
                   choices=["mix", "hybrid", "local", "global", "naive"],
                   help="检索模式 (默认 mix)")
    p.add_argument("--no-llm-judge", action="store_true",
                   help="禁用 LLM-as-judge (M5 llm_judge=None, 节省 API 费用)")
    p.add_argument("--no-decision-layer", action="store_true",
                   help="禁用决策层 (仅对比用，Phase 5C 完整评估应默认 ON)")
    p.add_argument("--output", default=None, help="报告输出路径 (默认 data/eval/phase5c_report.json)")
    p.add_argument("--verbose", action="store_true", help="详细日志")
    return p


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# 指标提取辅助
# ============================================================

def _metric_score(metrics: dict, key: str):
    m = metrics.get(key)
    if isinstance(m, dict):
        s = m.get("score")
        return s if isinstance(s, (int, float)) else None
    return None


def _m5_summary(m5: dict) -> dict:
    if not isinstance(m5, dict):
        return {"heuristic": None, "llm_judge": None}
    h = m5.get("heuristic", {})
    j = m5.get("llm_judge", {})
    hs = h.get("score") if isinstance(h, dict) else None
    js = j.get("score") if isinstance(j, dict) else None
    return {
        "heuristic": hs if isinstance(hs, (int, float)) else None,
        "llm_judge": js if isinstance(js, (int, float)) else None,
    }


def _m6_summary(m6: dict) -> dict:
    if not isinstance(m6, dict):
        return {"actual": None, "potential": None, "present": False, "grounded": False, "gap": []}
    pot = m6.get("potential", {})
    act = m6.get("actual", {})
    return {
        "potential": pot.get("score") if isinstance(pot, dict) else None,
        "actual": act.get("score") if isinstance(act, dict) else None,
        "present": bool(m6.get("citation_present", False)),
        "grounded": bool(m6.get("citation_grounded", False)),
        "gap": m6.get("gap", []),
    }


def _compact_question(r: dict) -> dict:
    """将单题 record 压缩为报告友好的结构（含 reliability_decision）。"""
    m = r.get("metrics", {})
    ca = r.get("citation_analysis", {})
    ua = r.get("uncertainty_analysis", {})
    refusal = r.get("refusal_detection", {})
    rd = r.get("reliability_decision", {}) or {}

    return {
        "question_id": r.get("question_id"),
        "question": r.get("question", "")[:200],
        "category": r.get("category"),
        "difficulty": r.get("difficulty"),
        "safety_level": r.get("safety_level"),
        "expected_refusal": bool(r.get("expected_refusal", False)),
        "actual_refusal": bool(r.get("actual_refusal", False)),
        "status": r.get("status"),
        "error": r.get("error"),
        "safety_check": r.get("safety_check"),
        "refusal_detection": {
            "is_refusal": refusal.get("is_refusal"),
            "confidence": refusal.get("confidence"),
            "method": refusal.get("method"),
        },
        "reliability_decision": rd if isinstance(rd, dict) else None,
        "metrics": {
            "M1": _metric_score(m, "M1"),
            "M2": _metric_score(m, "M2"),
            "M3": _metric_score(m, "M3"),
            "M4": _metric_score(m, "M4"),
            "M5": _m5_summary(m.get("M5")),
            "M6": _m6_summary(m.get("M6")),
            "M8": m.get("M8"),
            "M9": m.get("M9"),
            "M10": m.get("M10"),
            "M12": m.get("M12"),
        },
        "citation_analysis": {
            "present": bool(ca.get("citation_present", False)) if isinstance(ca, dict) else None,
            "grounded": bool(ca.get("citation_grounded", False)) if isinstance(ca, dict) else None,
            "grounded_ratio": ca.get("grounded_ratio", 0.0) if isinstance(ca, dict) else None,
            "total_references": ca.get("total_references", 0) if isinstance(ca, dict) else 0,
            "grounding_method": ca.get("grounding_method") if isinstance(ca, dict) else None,
        },
        "uncertainty_analysis": {
            "uncertainty_score": ua.get("uncertainty_score") if isinstance(ua, dict) else None,
            "recommended_action": ua.get("recommended_action") if isinstance(ua, dict) else None,
            "reasons": ua.get("reasons", []) if isinstance(ua, dict) else [],
        },
    }


# ============================================================
# 汇总构建
# ============================================================

def _avg_min_max(scores: list) -> dict:
    if not scores:
        return {"avg": None, "count": 0, "min": None, "max": None}
    return {
        "avg": round(sum(scores) / len(scores), 4),
        "count": len(scores),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
    }


def _level_distribution(results: list, metric_key: str) -> dict:
    dist: dict[str, int] = {}
    for r in results:
        m = r.get("metrics", {}).get(metric_key)
        if isinstance(m, dict) and m.get("level"):
            lv = str(m["level"])
            dist[lv] = dist.get(lv, 0) + 1
    return dist


def _category_distribution(results: list, metric_key: str) -> dict:
    cats: dict[str, list] = {}
    for r in results:
        if r.get("status") != "success":
            continue
        m = r.get("metrics", {}).get(metric_key)
        if isinstance(m, dict) and isinstance(m.get("score"), (int, float)):
            cats.setdefault(r.get("category", "unknown"), []).append(m["score"])
    return {
        cat: round(sum(v) / len(v), 4) for cat, v in sorted(cats.items())
    } if cats else {}


def _safety_level_distribution(results: list) -> dict:
    dist: dict[str, int] = {}
    for r in results:
        sc = r.get("safety_check") or {}
        lv = sc.get("level", "unknown") if isinstance(sc, dict) else "unknown"
        dist[lv] = dist.get(lv, 0) + 1
    return dist


def _fp_details(results: list) -> list[dict]:
    out = []
    for r in results:
        if not r.get("expected_refusal", False) and r.get("actual_refusal"):
            out.append({
                "question_id": r.get("question_id"),
                "question": r.get("question", "")[:100],
                "safety_level": r.get("safety_level"),
            })
    return out


def _decision_distribution(results: list) -> dict:
    """Phase 5C 新增：reliability_decision 分布统计。"""
    dist: dict[str, int] = {}
    reason_codes: dict[str, int] = {}
    refuse_triggered: dict[str, int] = {}
    for r in results:
        rd = r.get("reliability_decision") or {}
        if not isinstance(rd, dict) or rd.get("enabled") is False:
            continue
        dec = rd.get("decision")
        if dec:
            dist[dec] = dist.get(dec, 0) + 1
        rc = rd.get("reason_code")
        if rc:
            reason_codes[rc] = reason_codes.get(rc, 0) + 1
        if dec == "refuse":
            for rule in (rd.get("triggered_rules") or []):
                refuse_triggered[rule] = refuse_triggered.get(rule, 0) + 1
    return {
        "decision_distribution": dist,
        "reason_code_distribution": reason_codes,
        "refuse_triggered_rule_counts": refuse_triggered,
        "total_with_decision": sum(dist.values()),
    }


def _load_a8_baseline() -> dict:
    if not A8_BASELINE_PATH.exists():
        return {}
    try:
        with open(A8_BASELINE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        rs = data.get("runner_summary", {})
        mo = rs.get("metrics_overall", {})
        m7 = rs.get("M7_refusal_accuracy", {})
        return {
            "M1": mo.get("M1_retrieval_entity_recall", {}).get("avg"),
            "M2": mo.get("M2_retrieval_chunk_relevance", {}).get("avg"),
            "M3": mo.get("M3_answer_entity_recall", {}).get("avg"),
            "M4": mo.get("M4_context_precision", {}).get("avg"),
            "M5_llm_judge": mo.get("M5_faithfulness", {}).get("llm_judge", {}).get("avg"),
            "M6_actual": mo.get("M6_citation", {}).get("actual_avg"),
            "M7": m7.get("accuracy"),
            "M7_TP_TN_FP_FN": [m7.get("TP"), m7.get("TN"), m7.get("FP"), m7.get("FN")],
            "source": str(A8_BASELINE_PATH),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def _load_b6_baseline() -> dict:
    """读取 Phase 5B B6 基线 (data/eval/phase5b_report.json)。"""
    if not B6_BASELINE_PATH.exists():
        return {"error": f"{B6_BASELINE_PATH} 不存在"}
    try:
        with open(B6_BASELINE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        m1m7 = data.get("m1_m7_summary", {})
        m8m12 = data.get("m8_m12_summary", {})
        m7 = m1m7.get("M7_refusal_accuracy", {})
        cm = data.get("refusal_confusion_matrix", {})
        return {
            "M1": m1m7.get("M1", {}).get("avg"),
            "M2": m1m7.get("M2", {}).get("avg"),
            "M3": m1m7.get("M3", {}).get("avg"),
            "M4": m1m7.get("M4", {}).get("avg"),
            "M5_heuristic": m1m7.get("M5_faithfulness", {}).get("heuristic", {}).get("avg"),
            "M5_llm_judge": m1m7.get("M5_faithfulness", {}).get("llm_judge", {}).get("avg"),
            "M6_actual": m1m7.get("M6_citation", {}).get("actual_avg"),
            "M6_present_rate": m1m7.get("M6_citation", {}).get("citation_present_rate"),
            "M6_grounded_rate": m1m7.get("M6_citation", {}).get("citation_grounded_rate"),
            "M7": m7.get("accuracy"),
            "M7_TP_TN_FP_FN": [m7.get("TP"), m7.get("TN"), m7.get("FP"), m7.get("FN")],
            "M8": m8m12.get("M8_uncertainty", {}).get("avg"),
            "M9": m8m12.get("M9_citation_quality", {}).get("avg"),
            "M10": m8m12.get("M10_safety_signal_presence", {}).get("avg"),
            "M12": m8m12.get("M12_combined_reliability", {}).get("avg"),
            "confusion_matrix": cm,
            "source": str(B6_BASELINE_PATH),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}


KNOWN_LIMITATIONS = [
    "Q028 (个人诊断) 已由 Phase 5C R3 决策层修复为 REFUSE（SafetyChecker 盲区由 B4 diagnosis 信号兜底）。",
    "Q016 (COVID out-of-KB) 仍为已知 FN：out-of-KB measurement 局限（C2.5 审计结论：现有信号依赖 GT 缺失，无法在真实运行时安全区分）。未修改 R5，未加 qid 特判。未来用内容级 question↔chunk 语义支撑测量解决。",
    "Q021 (二甲双胍能否治愈 2 型糖尿病) 仍为已知 FP：由 LLM 非确定性产生 matched_patterns 拒答，narrow veto 不覆盖（非 emergency/short_referral 分支）。不特判。",
    "M7 语义变化：actual_refusal 由 decision layer 合成（决策REFUSE OR (detect_refusal AND NOT veto)），与 B6 纯 answer 检测不可直接对比；可用 --no-decision-layer 复现 B6。",
    "M4 Context Precision 全局偏低 (≤0.023)：LightRAG 广域检索固有现象，不修改指标。",
    "M5 Heuristic (bigram Jaccard) 对 LLM 改写文本效果有限 (≈0.02)，使用 LLM-as-judge 替代。",
    "M5 LLM Judge / M6 citation present 受 LLM 非确定性影响，运行间波动属预期。",
]


# ============================================================
# 报告组装
# ============================================================

def build_report(runner, results: list, *, mode: str, llm_judge: bool,
                 decision_layer: bool, command: str, elapsed_ms: int,
                 output_path: Path) -> dict:
    summary = runner.summary
    mo = summary.get("metrics_overall", {})
    m7 = summary.get("M7_refusal_accuracy", {})
    exec_info = summary.get("execution", {})

    dataset_by_category: dict[str, int] = {}
    dataset_by_safety: dict[str, int] = {}
    expected_refusal_total = 0
    for r in results:
        dataset_by_category[r.get("category", "unknown")] = \
            dataset_by_category.get(r.get("category", "unknown"), 0) + 1
        dataset_by_safety[r.get("safety_level", "unknown")] = \
            dataset_by_safety.get(r.get("safety_level", "unknown"), 0) + 1
        if r.get("expected_refusal"):
            expected_refusal_total += 1

    m8_scores = [r["metrics"]["M8"]["score"] for r in results if r.get("metrics", {}).get("M8", {}).get("score") is not None]
    m9_scores = [r["metrics"]["M9"]["score"] for r in results if r.get("metrics", {}).get("M9", {}).get("score") is not None]
    m10_scores = [r["metrics"]["M10"]["score"] for r in results if r.get("metrics", {}).get("M10", {}).get("score") is not None]
    m12_scores = [r["metrics"]["M12"]["score"] for r in results if r.get("metrics", {}).get("M12", {}).get("score") is not None]

    ca_present = sum(1 for r in results
                     if isinstance(r.get("citation_analysis"), dict)
                     and r["citation_analysis"].get("citation_present", False))
    ca_grounded = sum(1 for r in results
                      if isinstance(r.get("citation_analysis"), dict)
                      and r["citation_analysis"].get("citation_grounded", False))
    ca_ratios = [r["citation_analysis"].get("grounded_ratio", 0.0)
                 for r in results
                 if isinstance(r.get("citation_analysis"), dict)
                 and r["citation_analysis"].get("citation_present", False)]
    ca_methods: dict[str, int] = {}
    for r in results:
        gm = r["citation_analysis"].get("grounding_method")
        if gm:
            ca_methods[gm] = ca_methods.get(gm, 0) + 1

    action_dist: dict[str, int] = {}
    for r in results:
        a = r["uncertainty_analysis"].get("recommended_action")
        if a:
            action_dist[a] = action_dist.get(a, 0) + 1

    decision_stats = _decision_distribution(results)

    def _q_analysis(qid: str) -> dict:
        for r in results:
            if r["question_id"] == qid:
                return {
                    "question": r["question"],
                    "category": r["category"],
                    "safety_check": r["safety_check"],
                    "safety_sensitive_indicators": next(
                        (s.get("value") for s in r.get("uncertainty_analysis", {}).get("signals", [])
                         if s.get("name") == "safety_sensitive_indicators"), None),
                    "actual_refusal": r["actual_refusal"],
                    "expected_refusal": r["expected_refusal"],
                    "reliability_decision": r.get("reliability_decision"),
                    "metrics": r["metrics"],
                    "citation_analysis": r["citation_analysis"],
                    "uncertainty_analysis": {
                        "uncertainty_score": r["uncertainty_analysis"].get("uncertainty_score"),
                        "recommended_action": r["uncertainty_analysis"].get("recommended_action"),
                    },
                }
        return {"error": f"{qid} 未找到"}

    b6_base = _load_b6_baseline()
    current_metrics = {
        "M1": mo.get("M1_retrieval_entity_recall", {}).get("avg"),
        "M2": mo.get("M2_retrieval_chunk_relevance", {}).get("avg"),
        "M3": mo.get("M3_answer_entity_recall", {}).get("avg"),
        "M4": mo.get("M4_context_precision", {}).get("avg"),
        "M5_heuristic": mo.get("M5_faithfulness", {}).get("heuristic", {}).get("avg"),
        "M5_llm_judge": mo.get("M5_faithfulness", {}).get("llm_judge", {}).get("avg"),
        "M6_actual": mo.get("M6_citation", {}).get("actual_avg"),
        "M6_present_rate": mo.get("M6_citation", {}).get("citation_present_rate"),
        "M6_grounded_rate": mo.get("M6_citation", {}).get("citation_grounded_rate"),
        "M7": m7.get("accuracy"),
        "M7_TP_TN_FP_FN": [m7.get("TP"), m7.get("TN"), m7.get("FP"), m7.get("FN")],
        "M8": _avg_min_max(m8_scores)["avg"],
        "M9": _avg_min_max(m9_scores)["avg"],
        "M10": _avg_min_max(m10_scores)["avg"],
        "M12": _avg_min_max(m12_scores)["avg"],
    }

    # B6 → C4 差异 (仅当 B6 基线可读时)
    b6_diff = {}
    if isinstance(b6_base, dict) and "error" not in b6_base:
        for k in ("M1", "M2", "M3", "M4", "M5_llm_judge", "M6_actual", "M7",
                  "M8", "M9", "M10", "M12"):
            b6v = b6_base.get(k)
            cv = current_metrics.get(k)
            if isinstance(b6v, (int, float)) and isinstance(cv, (int, float)):
                b6_diff[k] = round(cv - b6v, 4)
        if isinstance(b6_base.get("M7_TP_TN_FP_FN"), list):
            b6_diff["M7_TP_TN_FP_FN_delta"] = [
                current_metrics["M7_TP_TN_FP_FN"][i] - b6_base["M7_TP_TN_FP_FN"][i]
                for i in range(4)
            ]

    report = {
        "_meta": {
            "phase": "5C",
            "step": "C4",
            "mode": mode,
            "llm_judge": llm_judge,
            "decision_layer": decision_layer,
            "generated_at": _now(),
            "command": command,
            "output": str(output_path),
        },
        "dataset_summary": {
            "total": len(results),
            "by_category": dataset_by_category,
            "by_gt_safety_level": dataset_by_safety,
            "expected_refusal": expected_refusal_total,
            "expected_answer": len(results) - expected_refusal_total,
        },
        "baseline_comparison": {
            "phase5a_a8": _load_a8_baseline(),
            "phase5b_b6": b6_base,
            "phase5c_c4_current": current_metrics,
            "b6_to_c4_delta": b6_diff,
        },
        "m1_m7_summary": {
            "M1": mo.get("M1_retrieval_entity_recall", {}),
            "M2": mo.get("M2_retrieval_chunk_relevance", {}),
            "M3": mo.get("M3_answer_entity_recall", {}),
            "M4": mo.get("M4_context_precision", {}),
            "M5_faithfulness": mo.get("M5_faithfulness", {}),
            "M6_citation": mo.get("M6_citation", {}),
            "M7_refusal_accuracy": m7,
            "by_category": summary.get("by_category", {}),
            "special_categories": summary.get("special_categories", {}),
        },
        "m8_m12_summary": {
            "M8_uncertainty": {
                **_avg_min_max(m8_scores),
                "level_distribution": _level_distribution(results, "M8"),
                "by_category_avg": _category_distribution(results, "M8"),
            },
            "M9_citation_quality": {
                **_avg_min_max(m9_scores),
                "by_category_avg": _category_distribution(results, "M9"),
            },
            "M10_safety_signal_presence": {
                **_avg_min_max(m10_scores),
                "safety_level_distribution": _safety_level_distribution(results),
                "by_category_avg": _category_distribution(results, "M10"),
            },
            "M12_combined_reliability": {
                **_avg_min_max(m12_scores),
                "level_distribution": _level_distribution(results, "M12"),
                "by_category_avg": _category_distribution(results, "M12"),
            },
        },
        "decision_layer_summary": {
            "enabled": decision_layer,
            "total_with_decision": decision_stats["total_with_decision"],
            "decision_distribution": decision_stats["decision_distribution"],
            "reason_code_distribution": decision_stats["reason_code_distribution"],
            "refuse_triggered_rule_counts": decision_stats["refuse_triggered_rule_counts"],
            "note": (
                "actual_refusal 由 decision layer 合成：decision==REFUSE 或 "
                "(detect_refusal AND NOT emergency/short_referral veto)"
            ),
        },
        "per_question": [_compact_question(r) for r in results],
        "q027_analysis": _q_analysis("Q027"),
        "q028_analysis": _q_analysis("Q028"),
        "q016_analysis": _q_analysis("Q016"),
        "refusal_confusion_matrix": {
            "TP": m7.get("TP"),
            "TN": m7.get("TN"),
            "FP": m7.get("FP"),
            "FN": m7.get("FN"),
            "accuracy": m7.get("accuracy"),
            "weighted_accuracy": m7.get("weighted_accuracy"),
            "fn_warning": m7.get("fn_warning"),
            "fn_details": m7.get("fn_details", []),
            "fp_details": _fp_details(results),
            "by_safety_level": m7.get("by_safety_level", {}),
        },
        "citation_summary": {
            "M6_present": mo.get("M6_citation", {}).get("citation_present_count"),
            "M6_grounded": mo.get("M6_citation", {}).get("citation_grounded_count"),
            "M6_present_rate": mo.get("M6_citation", {}).get("citation_present_rate"),
            "M6_grounded_rate": mo.get("M6_citation", {}).get("citation_grounded_rate"),
            "B3_present": ca_present,
            "B3_grounded": ca_grounded,
            "B3_grounded_of_present": (round(ca_grounded / ca_present, 4) if ca_present else None),
            "B3_avg_grounded_ratio_present": (
                round(sum(ca_ratios) / len(ca_ratios), 4) if ca_ratios else None),
            "B3_grounding_method_distribution": ca_methods,
        },
        "uncertainty_summary": {
            "M8_avg": _avg_min_max(m8_scores)["avg"],
            "M8_level_distribution": _level_distribution(results, "M8"),
            "recommended_action_distribution": action_dist,
        },
        "safety_summary": {
            "safety_check_level_distribution": _safety_level_distribution(results),
            "M10_avg": _avg_min_max(m10_scores)["avg"],
            "M10_gt_level_distribution": dataset_by_safety,
        },
        "failed_questions": summary.get("failed_questions", []),
        "execution": {
            "total": exec_info.get("total_questions"),
            "successful": exec_info.get("successful"),
            "failed": exec_info.get("failed"),
            "total_duration_ms": elapsed_ms,
        },
        "known_limitations": KNOWN_LIMITATIONS,
        "reproducibility": {
            "command": command,
            "mode": mode,
            "llm_judge": llm_judge,
            "decision_layer": decision_layer,
            "generated_at": _now(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "note": "确定性指标 (M1-M4/M6/M7/M8-M12) 可复现; M5 LLM Judge 与 M6 present 受 LLM 非确定性影响",
        },
    }
    return report


# ============================================================
# 主流程
# ============================================================

async def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = " ".join(sys.argv)

    try:
        from evaluation.runner import EvaluationRunner

        t0 = time.time()
        runner = EvaluationRunner(
            mode=args.mode,
            output_dir=str(DEFAULT_RUNNER_DIR),
            enable_llm_judge=not args.no_llm_judge,
            enable_decision_layer=not args.no_decision_layer,
        )
        await runner.initialize()
        await runner.run()
        elapsed_ms = round((time.time() - t0) * 1000)

        if runner.failure_count > 0:
            logger.warning(f"存在 {runner.failure_count} 条失败记录，见报告 failed_questions")

        report = build_report(
            runner, runner.results,
            mode=args.mode,
            llm_judge=not args.no_llm_judge,
            decision_layer=not args.no_decision_layer,
            command=command,
            elapsed_ms=elapsed_ms,
            output_path=output_path,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[OK] Phase 5C 最终报告已写入: {output_path}")

        mo = runner.summary.get("metrics_overall", {})
        m7 = runner.summary.get("M7_refusal_accuracy", {})
        print("\n" + "=" * 72)
        print("  Phase 5C C4 — 完整评估摘要")
        print("=" * 72)
        print(f"  Decision Layer: {'ON' if not args.no_decision_layer else 'OFF'}")
        print(f"  成功: {runner.success_count}/{len(runner.results)}  耗时: {elapsed_ms}ms")
        print(f"  M1={mo.get('M1_retrieval_entity_recall',{}).get('avg')}  "
              f"M2={mo.get('M2_retrieval_chunk_relevance',{}).get('avg')}  "
              f"M7={m7.get('accuracy')}  "
              f"TP/TN/FP/FN={m7.get('TP')}/{m7.get('TN')}/{m7.get('FP')}/{m7.get('FN')}")
        print(f"  M8={report['m8_m12_summary']['M8_uncertainty']['avg']}  "
              f"M9={report['m8_m12_summary']['M9_citation_quality']['avg']}  "
              f"M10={report['m8_m12_summary']['M10_safety_signal_presence']['avg']}  "
              f"M12={report['m8_m12_summary']['M12_combined_reliability']['avg']}")
        print(f"  decision 分布: {report['decision_layer_summary']['decision_distribution']}")

        for qid in ("Q027", "Q028", "Q016"):
            qa = report[f"{qid.lower()}_analysis"]
            if "error" in qa:
                print(f"  {qid}: 未找到")
                continue
            m = qa["metrics"]
            rd = qa["reliability_decision"] or {}
            print("-" * 72)
            print(f"  {qid}: {qa['question']}")
            print(f"    safety_check={qa['safety_check']}  actual_refusal={qa['actual_refusal']}")
            print(f"    M8={m['M8']['score']}  M9={m['M9']['score']}  "
                  f"M10={m['M10']['score']}  M12={m['M12']['score']}")
            print(f"    decision={rd.get('decision')}  reason_code={rd.get('reason_code')}  "
                  f"triggered={rd.get('triggered_rules')}")
            print(f"    s11={qa['safety_sensitive_indicators']}")

        fn = m7.get("fn_details", [])
        fp = report["refusal_confusion_matrix"]["fp_details"]
        if fn:
            print("-" * 72)
            print(f"  [M7 FN] {len(fn)} 条漏拒: " + ", ".join(f['question_id'] for f in fn))
        if fp:
            print(f"  [M7 FP] {len(fp)} 条误拒: " + ", ".join(f['question_id'] for f in fp))
        print("=" * 72)

    except Exception as exc:
        logger.error(f"评估失败: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
