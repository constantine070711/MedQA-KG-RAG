# -*- coding: utf-8 -*-
"""Phase 10-B B2 — 旁路检索级 Reranker（bge-reranker-base，CPU，可回退）。

作用：对 LightRAG 检索出的文本 chunks 做 cross-encoder 相关性重排，截取 top_m。
严格旁路，不进入生产 Agent / RAG / Fusion / Neo4j。

设计原则（docs/phase10_b_reranker_design_audit.md §13 Failure Fallback）：
  - 模型未加载 / 加载失败 → 返回原始候选顺序（截取 top_m）
  - 推理异常 / timeout → 返回原始候选顺序
  - score 含 NaN/Inf 等异常 → 返回原始候选顺序
  - 空候选 → 返回 []（空结果如实返回）
  - **不修改输入对象**：返回新列表（浅拷贝 + rerank_score），绝不 mutate 输入 chunk dict

用法（实验 / benchmark 用）：
    from src.retrieval.rerank import Reranker, rerank_chunks
    rk = Reranker()                      # 懒加载 models/bge-reranker-base 或 BAAI/bge-reranker-base
    top = rerank_chunks(question, chunks, top_m=14, reranker=rk)

依赖：transformers / torch（均已安装）；模型路径支持本地与 HF 名。
"""

import copy
import logging
import math
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

# 默认模型：优先本地 models/bge-reranker-base，回退 HF 名
DEFAULT_MODEL_PATH = "models/bge-reranker-base"
DEFAULT_MODEL_NAME = "BAAI/bge-reranker-base"


def _resolve_model_path(model_name_or_path: Optional[str]) -> str:
    """解析模型路径：显式传入优先；否则本地存在则本地，否则 HF 名。"""
    if model_name_or_path:
        return model_name_or_path
    local = Path(__file__).resolve().parent.parent.parent / DEFAULT_MODEL_PATH
    if local.exists():
        return str(local)
    return DEFAULT_MODEL_NAME


class Reranker:
    """bge-reranker-base cross-encoder 封装（CPU，懒加载，异常安全）。"""

    def __init__(self, model_name_or_path: Optional[str] = None, device: str = "cpu"):
        self.model_name_or_path = _resolve_model_path(model_name_or_path)
        self.device = device
        self._model = None
        self._tokenizer = None
        self._load_error: Optional[str] = None

    # ----------------------------------------------------------
    # 加载
    # ----------------------------------------------------------

    def _load(self) -> bool:
        """加载模型 + tokenizer；失败记录 _load_error 并返回 False（不抛给调用方）。

        失败结果缓存：同一实例重复调用 is_available 不再重试（避免反复网络超时）。
        """
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path
            )
            self._model.eval()
            self._model.to(self.device)
            return True
        except Exception as exc:  # noqa: BLE001 — 加载失败必须可回退
            self._load_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.warning("Reranker 加载失败，将回退原始顺序: %s", self._load_error)
            return False

    @property
    def is_available(self) -> bool:
        """模型是否可用（尝试加载）。"""
        return self._load()

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ----------------------------------------------------------
    # 打分
    # ----------------------------------------------------------

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """对 (query, passage) 对打分，返回相关性分数列表（升序对应不相关→相关）。

        异常（模型不可用 / 推理失败 / score 异常）一律 raise，由调用方回退。
        """
        if not pairs:
            return []
        if not self._load():
            raise RuntimeError(f"reranker model unavailable: {self._load_error}")

        import torch
        texts_a = [p[0] for p in pairs]
        texts_b = [p[1] for p in pairs]
        with torch.no_grad():
            inputs = self._tokenizer(
                texts_a, texts_b, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(self.device)
            logits = self._model(**inputs).logits

        scores: list[float] = []
        for row in logits:
            row = row.detach().cpu()
            if row.ndim == 0:
                s = float(row)
            else:
                if row.shape[0] >= 2:
                    import torch.nn.functional as F
                    s = float(F.softmax(row, dim=-1)[0])  # label 0 = relevant
                else:
                    s = float(row[0])
            if not math.isfinite(s):
                raise ValueError("rerank score 非有限值")
            scores.append(s)
        return scores


def _fallback(chunks: Sequence[dict], top_m: int) -> list[dict]:
    """回退：返回原始顺序的前 top_m（浅拷贝，不 mutate 输入）。"""
    out: list[dict] = []
    for c in list(chunks)[: max(top_m, 0)]:
        if isinstance(c, dict):
            out.append(copy.copy(c))
    return out


def rerank_chunks(
    question: str,
    chunks: Sequence[dict],
    top_m: int,
    reranker: Optional[Any] = None,
) -> list[dict]:
    """对 chunks 做 cross-encoder 相关性重排，截取 top_m。

    Args:
        question: 当前轮查询问题。
        chunks: LightRAG retrieve() 返回的 chunks 列表（含 "content" 字段）。
        top_m: 截取数量（进入最终 context 的 chunk 数）。
        reranker: 具备 `score_pairs(pairs) -> list[float]` 与 `is_available` 的对象；
            None 时尝试默认 Reranker（模型不可用则回退）。

    Returns:
        新列表（每项 = 输入 chunk 的浅拷贝 + `rerank_score`）；输入 chunks 不被修改。
        空候选 / 不可用 / 异常 → 原始顺序前 top_m（fallback）。
    """
    if not chunks or top_m is None or top_m <= 0:
        return []

    # 候选内容提取（不 mutate）
    candidates: list[dict] = [c for c in chunks if isinstance(c, dict) and c.get("content")]

    # 尝试打分
    scorer = reranker
    try:
        if scorer is None:
            scorer = Reranker()
        available = bool(scorer.is_available) if hasattr(scorer, "is_available") else True
        if not available:
            return _fallback(chunks, top_m)
        pairs = [(str(question or ""), str(c.get("content", "") or "")) for c in candidates]
        scores = scorer.score_pairs(pairs)
        if len(scores) != len(candidates):
            raise ValueError("rerank 分数数量与候选数量不一致")
    except Exception as exc:  # noqa: BLE001 — 任何异常都可回退
        logger.warning("rerank 失败，回退原始顺序: %s", f"{type(exc).__name__}: {str(exc)[:120]}")
        return _fallback(chunks, top_m)

    # 排序 + 截取（新列表，不 mutate 输入）
    ranked = sorted(
        zip(candidates, scores),
        key=lambda t: t[1],
        reverse=True,
    )[: max(top_m, 0)]
    out: list[dict] = []
    for c, s in ranked:
        new_c = copy.copy(c)
        new_c["rerank_score"] = round(float(s), 6)
        out.append(new_c)
    return out
