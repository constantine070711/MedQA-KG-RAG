# -*- coding: utf-8 -*-
"""Phase 10-C C2 — 旁路只读可观测性层（request_id + JSONL structured logging + metrics）。

设计（docs/phase10_c_logging_design_audit.md）：
  - **旁路**：只读 final_response + 调用计时；绝不修改 final_response / AgentState /
    SafetyChecker 输入 / 检索输入 / fused_context / ReliabilityDecision / Memory / UI 展示。
  - **结构化**：JSON Lines → data/logs/medqa.jsonl（gitignored），UTF-8，单行可独立解析。
  - **安全**：question/answer/prompt/API key/token/password/traceback 一律不入日志；
    question 仅记 sha256[:16] hash；字段白名单 + 防御性 sanitize（可处理嵌套 dict/list）。
  - **异常**：日志任何失败（目录不可写 / 序列化失败 / 字段缺失 / 结构异常）静默降级，
    仅 stdlib logger warning，绝不向用户抛出新异常、绝不影响问答。
  - **零第三方依赖**（stdlib only）。

字段（白名单，从 final_response 旁路提取；缺失记 None 不伪造）：
    question_hash / total_ms / safety_level / safety_reason / neo4j_status /
    neo4j_evidence_count / generation_count / decision / reason_code /
    is_final_refusal / confidence / m12 / citation_present / grounded_ratio /
    memory_used / error_category
"""

import hashlib
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# 默认日志路径（项目根/data/logs/medqa.jsonl）
_DEFAULT_LOG_NAME = Path("data") / "logs" / "medqa.jsonl"

# 日志轮转默认（Phase 10-C C3）：单文件超 5MB 滚动，保留最近 3 份备份（.1 最新）
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 3

# ---- 防御性脱敏模式（字段本身已白名单化；此为纵深防御，处理任何字符串值）----
_SENSITIVE_KEY_RE = re.compile(
    r"((?:api[_-]?key|password|passwd|secret|token|authorization|bearer)"
    r"\s*[=:]\s*\S+)",
    re.IGNORECASE,
)
_SK_RE = re.compile(r"\b(sk-[A-Za-z0-9_\-]{6,})\b")
_URI_CRED_RE = re.compile(r"(://)[^:@/\s]+:([^@/\s]+)@")


# ============================================================
# request_id / question_hash
# ============================================================

def new_request_id() -> str:
    """生成唯一请求 ID（uuid4 hex 前 12 位）。"""
    return uuid.uuid4().hex[:12]


def question_hash(question: Any) -> str:
    """question 的不可逆摘要（sha256 前 16 hex）；不保留原文。"""
    return hashlib.sha256(str(question or "").encode("utf-8")).hexdigest()[:16]


# ============================================================
# sanitize（嵌套安全）
# ============================================================

def _mask_str(s: str) -> str:
    out = _URI_CRED_RE.sub(r"\1***:***@", s)
    out = _SK_RE.sub("[隐藏:sk-key]", out)
    out = _SENSITIVE_KEY_RE.sub("[隐藏:凭据]", out)
    return out


def sanitize(value: Any) -> Any:
    """递归脱敏嵌套 dict/list/str；其它类型原样返回。返回新结构，不修改输入。"""
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        return _mask_str(value)
    return value


# ============================================================
# 内部工具
# ============================================================

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve(out: Optional[Path]) -> Path:
    if out is not None:
        return Path(out)
    return _project_root() / _DEFAULT_LOG_NAME


def _rotate_if_needed(path: Path, max_bytes: Optional[int], backup_count: int) -> None:
    """日志大小轮转（stdlib-only，失败静默）。

    当 `path` 当前大小超过 `max_bytes` 时滚动：`medqa.jsonl → .1 → .2 → …`，
    保留最近 `backup_count` 份备份（`.1` 最新）；`max_bytes` 为 None/<=0 时禁用。
    任何异常仅 stdlib warning，绝不抛给调用方（与 log_event 的静默降级一致）。
    """
    if max_bytes is None or max_bytes <= 0 or backup_count < 1:
        return
    try:
        if not path.exists() or path.stat().st_size <= max_bytes:
            return
        # 从旧到新平移备份：.backup_count-1 → .backup_count … .1 → .2
        for i in range(backup_count - 1, 0, -1):
            src = Path(f"{path}.{i}")
            dst = Path(f"{path}.{i + 1}")
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
        newest = Path(f"{path}.1")
        if newest.exists():
            newest.unlink()
        path.rename(newest)
    except Exception as exc:  # noqa: BLE001 — 轮转失败静默，仍尝试追加当前文件
        logger.warning(
            "monitoring 日志轮转失败，静默降级: %s",
            f"{type(exc).__name__}: {str(exc)[:120]}",
        )


def _num(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _count(v: Any) -> int:
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def _neo4j_status(fr: dict) -> str:
    """Neo4j 四态（与 UI format_neo4j_evidence 同语义）：
    not_checked > unavailable > hit > no_hit。"""
    level = str(fr.get("safety_level") or "").lower()
    cov = fr.get("coverage") or {}
    cov = cov if isinstance(cov, dict) else {}
    if level == "blocked":
        return "not_checked"
    if bool(fr.get("neo4j_unavailable")) or bool(cov.get("neo4j_unavailable")):
        return "unavailable"
    if bool(cov.get("neo4j_has_evidence")):
        return "hit"
    return "no_hit"


# ============================================================
# metrics 提取（白名单，只读 final_response）
# ============================================================

def _extract_observation(
    final_response: Any,
    conversation_context: Optional[str] = None,
) -> dict:
    """从 final_response 旁路提取白名单 metrics；缺失字段记 None，不伪造。

    说明：
      - m12 = reliability_decision.reliability_score（reliability_decision.py 注释
        「透传 M12.score」）；final_response 无独立 m12 字段。
      - safety_reason 不在 final_response 信封 → None（不伪造）。
    """
    fr = final_response if isinstance(final_response, dict) else {}
    rd = fr.get("reliability_decision") or {}
    rd = rd if isinstance(rd, dict) else {}
    cit = fr.get("citations") or {}
    cit = cit if isinstance(cit, dict) else {}
    fused = fr.get("fused_context") or {}
    fused = fused if isinstance(fused, dict) else {}
    ev = (fused.get("evidence_sources") or {}).get("neo4j_graph") or {}
    ev = ev if isinstance(ev, dict) else {}

    error_type = None
    errors = fr.get("errors")
    if isinstance(errors, list) and errors and isinstance(errors[0], dict):
        error_type = errors[0].get("type")

    return {
        "safety_level": fr.get("safety_level"),
        "safety_reason": None,  # 不在 final_response 信封（不伪造）
        "neo4j_status": _neo4j_status(fr),
        "neo4j_evidence_count": _count(ev.get("triples")) + _count(ev.get("nodes")),
        "generation_count": fr.get("llm_generation_count"),
        "decision": rd.get("decision"),
        "reason_code": rd.get("reason_code"),
        "is_final_refusal": bool(rd.get("is_final_refusal")),
        "confidence": rd.get("confidence"),
        "m12": _num(rd.get("reliability_score")),
        "citation_present": bool(cit.get("citation_present")),
        "grounded_ratio": _num(cit.get("grounded_ratio")),
        "memory_used": bool(conversation_context),
        "error_category": error_type,
    }


# ============================================================
# log_event / log_run（异常静默降级）
# ============================================================

def log_event(
    request_id: str,
    level: str = "info",
    event: str = "medqa",
    *,
    out: Optional[Path] = None,
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    **fields: Any,
) -> bool:
    """写一条 JSONL 日志。任何失败返回 False 并仅 stdlib warning，不抛异常。

    max_bytes / backup_count（Phase 10-C C3）：写前按文件大小轮转
    （medqa.jsonl → .1 → .2 …，保留 backup_count 份备份）；None/<=0 禁用。
    轮转失败静默，仍尝试追加当前文件。向后兼容：缺省与旧行为一致。
    """
    try:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "request_id": request_id,
            "level": level,
            "event": event,
        }
        record.update(sanitize(fields))
        line = json.dumps(record, ensure_ascii=False)
        path = _resolve(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        _rotate_if_needed(path, max_bytes, backup_count)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True
    except Exception as exc:  # noqa: BLE001 — 日志失败绝不影响调用方
        logger.warning(
            "monitoring log_event 失败，静默降级: %s",
            f"{type(exc).__name__}: {str(exc)[:120]}",
        )
        return False


def log_run(
    request_id: str,
    question: Any,
    final_response: Any,
    total_ms: Optional[float],
    conversation_context: Optional[str] = None,
    *,
    out: Optional[Path] = None,
    event: str = "agent_run",
    max_bytes: Optional[int] = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> bool:
    """旁路记录一次 Agent 调用观测。不修改 final_response；失败静默降级。

    max_bytes / backup_count 透传给 log_event（C3 轮转）。
    """
    try:
        fields = {
            "question_hash": question_hash(question),
            "total_ms": _num(total_ms),
        }
        fields.update(_extract_observation(final_response, conversation_context))
        return log_event(
            request_id, "info", event,
            out=out, max_bytes=max_bytes, backup_count=backup_count, **fields,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "monitoring log_run 失败，静默降级: %s",
            f"{type(exc).__name__}: {str(exc)[:120]}",
        )
        return False


# ============================================================
# ObservedAgent（可选包装：独立脚本/测试用；run_agent_sync 直接用 log_run）
# ============================================================

class ObservedAgent:
    """对 async agent 的旁路观测包装：run 后写日志，返回 final_response 不变。

    用法（独立脚本/测试）：
        agent = ObservedAgent(MedQAAgent())
        fr = await agent.run(question)
    """

    def __init__(self, agent: Any, out: Optional[Path] = None):
        self._agent = agent
        self._out = out

    async def run(self, question: str, conversation_context: Optional[str] = None) -> Any:
        rid = new_request_id()
        t0 = time.perf_counter()
        fr = await self._agent.run(question, conversation_context=conversation_context)
        total_ms = (time.perf_counter() - t0) * 1000.0
        log_run(rid, question, fr, total_ms,
                conversation_context=conversation_context, out=self._out)
        return fr
