# -*- coding: utf-8 -*-
"""Phase 8 P1 — UI 展示辅助层（纯函数，零 streamlit import，可独立单测）。

职责：
    - 输入 MedQAAgent 的 final_response → 输出 UI 可消费的结构化数据
    - 不重新执行 Agent / LLM / DB / 安全判断 / 拒答判断 / 引用分析 / 证据融合
    - 对 None / 未知字段 / 空列表 安全处理，缺失字段不抛异常

关键实现说明（run_agent_sync 持久事件循环）：
    Streamlit 每次 rerun 是同步脚本；若每次用 `asyncio.run()` 调用 MedQAAgent.run()，
    会为每次交互创建新的事件循环。LightRAG 的异步资源（PriorityQueue / asyncio.Lock /
    LLM·Embedding workers）绑定在首次查询的事件循环上 —— 第二次查询在新循环中会报
    "<obj> is bound to a different event loop"（P1 已实测），导致静默错误回答。
    因此本模块维护一个模块级持久事件循环（daemon 线程 + run_coroutine_threadsafe），
    让所有查询共享同一事件循环，保证跨多次 Streamlit 交互的正确性。
    这不是新增异步框架，只是 asyncio 循环生命周期管理。
"""

import asyncio
import re
import threading
import time
from typing import Any, Optional


# ============================================================
# 持久事件循环（run_agent_sync 用）
# ============================================================

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """获取进程级持久事件循环（daemon 线程运行，首次调用创建）。"""
    global _loop
    if _loop is None:
        with _loop_lock:
            if _loop is None:
                _loop = asyncio.new_event_loop()
                t = threading.Thread(
                    target=_loop.run_forever,
                    daemon=True,
                    name="medqa-async-loop",
                )
                t.start()
    return _loop


def _error_envelope(etype: str, message: str) -> dict:
    """构造 UI 安全错误信封（不含 traceback / 敏感信息）。"""
    return {
        "question": "",
        "answer": "",
        "status": "error",
        "errors": [{"type": etype, "message": str(message)[:200]}],
    }


def run_agent_sync(agent: Any, question: str, conversation_context: Optional[str] = None) -> dict:
    """同步调用 MedQAAgent.run(question)（内部为异步，用持久事件循环驱动）。

    Args:
        agent: 具有 `async run(question) -> dict` 的对象（MedQAAgent 或 mock）。
        question: 用户问题。
        conversation_context: 多轮对话背景（Memory，Phase 9 P2）。仅在非空时透传，
            空/None 时不传（与既有单参 mock agent 兼容）。

    Returns:
        final_response dict；异常/空输入/agent 不可用时返回错误信封（status="error"）。

    Phase 10-C C2（旁路观测）：
        生成 request_id + 计时，调用结束后经 src.monitoring.log_run 写一条 JSONL 观测日志。
        观测层只读 final_response / 输入 question 的 hash / 调用耗时；日志失败静默降级，
        绝不影响返回结果（final_response 原样返回）。
    """
    from src.monitoring import log_run, new_request_id

    request_id = new_request_id()
    _t0 = time.perf_counter()

    def _finish(fr: dict) -> dict:
        # 旁路观测：只读 final_response + 计时；任何日志失败均静默，不影响返回
        try:
            log_run(request_id, question, fr,
                    (time.perf_counter() - _t0) * 1000.0,
                    conversation_context=conversation_context)
        except Exception:
            pass
        return fr

    if agent is None:
        return _finish(_error_envelope("AgentNotReady", "Agent 尚未初始化"))
    if not question or not str(question).strip():
        return _finish(_error_envelope("EmptyQuestion", "问题为空"))

    loop = _get_event_loop()
    try:
        if conversation_context:
            coro = agent.run(question, conversation_context=conversation_context)
        else:
            coro = agent.run(question)
        if coro is None or not hasattr(coro, "__await__"):
            return _finish(_error_envelope("InvalidAgentRun", "Agent.run() 未返回可等待对象"))
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        result = future.result()  # 阻塞等待；Agent 内部已捕获异常返回错误信封
        if isinstance(result, dict):
            return _finish(result)
        return _finish(_error_envelope("InvalidAgentResult", "Agent 返回了非 dict 结果"))
    except Exception as exc:
        return _finish(_error_envelope(type(exc).__name__, str(exc)[:200]))


# ============================================================
# 会话历史（纯函数，供 Session State 逻辑可测试）
# ============================================================

def add_message(messages: Any, role: str, content: str, response: Any = None) -> list[dict]:
    """追加一条对话消息，返回新列表（不修改入参）。role ∈ {user, assistant}。

    可选 `response`：assistant 消息保存对应的 final_response，用于历史结构化回显
    （仅为 UI 层展示，不拼接入 Agent.question，Agent 保持单轮）。
    """
    out = list(messages) if isinstance(messages, list) else []
    msg: dict = {"role": role, "content": content}
    if response is not None:
        msg["response"] = response
    out.append(msg)
    return out


def reset_chat_state() -> dict:
    """清空对话状态（UI 层），返回新状态字典。注意：不清除 Agent 实例。"""
    return {"messages": [], "history": [], "last_response": None}


def lazy_init(state: dict, key: str, factory: Any) -> Any:
    """懒初始化：state 中无 key 时才调用 factory() 创建并缓存，之后复用同一实例。

    用于 session_state 懒创建 Agent / event-loop 依赖，保证 rerun 不重建。
    """
    if key not in state:
        state[key] = factory()
    return state[key]


def resolve_question(prompt: Any, example: Any = None) -> Optional[str]:
    """解析本轮问题：优先非空 prompt，其次非空示例问题；均空返回 None。

    用于区分 chat_input 提交与示例按钮触发，保证一次交互只产生一个 question。
    """
    p = str(prompt or "").strip()
    if p:
        return p
    e = str(example or "").strip()
    return e or None


# ============================================================
# 多轮对话 Memory（Phase 9 P2）— 构建对话背景（纯函数，零 LLM）
# ============================================================

def _memory_turn_is_eligible(entry: Any) -> Optional[tuple]:
    """判断一条历史轮是否可进入 Memory；合格则返回 (question, answer)，否则 None。

    规则：status!=error、safety_level!=blocked、is_final_refusal!=True，
    且 question/answer 均非空。refusal/blocked/error 一律不进入。
    """
    if not isinstance(entry, dict):
        return None
    fr = entry.get("final_response")
    if not isinstance(fr, dict):
        return None
    if fr.get("status") == "error":
        return None
    if str(fr.get("safety_level") or "").lower() == "blocked":
        return None
    rd = fr.get("reliability_decision") or {}
    if isinstance(rd, dict) and rd.get("is_final_refusal"):
        return None
    q = str(entry.get("question") or "").strip()
    ans = str(fr.get("answer") or "").strip()
    if not q or not ans:
        return None
    return (q, ans)


def build_conversation_context(
    history: Any,
    max_turns: int = 3,
    max_chars: int = 600,
) -> str:
    """从 UI 历史构建紧凑对话背景（Memory）。

    规则（Phase 9 P2，锁定参数 max_turns=3 / max_chars=600）：
    - 只纳入成功且非拒答轮；error / blocked / refusal 一律跳过；
    - 最多保留最近 max_turns 个合格轮；最终 ≤ max_chars 字符（超出时保留最近轮/截断）；
    - 只取 question + answer 文本，不含 fused_context / citation / coverage /
      Neo4j 结构 / debug 信息；经 mask_sensitive 脱敏；
    - 当前尚未完成的轮次（尚未写入 history）天然不会进入。
    无合格历史 → 返回 ""（与现有代码最兼容：Agent 侧 None/空 均视为无记忆）。

    Memory = 对话背景，不是 evidence / citation / retrieval / state 事实。
    """
    if not history:
        return ""
    pairs: list[tuple] = []
    for entry in history:
        hit = _memory_turn_is_eligible(entry)
        if hit:
            pairs.append(hit)
    if not pairs:
        return ""
    recent = pairs[-max_turns:] if max_turns > 0 else []
    if not recent:
        return ""

    def _block(pair) -> str:
        q, ans = pair
        return f"用户问题：{q}\n助手回答：{ans}\n"

    selected: list[str] = []
    used = 0
    for pair in reversed(recent):            # 从最近轮往前贪心
        block = _block(pair)
        if used + len(block) > max_chars:
            break
        selected.append(block)
        used += len(block)
    selected.reverse()                        # 恢复时间正序
    if selected:
        text = "".join(selected).strip()
    else:
        # 最近一轮本身超过 max_chars → 截断该轮回答
        text = _block(recent[-1])[:max_chars].rstrip()
    return mask_sensitive(text)


_MEMORY_USED_HINT = "本次回答已结合前文对话上下文（最近 ≤3 轮）。"


def memory_used_hint(memory: Optional[str], response: Any) -> Optional[str]:
    """判断本轮回答是否可提示「参考了前文对话上下文」（纯展示辅助，Phase 9 P4）。

    条件：memory 非空（本轮确实向 Agent 传入了前文背景）且当前为**实质回答**
    （非 error / 非 blocked / 非 final_refusal）→ 返回提示文案；否则 None。

    说明：
    - blocked / refusal 轮即使传了 memory 也不提示（回答是拦截/拒答，未提供实质内容）；
    - 本函数只读 final_response，不改变其语义。
    """
    if not memory:
        return None
    if not isinstance(response, dict) or not response:
        return None
    r = _as_dict(response)
    if r.get("status") == "error":
        return None
    if str(r.get("safety_level") or "").lower() == "blocked":
        return None
    rd = _as_dict(r.get("reliability_decision"))
    if rd.get("is_final_refusal"):
        return None
    return _MEMORY_USED_HINT


# ============================================================
# 敏感信息脱敏
# ============================================================

_SENSITIVE_KEY_RE = re.compile(
    r"((?:api[_-]?key|password|passwd|token|secret|authorization|bearer)"
    r"\s*[=:]\s*\S+)",
    re.IGNORECASE,
)
_SK_RE = re.compile(r"\b(sk-[A-Za-z0-9_\-]{6,})\b")
_URI_CRED_RE = re.compile(r"(://)[^:@/\s]+:([^@/\s]+)@")
_SENSITIVE_WORDS = ("apikey", "api-key", "password", "passwd", "secret", "bearer")


def mask_sensitive(text: str) -> str:
    """掩盖文本中的敏感信息（API key / password / token / 带凭据的 URI）。"""
    if not text:
        return ""
    out = str(text)
    out = _URI_CRED_RE.sub(r"\1***:***@", out)
    out = _SK_RE.sub("[隐藏:sk-key]", out)
    out = _SENSITIVE_KEY_RE.sub("[隐藏:凭据]", out)
    return out


def sanitize_errors(errors: Any) -> list[dict]:
    """清洗错误列表：仅保留 type + message，剥离 traceback/敏感字段，掩盖敏感内容。

    返回空列表表示无错误或输入非法。
    """
    if not isinstance(errors, list):
        return []
    out: list[dict] = []
    for e in errors:
        if not isinstance(e, dict):
            continue
        msg = mask_sensitive(str(e.get("message", "")))[:300]
        out.append({"type": str(e.get("type", "Unknown"))[:80], "message": msg})
    return out


# ============================================================
# 展示格式化
# ============================================================

_SAFETY_LABELS = {
    "safe": "正常知识回答",
    "caution": "谨慎回答，建议结合专业医生意见",
    "restricted": "受限回答（仅提供通用医学知识）",
    "blocked": "安全拦截（已拒绝实质性医学内容）",
}
_SAFETY_TONES = {
    "safe": "ok",
    "caution": "warn",
    "restricted": "warn",
    "blocked": "danger",
}
_DECISION_LABELS = {
    "allow": "允许回答",
    "revise": "回答并附免责/就医建议",
    "refuse": "拒绝提供实质性医学内容",
}
_NEO4J_STATUS_LABELS = {
    "not_checked": "安全拦截，未执行图谱检索",
    "unavailable": "不可用（未使用结构化图谱证据）",
    "hit": "已提供结构化图谱证据",
    "no_hit": "未命中 / 未产生图谱证据",
}

# ---- P3: 证据展示常量 ----
_EVIDENCE_KIND_LABELS = {
    "neo4j_triple": "图谱三元组",
    "neo4j_node": "图谱节点",
    "lightrag_entity": "实体",
    "lightrag_relation": "关系",
    "lightrag_chunk": "文本块",
}
_SOURCE_LABELS = {
    "lightrag_vector": "LightRAG 向量",
    "lightrag_graph": "LightRAG 图",
    "neo4j_graph": "Neo4j 图",
}
_LIGHTRAG_KINDS = {"lightrag_entity", "lightrag_relation", "lightrag_chunk"}
_DEFAULT_EVIDENCE_MAX = 5


def _as_dict(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def format_safety_level(safety_level: Any) -> dict:
    """安全等级 → {level, label, tone}。tone ∈ {ok, warn, danger, muted}。"""
    level = str(safety_level or "unknown").lower()
    return {
        "level": level,
        "label": _SAFETY_LABELS.get(level, "未知"),
        "tone": _SAFETY_TONES.get(level, "muted"),
    }


def format_reliability_decision(decision: Any) -> dict:
    """可靠性决策 → 展示 dict（含 has_error / is_final_refusal）。"""
    d = _as_dict(decision)
    if "error" in d:
        return {
            "has_error": True,
            "decision": "error",
            "decision_label": "可靠性判断不可用",
            "reason_code": None,
            "reason": None,
            "triggered_rules": [],
            "confidence": None,
            "is_final_refusal": False,
            "reliability_score": None,
            "uncertainty_level": None,
            "citation_quality": None,
            "explanation": None,
            "disclaimer_needed": False,
            "note": str(d.get("message", ""))[:200],
        }
    dec = d.get("decision")
    return {
        "has_error": False,
        "decision": dec,
        "decision_label": _DECISION_LABELS.get(dec, str(dec or "unknown")),
        "reason_code": d.get("reason_code"),
        "reason": d.get("reason"),
        "triggered_rules": list(d.get("triggered_rules", []) or []),
        "confidence": d.get("confidence"),
        "is_final_refusal": bool(d.get("is_final_refusal")),
        "reliability_score": d.get("reliability_score"),
        "uncertainty_level": d.get("uncertainty_level"),
        "citation_quality": d.get("citation_quality"),
        "explanation": d.get("explanation"),
        "disclaimer_needed": bool(d.get("disclaimer_needed")),
        "note": d.get("note"),
    }


def format_citations(citation_analysis: Any) -> dict:
    """来源引用 → 展示 dict。

    输入为 final_response["citations"]（B3 citation_analysis asdict 或 error dict）。
    """
    c = _as_dict(citation_analysis)
    if "error" in c:
        return {
            "has_error": True,
            "present": False,
            "items": [],
            "grounded_ratio": None,
            "grounded_count": 0,
            "ungrounded_count": 0,
            "total": 0,
            "note": "来源引用分析不可用",
        }
    grounded_set = set(c.get("grounded_entities", []) or [])
    items: list[dict] = []
    for ref in c.get("reference_items", []) or []:
        if isinstance(ref, dict):
            ent = str(ref.get("entity_name", "") or "")
            items.append({
                "raw_text": str(ref.get("raw_text", "") or ""),
                "entity_name": ent,
                "grounded": bool(ent and ent in grounded_set),
                "source_file": ref.get("source_file"),
                "entity_type": ref.get("entity_type"),
            })
    return {
        "has_error": False,
        "present": bool(c.get("citation_present")),
        "items": items,
        "grounded_ratio": c.get("grounded_ratio"),
        "grounded_count": len(grounded_set),
        "ungrounded_count": len(set(c.get("ungrounded_entities", []) or [])),
        "total": c.get("total_references", len(items)),
        "grounding_method": c.get("grounding_method"),
        "note": None,
    }


def format_coverage(response: Any) -> dict:
    """知识覆盖 → 展示 dict（Neo4j 证据 / notes / mentions / evidence_sources 计数）。"""
    r = _as_dict(response)
    cov = _as_dict(r.get("coverage"))
    fused = _as_dict(r.get("fused_context"))
    ev = _as_dict(fused.get("evidence_sources"))
    lr_graph = _as_dict(ev.get("lightrag_graph"))
    n4 = _as_dict(ev.get("neo4j_graph"))

    # 匹配到的实体名（来自 evidence 条目，去重保序，脱敏）
    entity_names: list[str] = []
    seen: set[str] = set()
    for it in fused.get("evidence") or []:
        if not isinstance(it, dict):
            continue
        if it.get("kind") in ("lightrag_entity", "neo4j_node"):
            nm = str(it.get("name", "") or "")
            if nm and nm not in seen:
                seen.add(nm)
                entity_names.append(nm)

    return {
        "neo4j_has_evidence": bool(cov.get("neo4j_has_evidence")),
        "neo4j_unavailable": bool(cov.get("neo4j_unavailable")),
        "notes": list(cov.get("notes", []) or []),
        "mentions": list(r.get("mentions", []) or []),
        "entity_names": entity_names,
        "lightrag_vector_chunks": _int(ev.get("lightrag_vector")),
        "lightrag_graph_entities": _int(lr_graph.get("entities")),
        "lightrag_graph_relationships": _int(lr_graph.get("relationships")),
        "neo4j_nodes": _int(n4.get("nodes")),
        "neo4j_edges": _int(n4.get("edges")),
        "neo4j_triples": _int(n4.get("triples")),
    }


def format_system_status(response: Any) -> dict:
    """系统状态 → {neo4j_status, neo4j_label, llm_generation_count, agent_status}。

    Neo4j 状态与 format_neo4j_evidence 统一（四态，单一来源）。
    """
    r = _as_dict(response)
    n4 = format_neo4j_evidence(r)
    agent_status = "error" if r.get("status") == "error" else "ok"
    return {
        "neo4j_status": n4["status"],
        "neo4j_label": n4["label"],
        "llm_generation_count": _int(r.get("llm_generation_count")),
        "agent_status": agent_status,
    }


# ============================================================
# P3: 证据展示（只消费 fused_context，不重新检索 / 不伪造）
# ============================================================

def _provenance_labels(sources: Any) -> list[str]:
    """source key → 可读标签（缺失返回空列表）。"""
    out: list[str] = []
    for s in sources or []:
        if isinstance(s, str):
            out.append(_SOURCE_LABELS.get(s, s))
    return out


def _evidence_description(item: dict) -> str:
    """按 kind 生成证据内容摘要（仅取已有字段，脱敏）。"""
    kind = item.get("kind")
    p = _as_dict(item.get("payload"))
    text = ""
    if kind == "neo4j_triple":
        text = (f"{p.get('canonical_source', p.get('source', ''))} "
                f"--{p.get('relation_type', '')}--> "
                f"{p.get('canonical_target', p.get('target', ''))}")
    elif kind == "neo4j_node":
        text = str(p.get("name", "") or "")
    elif kind == "lightrag_entity":
        text = str(p.get("description", "") or "")[:100]
    elif kind == "lightrag_relation":
        text = str(p.get("description", "") or "")[:100]
    elif kind == "lightrag_chunk":
        content = str(p.get("content", "") or "")
        text = content[:120] + ("…" if len(content) > 120 else "")
    return mask_sensitive(text)


def _display_evidence_item(item: dict) -> dict:
    """单条 evidence → 展示 dict（脱敏，不 dump 原始对象）。"""
    kind = str(item.get("kind", "") or "")
    p = _as_dict(item.get("payload"))
    name = mask_sensitive(str(item.get("name", "") or ""))
    return {
        "kind": kind,
        "kind_label": _EVIDENCE_KIND_LABELS.get(kind, kind or "unknown"),
        "name": name,
        "description": _evidence_description(item),
        "provenance": _provenance_labels(item.get("sources")),
        "priority": item.get("priority"),
        "chunk_id": mask_sensitive(str(p.get("chunk_id") or p.get("id") or "") or ""),
    }


def format_evidence_overview(response: Any) -> dict:
    """来源概览：是否使用证据、来自哪些来源、Neo4j 状态、mention 数。

    仅如实反映 evidence_sources / coverage，不添加可信度结论。
    """
    r = _as_dict(response)
    fused = _as_dict(r.get("fused_context"))
    ev = _as_dict(fused.get("evidence_sources"))
    lr_graph = _as_dict(ev.get("lightrag_graph"))
    cov = _as_dict(r.get("coverage"))

    lightrag_has = (_int(ev.get("lightrag_vector"))
                    + _int(lr_graph.get("entities"))
                    + _int(lr_graph.get("relationships"))) > 0
    neo4j_has = bool(cov.get("neo4j_has_evidence"))
    neo4j_unavailable = bool(cov.get("neo4j_unavailable")) or bool(r.get("neo4j_unavailable"))

    sources: list[str] = []
    if lightrag_has:
        sources.append("LightRAG")
    if neo4j_has:
        sources.append("Neo4j")

    return {
        "has_evidence": bool(lightrag_has or neo4j_has),
        "source_labels": sources,
        "source_count": len(sources),
        "lightrag_has": lightrag_has,
        "neo4j_has": neo4j_has,
        "neo4j_unavailable": neo4j_unavailable,
        "mention_count": len(r.get("mentions") or []),
    }


def format_lightrag_evidence(response: Any, max_items: int = _DEFAULT_EVIDENCE_MAX) -> dict:
    """LightRAG / 文档证据展示（默认最多 max_items 条，不改变原始 evidence 排序/内容）。"""
    fused = _as_dict(_as_dict(response).get("fused_context"))
    items = [_display_evidence_item(it) for it in (fused.get("evidence") or [])
             if isinstance(it, dict) and it.get("kind") in _LIGHTRAG_KINDS]
    total = len(items)
    return {
        "items": items[:max(max_items, 0)],
        "total": total,
        "max_items": max_items,
        "has_more": total > max_items,
    }


def format_neo4j_evidence(response: Any) -> dict:
    """Neo4j 图谱证据：四状态（not_checked / unavailable / hit / no_hit）+ 计数 + 结构化 triple。

    优先级：not_checked > unavailable > hit > no_hit。
    - not_checked：safety_level == "blocked"（安全拦截，未执行图谱检索，优先级最高）
    - unavailable：非 BLOCKED，尝试连接但不可用
    - hit：非 BLOCKED 且可用且有真实证据
    - no_hit：非 BLOCKED 且可用但未命中

    计数来自 evidence_sources.neo4j_graph（后端事实）；triples 来自 evidence 条目（仅展示已有）。
    """
    r = _as_dict(response)
    fused = _as_dict(r.get("fused_context"))
    ev = _as_dict(fused.get("evidence_sources"))
    n4 = _as_dict(ev.get("neo4j_graph"))
    cov = _as_dict(r.get("coverage"))

    blocked = str(r.get("safety_level") or "").lower() == "blocked"
    unavailable = bool(cov.get("neo4j_unavailable")) or bool(r.get("neo4j_unavailable"))
    has = bool(cov.get("neo4j_has_evidence"))

    triples: list[dict] = []
    for it in fused.get("evidence") or []:
        if not isinstance(it, dict):
            continue
        if it.get("kind") == "neo4j_triple":
            p = _as_dict(it.get("payload"))
            triples.append({
                "source": mask_sensitive(str(p.get("canonical_source", p.get("source", "")) or "")),
                "relation": str(p.get("relation_type", "") or ""),
                "target": mask_sensitive(str(p.get("canonical_target", p.get("target", "")) or "")),
            })

    if blocked:
        status = "not_checked"       # 安全拦截 → 未执行图谱检索（即使 neo4j_unavailable=False）
    elif unavailable:
        status = "unavailable"
    elif has:
        status = "hit"
    else:
        status = "no_hit"

    return {
        "status": status,
        "label": _NEO4J_STATUS_LABELS.get(status, status),
        "unavailable": unavailable,
        "has_evidence": has,
        "nodes": _int(n4.get("nodes")),
        "edges": _int(n4.get("edges")),
        "triples_count": _int(n4.get("triples")),
        "triples": triples,
    }


def build_kg_graph_data(response: Any, max_nodes: int = 20, max_edges: int = 20) -> dict:
    """构建 KG 可视化图数据（仅 hit 态；纯函数，不修改 response / fused_context）。

    数据只来自 `fused_context.evidence` 中 `sources` 含 `"neo4j_graph"` 的条目：
    - 节点：`neo4j_node` evidence + triples 的 canonical_source/canonical_target 并集（去重）
    - 边：`neo4j_triple` evidence → {source, target, label=relation_type}
    禁止根据 mentions / question 推断或拼接关系；未命中 / 不可用 / 安全拦截一律 has_graph=False。

    Args:
        response: final_response。
        max_nodes / max_edges: 展示上限（默认 20）。

    Returns:
        {"status", "has_graph", "nodes", "edges", "node_total", "node_truncated",
         "edge_total", "edge_truncated", "has_more"}
    """
    n4 = format_neo4j_evidence(response)
    if n4["status"] != "hit":
        return {
            "status": n4["status"], "has_graph": False,
            "nodes": [], "edges": [],
            "node_total": 0, "node_truncated": 0,
            "edge_total": 0, "edge_truncated": 0, "has_more": False,
        }

    fused = _as_dict(_as_dict(response).get("fused_context"))
    evidence = fused.get("evidence") or []

    node_map: dict[str, dict] = {}
    edges_raw: list[dict] = []
    seen_edges: set[tuple] = set()

    def _mk_node(raw: str, extra_title: str = "") -> dict:
        # id/label/title 均脱敏，避免敏感名泄漏到 pyvis HTML
        safe = mask_sensitive(raw)
        title = mask_sensitive(raw + (f"（{extra_title}）" if extra_title else ""))
        return {"id": safe, "label": safe, "title": title}

    for it in evidence:
        if not isinstance(it, dict):
            continue
        if "neo4j_graph" not in (it.get("sources") or []):
            continue
        kind = it.get("kind")
        p = _as_dict(it.get("payload"))
        if kind == "neo4j_node":
            nm = str(p.get("name") or p.get("id") or it.get("name") or "")
            if nm:
                node_map.setdefault(nm, _mk_node(nm, str(p.get("entity_type") or "")))
        elif kind == "neo4j_triple":
            s = str(p.get("canonical_source", p.get("source", "")) or "")
            t = str(p.get("canonical_target", p.get("target", "")) or "")
            rel = str(p.get("relation_type", "") or "")
            if s and t:
                node_map.setdefault(s, _mk_node(s))
                node_map.setdefault(t, _mk_node(t))
                ekey = (s, rel, t)
                if ekey not in seen_edges:
                    seen_edges.add(ekey)
                    edges_raw.append({
                        "source": mask_sensitive(s),
                        "target": mask_sensitive(t),
                        "label": rel,
                    })

    nodes = list(node_map.values())
    node_total = len(nodes)
    edge_total = len(edges_raw)
    shown_nodes = nodes[:max_nodes]
    shown_ids = {nd["id"] for nd in shown_nodes}
    shown_edges = [e for e in edges_raw if e["source"] in shown_ids and e["target"] in shown_ids][:max_edges]
    node_truncated = node_total - len(shown_nodes)
    edge_truncated = edge_total - len(shown_edges)

    return {
        "status": "hit", "has_graph": True,
        "nodes": shown_nodes, "edges": shown_edges,
        "node_total": node_total, "node_truncated": node_truncated,
        "edge_total": edge_total, "edge_truncated": edge_truncated,
        "has_more": bool(node_truncated or edge_truncated),
    }


# ============================================================
# 心理健康热线展示校正 + KG 居中（Phase 9 P0）
# ============================================================

_MENTAL_HOTLINE_TRIGGERS = ("心理健康提醒", "心理援助热线", "心理危机")
_BEIJING_HOTLINE_MARKERS = ("北京心理危机", "010-82951332")
_NATIONAL_HOTLINE_BULLET = "- 全国统一心理援助热线：12356"


def normalize_mental_health_hotline(text: str) -> str:
    """心理危机拦截文案的 UI 层展示校正（纯函数，可单测）。

    仅当文本命中心理健康/心理热线内容时生效：
    1. 删除北京属地热线行（"北京心理危机研究与干预中心：010-82951332"）；
    2. 若文案尚无全国统一心理援助热线（12356，国家卫健委 2025-05-01 起全国启用），
       在首个"- 热线" bullet 后补一行。

    Phase 9 P0 修复：拦截提示不应携带属地热线；此为展示层修正，
    不改 src/medical_core/safety.py（SafetyChecker 消息源保持原样）。
    """
    if not text:
        return text
    if not any(m in text for m in _MENTAL_HOTLINE_TRIGGERS):
        return text
    lines = [ln for ln in text.splitlines() if not any(m in ln for m in _BEIJING_HOTLINE_MARKERS)]
    if "12356" in "".join(lines):
        return "\n".join(lines)
    out: list[str] = []
    inserted = False
    for ln in lines:
        out.append(ln)
        if not inserted and ln.startswith("- ") and "热线" in ln:
            out.append(_NATIONAL_HOTLINE_BULLET)
            inserted = True
    if not inserted:
        out.append(_NATIONAL_HOTLINE_BULLET)
    return "\n".join(out)


_KG_CENTER_SNIPPET = (
    "\n            // Phase 9 P0: 图谱居中（禁用物理；等画布可见后再 fit，避免折叠 expander 0x0 画布上无效 fit）\n"
    "            if (typeof network !== 'undefined') {\n"
    "                network.setOptions({ physics: { enabled: false } });\n"
    "                (function(){\n"
    "                  var fitted = false;\n"
    "                  function doFit(){ if (fitted) return; fitted = true; try { network.fit({ animation: false }); } catch(e){} }\n"
    "                  function sizeOk(){\n"
    "                    var cv = document.querySelector('#mynetwork canvas');\n"
    "                    return cv && cv.width > 0 && cv.height > 0;\n"
    "                  }\n"
    "                  if (sizeOk()) { doFit(); return; }\n"
    "                  try { network.once('resize', doFit); } catch(e){}\n"
    "                  var n = 0;\n"
    "                  var timer = setInterval(function(){\n"
    "                    n++;\n"
    "                    if (sizeOk()) { doFit(); clearInterval(timer); }\n"
    "                    else if (n > 60) { doFit(); clearInterval(timer); }\n"
    "                  }, 300);\n"
    "                })();\n"
    "            }\n"
)


def inject_kg_centering(html: str) -> str:
    """在 pyvis HTML 主脚本末尾（drawGraph() 之后、同一作用域）注入居中逻辑（纯字符串，可单测）。

    Phase 9 P0 修复：pyvis 默认 physics.enabled=true，稳定化 fit 后物理仍持续模拟，节点漂移
    导致图谱默认偏到容器角落。本函数把"禁用物理 + network.fit()"注入到 pyvis 主脚本的
    **最后一个 <script> 块内**（与 network 同一作用域，紧随 drawGraph() 调用之后执行），
    消除跨脚本作用域 / 事件时序的全部不确定性：
      - 禁用物理 → 图谱冻结不再漂移（仍可拖拽查看）；
      - network.fit() → 视图居中到图谱包围盒。
    幂等：已含注入标记的 HTML 不重复注入。
    """
    if not html:
        return html
    if "Phase 9 P0: 图谱居中" in html:
        return html
    idx = html.rfind("</script>")
    if idx != -1:
        return html[:idx] + _KG_CENTER_SNIPPET + html[idx:]
    return html + _KG_CENTER_SNIPPET


def build_kg_layout(nodes: list, edges: list, canvas_w: float = 560,
                    canvas_h: float = 420, seed: int = 42) -> dict:
    """用 networkx 计算 KG 节点确定性布局并归一化居中（纯函数，可单测）。

    返回 {node_id: {"x": float, "y": float}}；坐标以画布中心为原点，已缩放适配画布。
    配合 pyvis 固定坐标 + 关闭物理，保证图谱在容器中居中显示且不漂移。

    - 确定性：固定 seed（同一数据 → 同一布局），避免每次刷新图都变；
    - 居中：布局包围盒中心平移到原点，并按 min(画布宽/高) 缩放（留 15% 边距）;
    - 依赖：networkx（LightRAG 既有依赖），懒加载，不影响本模块轻量 import。
    """
    import networkx as nx

    ids = [str(n.get("id")) for n in nodes if n.get("id")]
    if not ids:
        return {}
    g = nx.Graph()
    g.add_nodes_from(ids)
    for e in edges:
        s, t = str(e.get("source") or ""), str(e.get("target") or "")
        if s and t and s in ids and t in ids:
            g.add_edge(s, t)
    pos = nx.spring_layout(g, seed=seed, k=1.5, iterations=100)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    w = max(max(xs) - min(xs), 1e-6)
    h = max(max(ys) - min(ys), 1e-6)
    scale = min(canvas_w / w, canvas_h / h) * 0.85
    return {
        nid: {"x": float((x - cx) * scale), "y": float((y - cy) * scale)}
        for nid, (x, y) in pos.items()
    }


def build_display_sections(response: Any) -> dict:
    """输入 final_response → 输出 UI 展示分区（answer/safety/decision/citations/coverage/system/debug）。

    所有字段安全兜底；response 为 None 或缺失字段不会抛异常。
    """
    r = _as_dict(response)
    answer = normalize_mental_health_hotline(str(r.get("answer", "") or ""))
    sections = {
        "answer": answer,
        "safety": format_safety_level(r.get("safety_level")),
        "decision": format_reliability_decision(r.get("reliability_decision")),
        "evidence_overview": format_evidence_overview(r),
        "lightrag_evidence": format_lightrag_evidence(r),
        "neo4j_evidence": format_neo4j_evidence(r),
        "kg_graph": build_kg_graph_data(r),
        "citations": format_citations(r.get("citations")),
        "coverage": format_coverage(r),
        "system": format_system_status(r),
        "is_error": r.get("status") == "error",
        "errors": sanitize_errors(r.get("errors")),
    }
    # 调试信息（默认折叠展示，已清洗）
    sections["debug"] = {
        "intent": r.get("intent"),
        "mentions": list(r.get("mentions", []) or []),
        "neo4j_unavailable": bool(r.get("neo4j_unavailable")),
        "llm_generation_count": _int(r.get("llm_generation_count")),
        "reliability_decision": format_reliability_decision(r.get("reliability_decision")),
        "coverage": format_coverage(r),
        "errors": sections["errors"],
    }
    return sections


def _int(v: Any) -> int:
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0
