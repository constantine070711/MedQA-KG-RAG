# -*- coding: utf-8 -*-
"""Phase 8 P2 — MedQA Streamlit Web UI 入口（聊天式对话）。

启动：
    streamlit run app/streamlit_app.py

职责：
    - 只负责 UI 展示与交互，通过 MedQAAgent.run() 完成问答
    - 不直接调用 LightRAG / Neo4j / LLM / ReliabilityDecision / Evidence Fusion
    - 不在 UI 层复制任何安全判断 / 拒答 / 引用 / 融合逻辑
    - Session State 懒创建并复用 Agent（+ 持久事件循环），rerun 不重建
    - 对话历史仅 UI 展示，不拼接进 Agent.question（每轮独立 agent.run）
    - 医疗免责声明常驻顶部；Neo4j 不可用如实展示；异常脱敏展示
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.ui_helpers import (
    add_message,
    build_conversation_context,
    build_display_sections,
    build_kg_layout,
    inject_kg_centering,
    lazy_init,
    memory_used_hint,
    reset_chat_state,
    resolve_question,
    run_agent_sync,
    sanitize_errors,
)

DISCLAIMER = (
    "本系统仅提供医疗知识辅助，不替代医生诊断、处方或专业医疗意见。"
    "如存在紧急情况，请及时寻求专业医疗帮助（紧急情况请拨打 120）。"
)

ERROR_USER_MSG = "本次请求暂时无法完成，请稍后重试。"

EXAMPLES = [
    "2型糖尿病的常见症状有哪些？",
    "二甲双胍用于治疗什么疾病？",
    "高血压有哪些常见并发症？",
    "高血压和脑卒中有什么关系？",
]

_TONE_FN = {
    "ok": lambda msg: st.success(msg),
    "warn": lambda msg: st.warning(msg),
    "danger": lambda msg: st.error(msg),
    "muted": lambda msg: st.info(msg),
}


# ============================================================
# Session State
# ============================================================

def _init_state() -> None:
    """初始化 UI 会话状态（只属 UI 层，不改 Agent 单轮逻辑）。"""
    st.session_state.setdefault("messages", [])       # [{"role","content","response"?}]
    st.session_state.setdefault("history", [])        # [{"question","final_response"}]
    st.session_state.setdefault("last_response", None)
    # Phase 9 P4: 是否已发起过查询（用于冷启动提示；清空对话不清此标志，Agent 仍在）
    st.session_state.setdefault("_has_queried", False)


def _get_agent():
    """懒创建并复用 MedQAAgent（session_state 缓存，rerun/清空对话均不重建）。"""
    from src.agent import MedQAAgent
    return lazy_init(st.session_state, "agent", MedQAAgent)


# ============================================================
# 展示
# ============================================================

def _render_sections(sections: dict, debug: bool = True) -> None:
    """按 回答→安全→可靠性→来源→覆盖→系统→调试 顺序展示 final_response。

    Args:
        sections: build_display_sections() 输出。
        debug: 是否展示调试 expander（历史轮次传 False，最新轮次传 True）。
    """
    # ---- 回答 ----
    st.markdown(sections["answer"] or "_（无回答）_")

    # ---- 安全等级 ----
    s = sections["safety"]
    _TONE_FN.get(s["tone"], st.info)(f"安全等级：{s['level'].upper()} — {s['label']}")

    # ---- 可靠性判断 ----
    d = sections["decision"]
    if d["is_final_refusal"]:
        st.error("本次为拒答，未提供实质性医学内容。")
    elif d["has_error"]:
        st.info("可靠性判断不可用。")
    else:
        conf = d.get("confidence")
        conf_txt = f"（置信度 {float(conf):.0%}）" if isinstance(conf, (int, float)) else ""
        st.markdown(f"**可靠性判断**：{d['decision_label']}{conf_txt}")
        if d.get("reason"):
            st.caption(d["reason"])

    # ---- 证据与来源 ----
    st.markdown("**证据与来源**")

    # 来源概览
    ov = sections["evidence_overview"]
    if ov["has_evidence"]:
        src_txt = "、".join(ov["source_labels"])
        overview = f"本回答使用了 **{ov['source_count']} 类知识来源**：{src_txt}。"
        if ov["mention_count"]:
            overview += f"（图谱命中实体 {ov['mention_count']} 个）"
        st.markdown(overview)
    else:
        st.caption("本回答未产生可展示的结构化证据（可能基于向量检索或拒答/拦截）。")

    # LightRAG / 文档证据（可折叠，默认前 5 条）
    lr = sections["lightrag_evidence"]
    if lr["total"]:
        with st.expander(f"文档 / 检索证据（{lr['total']} 条）"):
            for it in lr["items"]:
                st.markdown(f"- **[{it['kind_label']}]** {it['description'] or it['name']}")
                prov = "，".join(it["provenance"]) if it["provenance"] else "—"
                meta = f"来源：{prov}"
                if it["chunk_id"]:
                    meta += f" | chunk: {it['chunk_id']}"
                st.caption(meta)
            if lr["has_more"]:
                st.caption(f"… 还有 {lr['total'] - lr['max_items']} 条未展示（为控制页面长度）。")

    # Neo4j 图谱证据（四状态：not_checked / unavailable / hit / no_hit）
    n4 = sections["neo4j_evidence"]
    if n4["status"] == "not_checked":
        st.warning("安全拦截，未执行图谱检索。")
    elif n4["status"] == "unavailable":
        st.info("Neo4j 当前不可用，本次回答未使用结构化图谱证据。")
    elif n4["status"] == "hit":
        st.success(
            f"Neo4j 结构化图谱证据：{n4['nodes']} 节点 / "
            f"{n4['edges']} 关系 / {n4['triples_count']} 三元组。"
        )
        if n4["triples"]:
            with st.expander(f"图谱三元组（{len(n4['triples'])} 条）"):
                for t in n4["triples"]:
                    st.markdown(f"- {t['source']} → **{t['relation']}** → {t['target']}")
    else:
        st.caption("本问题未命中当前知识图谱中的结构化实体/关系。")

    # KG 可视化（仅 hit 态，默认折叠；pyvis 本地资源，不依赖 CDN）
    kg = sections["kg_graph"]
    if kg["has_graph"]:
        with st.expander("知识图谱可视化"):
            try:
                from pyvis.network import Network
                # Phase 9 P0: networkx 确定性布局（居中）+ 固定坐标 + 关闭物理，避免图谱偏角落/漂移
                positions = build_kg_layout(kg["nodes"], kg["edges"])
                # Phase 9 P0: 内联 vis.js（in_line），自包含无 CDN —— 消除中国网络下 cdnjs 加载不稳导致图谱渲染异常
                net = Network(height="480px", width="100%", directed=True, notebook=False)
                net.cdn_resources = "in_line"
                for nd in kg["nodes"]:
                    p = positions.get(nd["id"], {})
                    net.add_node(
                        nd["id"],
                        label=nd["label"],
                        title=nd.get("title") or nd["label"],
                        x=p.get("x"),
                        y=p.get("y"),
                        physics=False,
                    )
                for ed in kg["edges"]:
                    net.add_edge(ed["source"], ed["target"], label=ed["label"], title=ed["label"])
                html = net.generate_html(name="medqa_kg.html", local=True, notebook=False)
                st.components.v1.html(inject_kg_centering(html), height=520, scrolling=False)
                if kg["has_more"]:
                    st.caption(
                        f"图谱较大，仅展示前 {len(kg['nodes'])} 个节点 / {len(kg['edges'])} 条关系"
                        f"（原始 {kg['node_total']} 节点 / {kg['edge_total']} 关系，"
                        f"截断 {kg['node_truncated']} 节点 / {kg['edge_truncated']} 关系）。"
                    )
            except Exception:
                st.warning("知识图谱可视化渲染失败（文本三元组与回答不受影响）。")

    # 来源引用
    c = sections["citations"]
    if c["present"] and c["items"]:
        for it in c["items"]:
            g = "已追溯" if it["grounded"] else "未追溯"
            line = f"- {it['raw_text'] or it['entity_name']}（{g}）"
            if it.get("source_file"):
                line += f" · 来源文件：{it['source_file']}"
            st.markdown(line)
        if c["grounded_ratio"] is not None:
            ratio_txt = f"引用可追溯比例：{float(c['grounded_ratio']):.0%}"
            if c.get("grounding_method"):
                ratio_txt += f"（{c['grounding_method']}）"
            st.caption(ratio_txt)
    else:
        st.caption("本回答没有生成可展示的来源引用。")

    # 覆盖情况
    cov = sections["coverage"]
    for n in cov["notes"]:
        st.caption(f"覆盖说明：{n}")
    if cov["mentions"]:
        st.caption(f"图谱命中实体：{', '.join(cov['mentions'])}")
    elif cov["entity_names"]:
        st.caption(f"检索匹配实体：{', '.join(cov['entity_names'])}")
    st.caption(
        f"检索覆盖：向量块 {cov['lightrag_vector_chunks']} | "
        f"图谱实体 {cov['lightrag_graph_entities']} | "
        f"图谱关系 {cov['lightrag_graph_relationships']}"
    )

    # ---- 系统状态 ----
    sys_st = sections["system"]
    st.markdown("**系统状态**")
    st.caption(f"Neo4j：{sys_st['neo4j_label']}")
    st.caption(f"LLM Generation：{sys_st['llm_generation_count']}")

    # ---- 调试信息（默认折叠，已清洗）----
    if debug:
        with st.expander("调试信息"):
            st.json(sections["debug"])


def _render_error(fr: dict, debug: bool = True) -> None:
    """Agent 错误信封 → 安全展示（不泄漏 traceback / API / prompt）。"""
    st.error(ERROR_USER_MSG)
    if debug:
        with st.expander("调试信息"):
            st.json({"errors": sanitize_errors(fr.get("errors"))})


def _render_history_message(msg: dict, is_latest: bool = False) -> None:
    """渲染一条历史消息。assistant 消息按保存的 response 结构化回显（默认不展开调试）。"""
    role = msg.get("role")
    with st.chat_message(role):
        if role == "assistant" and msg.get("response") is not None:
            resp = msg["response"]
            if isinstance(resp, dict) and resp.get("status") == "error":
                _render_error(resp, debug=is_latest)
            else:
                _render_sections(build_display_sections(resp), debug=is_latest)
        else:
            st.markdown(msg.get("content", ""))


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    st.set_page_config(page_title="MedQA 医疗知识智能问答系统", page_icon="🩺", layout="wide")

    st.title("MedQA — 基于知识图谱与 RAG 的医疗智能问答系统")
    st.warning(DISCLAIMER)

    _init_state()
    agent = _get_agent()

    # 侧边栏：清空对话（保留 Agent）+ 示例问题
    example = None
    with st.sidebar:
        st.markdown("## 操作")
        if st.button("清空对话", use_container_width=True):
            _fresh = reset_chat_state()
            st.session_state["messages"] = _fresh["messages"]
            st.session_state["history"] = _fresh["history"]
            st.session_state["last_response"] = _fresh["last_response"]
            st.rerun()
        st.markdown("---")
        st.markdown("### 你可以这样问")
        for ex in EXAMPLES:
            if st.button(ex, key=f"example_{ex}", use_container_width=True):
                example = ex
        st.markdown("---")
        with st.expander("关于本系统"):
            st.markdown(
                "基于 **知识图谱 + RAG** 的医疗智能问答系统。\n\n"
                "- **检索**：LightRAG（向量 + 图混合）\n"
                "- **图谱**：Neo4j（TREATS / HAS_COMPLICATION 结构化临床知识）\n"
                "- **工作流**：LangGraph Agent（安全 → 检索 → 生成 → 可靠性决策）\n"
                "- **安全**：分级拦截 + 诊断/剂量拒答 + 来源引用\n\n"
                "⚠️ 仅供医疗知识辅助，不替代医生诊断。"
            )

    # 历史消息（仅 UI 展示，不拼接入 Agent.question）
    msgs = st.session_state["messages"]
    for msg in msgs:
        _render_history_message(msg)

    # 输入：chat_input 优先，其次示例按钮触发
    prompt = st.chat_input("请输入您的医疗知识问题…")
    question = resolve_question(prompt, example)

    if question is None:
        # 仅在用户提交了空输入时提示，避免初始加载无意义打扰
        if prompt is not None and not str(prompt).strip():
            st.info("请输入你的问题。")
        return

    # ---- 一轮问答（唯一一次 Agent 调用；内部 1 次 generation，BLOCKED 0 次）----
    # Phase 9 P2: 由既有 history 构建多轮 Memory（仅作 generation prompt 背景）。
    # 当前轮尚未追加进 history → 不会进入本轮的 conversation_context。
    memory = build_conversation_context(st.session_state["history"])
    st.session_state["messages"] = add_message(
        st.session_state["messages"], "user", question,
    )
    with st.chat_message("user"):
        st.markdown(question)

    # Phase 9 P4: 首次查询冷启动（LightRAG 内核加载约 10-40 秒）提示
    first_query = not st.session_state["_has_queried"]
    with st.chat_message("assistant"):
        spin_text = (
            "正在检索医疗知识并生成回答…（首次查询需加载检索内核，约 10-40 秒，请耐心等待）"
            if first_query else "正在检索医疗知识并生成回答…"
        )
        with st.spinner(spin_text):
            fr = run_agent_sync(agent, question, conversation_context=memory)
        st.session_state["_has_queried"] = True

        st.session_state["last_response"] = fr
        st.session_state["history"] = st.session_state["history"] + [
            {"question": question, "final_response": fr}
        ]

        if fr.get("status") == "error":
            _render_error(fr, debug=True)
            assistant_content = ERROR_USER_MSG
        else:
            _render_sections(build_display_sections(fr), debug=True)
            assistant_content = fr.get("answer") or ""

        # Phase 9 P4: 本轮传入了前文上下文且为实质回答 → 轻量提示（仅 UI，不改 final_response）
        hint = memory_used_hint(memory, fr)
        if hint:
            st.caption(hint)

    # 记录助手消息（content + response 用于历史结构化回显）
    st.session_state["messages"] = add_message(
        st.session_state["messages"], "assistant", assistant_content, response=fr,
    )


if __name__ == "__main__":
    main()
