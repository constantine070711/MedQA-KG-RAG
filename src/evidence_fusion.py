"""Phase 6 F5 — Evidence Fusion（证据融合层，纯函数，零 LLM / 零 DB / 零副作用）。

职责:
    - 融合 LightRAG 检索证据（entities / relationships / chunks）与
      Neo4jGraphRetriever 结构化证据（nodes / edges / triples）
    - 按 identity 去重（非整段字符串）、跨来源实体合并并保留全部 provenance
    - 确定性排序（priority 0-4）+ 结构化截断（先截断列表再渲染，不破坏格式）
    - 覆盖不足如实记录（coverage.notes），不伪造任何实体 / 关系
    - 生成可注入 system_prompt 的 Neo4j 证据文本段（仅追加证据，不改变安全指令）

约束（对齐 Phase 6 F5 需求）:
    - 禁止 import LLM / embedding
    - 禁止数据库查询 / 网络调用
    - 禁止修改任何已有模块（rag_engine / evaluation / medical_core 等）

不实现:
    - 查询编排（scripts/run_graph_query.py）
    - 端到端评估（F6）
    - 任何 LLM / embedding 调用
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from evaluation.metrics import load_entity_name_map, resolve_entity_name


# ============================================================
# 常量
# ============================================================

# provenance / evidence source（固定枚举）
SOURCE_LIGHTRAG_VECTOR = "lightrag_vector"
SOURCE_LIGHTRAG_GRAPH = "lightrag_graph"
SOURCE_NEO4J_GRAPH = "neo4j_graph"

# evidence kind（固定枚举）
KIND_NEO4J_TRIPLE = "neo4j_triple"
KIND_NEO4J_NODE = "neo4j_node"
KIND_LIGHTRAG_ENTITY = "lightrag_entity"
KIND_LIGHTRAG_RELATION = "lightrag_relation"
KIND_LIGHTRAG_CHUNK = "lightrag_chunk"

# priority（0 最高，5 类）
KIND_PRIORITY: dict[str, int] = {
    KIND_NEO4J_TRIPLE: 0,          # Neo4j 结构化临床三元组（最高）
    KIND_NEO4J_NODE: 1,            # Neo4j curated 节点（Disease/Drug）
    KIND_LIGHTRAG_RELATION: 2,     # LightRAG 图关系
    KIND_LIGHTRAG_ENTITY: 3,       # LightRAG 实体
    KIND_LIGHTRAG_CHUNK: 4,        # LightRAG 向量文本块（最低）
}

# curated 实体 mention 匹配的最短别名长度（避免过短别名误匹配，如单字符）
MIN_MENTION_LEN = 2

# Neo4j 容器不可用标记 key
NEO4J_UNAVAILABLE_KEY = "unavailable"


# ============================================================
# 配置
# ============================================================

@dataclass
class FusionConfig:
    """证据融合截断配置（先排序后截断，先结构化列表后渲染 prompt）。"""
    max_neo4j_triples: int = 30
    max_neo4j_nodes: int = 40
    max_lightrag_entities: int = 60
    max_lightrag_relations: int = 60
    max_chunks: int = 14
    max_total: int = 100
    max_prompt_chars: int = 4000

    def kind_caps(self) -> dict[str, int]:
        return {
            KIND_NEO4J_TRIPLE: self.max_neo4j_triples,
            KIND_NEO4J_NODE: self.max_neo4j_nodes,
            KIND_LIGHTRAG_ENTITY: self.max_lightrag_entities,
            KIND_LIGHTRAG_RELATION: self.max_lightrag_relations,
            KIND_LIGHTRAG_CHUNK: self.max_chunks,
        }


# ============================================================
# 输出对象
# ============================================================

@dataclass
class EvidenceItem:
    """单条融合证据条目（JSON 可序列化）。

    Attributes:
        id:       去重 identity（canonical 实体键 / 关系键 / chunk_id）
        kind:     证据类型（neo4j_triple / neo4j_node / lightrag_entity /
                  lightrag_relation / lightrag_chunk）
        name:     显示名
        payload:  原始数据（跨源合并时保留多源 payload）
        sources:  provenance 列表（lightrag_vector / lightrag_graph / neo4j_graph）
        priority: 排序优先级（0-4）
    """
    id: str
    kind: str
    name: str
    payload: dict
    sources: list
    priority: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "payload": self.payload,
            "sources": list(self.sources),
            "priority": self.priority,
        }


# ============================================================
# D4 辅助（只读复用，不修改核心逻辑）
# ============================================================

def _is_resolved(name: Any, name_map: dict) -> bool:
    """名称是否在 D4 name_map 中可解析（canonical 或别名均算已解析）。"""
    return (
        isinstance(name, str)
        and bool(name.strip())
        and name.strip().lower() in name_map
    )


def _resolve(name: Any, name_map: dict) -> str:
    """D4 解析：可解析则返回 canonical，否则原样返回（不猜测）。"""
    if _is_resolved(name, name_map):
        return resolve_entity_name(name, name_map)
    return name if isinstance(name, str) else ""


def _default_name_map() -> dict:
    """加载 D4 entities.json name_map（默认路径）。"""
    return load_entity_name_map()


# ============================================================
# 1. extract_curated_mentions —— curated 实体确定性识别（零 LLM）
# ============================================================

def extract_curated_mentions(
    question: str,
    name_map: Optional[dict] = None,
) -> list[str]:
    """从问题文本中确定性识别 curated Disease / Drug 实体（及别名）。

    规则:
        - 使用已有 D4 name_map（canonical + 全部别名，lowercase key）
        - 纯子串匹配，零 LLM
        - 同一位置多个候选时取最长匹配（longest-match-wins）
        - 按首次出现位置排序，canonical 去重
        - 不识别到返回空列表，不强行猜测实体

    Args:
        question: 用户问题
        name_map: D4 load_entity_name_map() 输出（默认自动加载）

    Returns:
        curated 实体 canonical 名称列表（按在问题中出现的先后顺序）
    """
    if not question or not isinstance(question, str):
        return []
    nm = name_map if isinstance(name_map, dict) else _default_name_map()
    if not nm:
        return []

    q = question.lower()
    # 候选: (canonical, matched_surface, position, length)
    candidates: list[tuple[str, str, int, int]] = []
    for lower_surface, canonical in nm.items():
        if not isinstance(lower_surface, str) or not lower_surface:
            continue
        if len(lower_surface) < MIN_MENTION_LEN:
            continue
        pos = q.find(lower_surface)
        if pos >= 0:
            candidates.append((canonical, lower_surface, pos, len(lower_surface)))

    if not candidates:
        return []

    # 同位置取最长匹配；整体按位置升序
    candidates.sort(key=lambda c: (c[2], -c[3]))

    accepted: list[str] = []
    seen_canonical: set[str] = set()
    accepted_intervals: list[tuple[int, int]] = []

    for canonical, _surface, pos, ln in candidates:
        end = pos + ln
        if canonical in seen_canonical:
            continue
        # 与已接受更长的匹配重叠（被更专名覆盖）→ 跳过，避免"哮喘"误匹配"哮喘持续状态"类
        if any(a_pos <= pos < a_end or a_pos < end <= a_end
               for a_pos, a_end in accepted_intervals):
            continue
        accepted.append(canonical)
        seen_canonical.add(canonical)
        accepted_intervals.append((pos, end))

    return accepted


# ============================================================
# 2. Neo4j 证据条目构建
# ============================================================

def _normalize_triples(neo4j_data: dict) -> list[list[str]]:
    """从 neo4j_data 提取合法三元组（triples 优先，edges 兜底推导）。"""
    triples = neo4j_data.get("triples") or []
    edges = neo4j_data.get("edges") or []
    out: list[list[str]] = []
    seen: set[tuple] = set()

    def _push(s: Any, r: Any, o: Any) -> None:
        if not (isinstance(s, str) and isinstance(r, str) and isinstance(o, str)):
            return
        key = (s, r, o)
        if key in seen:
            return
        seen.add(key)
        out.append([s, r, o])

    for t in triples:
        if isinstance(t, (list, tuple)) and len(t) >= 3:
            _push(t[0], t[1], t[2])

    if not out:
        for e in edges:
            if isinstance(e, dict):
                _push(e.get("source"), e.get("relation_type"), e.get("target"))

    return out


def _build_neo4j_items(
    neo4j_data: dict,
    name_map: dict,
) -> tuple[list, dict, list[str]]:
    """构建 Neo4j 证据条目 + 实体索引 + 覆盖说明。

    Returns:
        (items, neo4j_triple_index, notes)
        neo4j_triple_index: {(canonical_src, rel_lower, canonical_tgt): item_idx}
    """
    items: list = []
    notes: list[str] = []
    entity_by_key: dict[str, int] = {}
    triple_index: dict[tuple, int] = {}

    if neo4j_data.get(NEO4J_UNAVAILABLE_KEY):
        notes.append("Neo4j 容器不可用，未产生结构化图谱证据。")
        return items, triple_index, notes

    triples = _normalize_triples(neo4j_data)
    nodes = [nd for nd in (neo4j_data.get("nodes") or []) if isinstance(nd, dict)]

    if neo4j_data.get("note"):
        notes.append(str(neo4j_data["note"]))
    if neo4j_data.get("coverage_note"):
        notes.append(str(neo4j_data["coverage_note"]))

    # ---- triples（priority 0）----
    for s, r, o in triples:
        cs = _resolve(s, name_map)
        ct = _resolve(o, name_map)
        key = f"triple::{cs}::{r}::{ct}"
        if key in entity_by_key:  # triples 用独立命名空间，不会与实体冲突
            continue
        idx = len(items)
        entity_by_key[key] = idx
        triple_index[(cs, r.lower(), ct)] = idx
        items.append(EvidenceItem(
            id=key,
            kind=KIND_NEO4J_TRIPLE,
            name=f"{cs} -{r}-> {ct}",
            payload={
                "source": s,
                "relation_type": r,
                "target": o,
                "canonical_source": cs,
                "canonical_target": ct,
            },
            sources=[SOURCE_NEO4J_GRAPH],
            priority=KIND_PRIORITY[KIND_NEO4J_TRIPLE],
        ))

    # ---- nodes（priority 1）----
    for nd in nodes:
        nm = nd.get("name")
        if not isinstance(nm, str) or not nm:
            continue
        canonical = _resolve(nm, name_map)
        key = f"entity::{canonical}"
        if key in entity_by_key:
            continue
        idx = len(items)
        entity_by_key[key] = idx
        items.append(EvidenceItem(
            id=key,
            kind=KIND_NEO4J_NODE,
            name=canonical,
            payload=nd,
            sources=[SOURCE_NEO4J_GRAPH],
            priority=KIND_PRIORITY[KIND_NEO4J_NODE],
        ))

    if not triples and not nodes:
        notes.append("Neo4j 未产生结构化图谱证据（节点 / 三元组为空）。")

    return items, triple_index, notes


# ============================================================
# 3. LightRAG 证据条目构建（与 Neo4j 实体跨源合并）
# ============================================================

def _build_lightrag_items(
    lightrag_data: dict,
    name_map: dict,
    neo4j_items: list,
    entity_by_key: dict,
    triple_index: dict,
) -> tuple[list, dict, int]:
    """构建 LightRAG 证据条目，并与 Neo4j 实体/三元组做跨源合并。

    Returns:
        (items, relation_by_key, merged_count)
    """
    items: list = []
    relation_by_key: dict[str, int] = {}
    merged = 0

    entities = lightrag_data.get("entities") or []
    relationships = lightrag_data.get("relationships") or []
    chunks = lightrag_data.get("chunks") or []

    # ---- entities（priority 3，跨源合并到 Neo4j 节点）----
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        en = ent.get("entity_name")
        if not isinstance(en, str) or not en:
            continue
        canonical = _resolve(en, name_map)
        key = f"entity::{canonical}"
        existing = entity_by_key.get(key)
        if existing is not None:
            it = neo4j_items[existing] if existing < len(neo4j_items) else None
            if it is not None and SOURCE_LIGHTRAG_GRAPH not in it.sources:
                it.sources.append(SOURCE_LIGHTRAG_GRAPH)
                merged += 1
            # 保留 LightRAG 实体 payload（merged 字段）
            if it is not None and isinstance(it.payload, dict) \
                    and "lightrag_entity" not in it.payload:
                it.payload["lightrag_entity"] = ent
            continue
        idx = len(neo4j_items) + len(items)
        entity_by_key[key] = idx
        items.append(EvidenceItem(
            id=key,
            kind=KIND_LIGHTRAG_ENTITY,
            name=canonical,
            payload=ent,
            sources=[SOURCE_LIGHTRAG_GRAPH],
            priority=KIND_PRIORITY[KIND_LIGHTRAG_ENTITY],
        ))

    # ---- relationships（priority 2，关键词作为关系类型；跨源合并保守）----
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        src_id = rel.get("src_id")
        tgt_id = rel.get("tgt_id")
        if not (isinstance(src_id, str) and isinstance(tgt_id, str)
                and src_id and tgt_id):
            continue
        kw = str(rel.get("keywords") or "").strip()
        src_ok = _is_resolved(src_id, name_map)
        tgt_ok = _is_resolved(tgt_id, name_map)
        cs = _resolve(src_id, name_map) if src_ok else src_id
        ct = _resolve(tgt_id, name_map) if tgt_ok else tgt_id
        key = f"relation::{cs}::{kw}::{ct}"

        # LightRAG 内部去重
        if key in relation_by_key:
            idx = relation_by_key[key]
            it = neo4j_items[idx] if idx < len(neo4j_items) else items[idx - len(neo4j_items)]
            if SOURCE_LIGHTRAG_GRAPH not in it.sources:
                it.sources.append(SOURCE_LIGHTRAG_GRAPH)
                merged += 1
            continue

        # 跨源合并：仅当两端 canonical 均解析成功，且 keywords 与 Neo4j 结构化类型
        # 精确一致（如 "treats" == "TREATS"）时才合并；不确定宁可不合并
        cross_idx = None
        if src_ok and tgt_ok and kw:
            cross_idx = triple_index.get((cs, kw.lower(), ct))
        if cross_idx is not None:
            it = neo4j_items[cross_idx]
            if SOURCE_LIGHTRAG_GRAPH not in it.sources:
                it.sources.append(SOURCE_LIGHTRAG_GRAPH)
                merged += 1
            if isinstance(it.payload, dict) and "lightrag_relation" not in it.payload:
                it.payload["lightrag_relation"] = rel
            relation_by_key[key] = cross_idx
            continue

        idx = len(neo4j_items) + len(items)
        relation_by_key[key] = idx
        items.append(EvidenceItem(
            id=key,
            kind=KIND_LIGHTRAG_RELATION,
            name=f"{cs} -{kw or '?'}-> {ct}",
            payload=rel,
            sources=[SOURCE_LIGHTRAG_GRAPH],
            priority=KIND_PRIORITY[KIND_LIGHTRAG_RELATION],
        ))

    # ---- chunks（priority 4，仅 LightRAG 内部去重）----
    chunk_seen: set[str] = set()
    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        cid = ch.get("chunk_id") or ch.get("id")
        if not isinstance(cid, str) or not cid:
            continue
        if cid in chunk_seen:
            continue
        chunk_seen.add(cid)
        items.append(EvidenceItem(
            id=f"chunk::{cid}",
            kind=KIND_LIGHTRAG_CHUNK,
            name=cid,
            payload=ch,
            sources=[SOURCE_LIGHTRAG_VECTOR],
            priority=KIND_PRIORITY[KIND_LIGHTRAG_CHUNK],
        ))

    return items, relation_by_key, merged


# ============================================================
# 4. 排序 + 截断
# ============================================================

def sort_evidence(items: list) -> list:
    """确定性排序：主键 priority 升序，组内保持构建（输入）顺序。

    Python 的 sort 稳定；为显式表达 tie-break，同时按构建顺序与 id 排序。
    相同输入必须得到完全相同的顺序。
    """
    order = {id(it): i for i, it in enumerate(items)}
    return sorted(
        items,
        key=lambda it: (it.priority, order[id(it)], it.id),
    )


def truncate_evidence(items: list, config: Optional[FusionConfig] = None) -> tuple[list, int]:
    """按 kind 上限 + 总上限截断（先排序后截断，保持高优先级）。

    Returns:
        (kept_items, dropped_count)
    """
    cfg = config or FusionConfig()
    caps = cfg.kind_caps()
    kept: list = []
    kind_count: dict[str, int] = {}
    dropped = 0

    for it in items:
        cap = caps.get(it.kind, cfg.max_total)
        if kind_count.get(it.kind, 0) >= cap:
            dropped += 1
            continue
        kind_count[it.kind] = kind_count.get(it.kind, 0) + 1
        kept.append(it)

    if cfg.max_total > 0 and len(kept) > cfg.max_total:
        dropped += len(kept) - cfg.max_total
        kept = kept[: cfg.max_total]

    return kept, dropped


# ============================================================
# 5. prompt_section 渲染（仅追加证据，不改变安全指令）
# ============================================================

def _render_neo4j_section(evidence_items: list, max_chars: int) -> str:
    triples = [it for it in evidence_items if it.kind == KIND_NEO4J_TRIPLE]
    nodes = [it for it in evidence_items if it.kind == KIND_NEO4J_NODE]

    lines = [
        "【结构化医疗知识图谱证据（来自 Neo4j）】",
        "以下是从临床知识图谱检索到的结构化事实，仅作为回答参考依据，"
        "不构成医疗诊断或治疗建议：",
    ]
    for it in triples:
        p = it.payload
        lines.append(
            f"- {p.get('canonical_source', p.get('source', ''))} "
            f"--{p.get('relation_type', '')}--> "
            f"{p.get('canonical_target', p.get('target', ''))}"
        )
    for it in nodes:
        nm = it.payload.get("name") if isinstance(it.payload, dict) else ""
        if nm:
            lines.append(f"- 相关实体: {nm}")

    text = "\n".join(lines)
    if max_chars > 0 and len(text) > max_chars:
        text = text[: max_chars] + "\n...(证据段已截断)"
    return text


def build_fusion_prompt(
    neo4j_result: Optional[dict] = None,
    base_prompt: Optional[str] = None,
    config: Optional[FusionConfig] = None,
) -> Optional[str]:
    """生成可注入 system_prompt 的 Neo4j 证据文本段（不覆盖 base safety prompt）。

    - 仅追加证据，不把 Neo4j 数据包装成系统级指令
    - 明确标注"检索到的医疗知识证据"
    - 无 Neo4j 证据 / 容器不可用 → 返回 base_prompt 原样（SAFE 场景保持 LightRAG 默认）
    - base_prompt 为 None 且无证据 → None
    """
    cfg = config or FusionConfig()
    n4 = neo4j_result if isinstance(neo4j_result, dict) else {}
    if n4.get(NEO4J_UNAVAILABLE_KEY):
        return base_prompt

    triples = _normalize_triples(n4)
    nodes = [nd for nd in (n4.get("nodes") or []) if isinstance(nd, dict) and nd.get("name")]
    if not triples and not nodes:
        return base_prompt

    items = _build_prompt_items(triples, nodes)
    section = _render_neo4j_section(items, cfg.max_prompt_chars)

    if base_prompt:
        return f"{base_prompt}\n\n{section}"
    return section


def _build_prompt_items(triples: list, nodes: list) -> list:
    """把原始 triples/nodes 转成渲染用的临时 EvidenceItem 列表（不参与去重）。"""
    items: list = []
    for s, r, o in triples:
        items.append(EvidenceItem(
            id=f"triple::{s}::{r}::{o}",
            kind=KIND_NEO4J_TRIPLE,
            name=f"{s} -{r}-> {o}",
            payload={"source": s, "relation_type": r, "target": o,
                     "canonical_source": s, "canonical_target": o},
            sources=[SOURCE_NEO4J_GRAPH],
            priority=0,
        ))
    for nd in nodes:
        items.append(EvidenceItem(
            id=f"entity::{nd.get('name', '')}",
            kind=KIND_NEO4J_NODE,
            name=nd.get("name", ""),
            payload=nd,
            sources=[SOURCE_NEO4J_GRAPH],
            priority=1,
        ))
    return items


# ============================================================
# 6. fuse_evidence —— 主融合入口
# ============================================================

def fuse_evidence(
    lightrag_data: Optional[dict] = None,
    neo4j_data: Optional[dict] = None,
    entity_name_map: Optional[dict] = None,
    config: Optional[FusionConfig] = None,
) -> dict:
    """融合 LightRAG 与 Neo4j 证据，返回统一融合上下文。

    Args:
        lightrag_data:  {"entities", "relationships", "chunks"}（可空 / None）
        neo4j_data:     Neo4jGraphRetriever 输出或 {"unavailable": True, ...}
        entity_name_map: D4 load_entity_name_map() 输出（默认自动加载）
        config:         FusionConfig（默认值见类定义）

    Returns:
        见模块 docstring / 需求：status / evidence_sources / evidence /
        summary / coverage / prompt_section
    """
    cfg = config or FusionConfig()
    name_map = entity_name_map if isinstance(entity_name_map, dict) else _default_name_map()
    lr = lightrag_data if isinstance(lightrag_data, dict) else {}
    n4 = neo4j_data if isinstance(neo4j_data, dict) else {}

    # ---- 构建 Neo4j 侧 ----
    neo4j_items, triple_index, notes = _build_neo4j_items(n4, name_map)

    # 实体索引（供 LightRAG 跨源合并）
    entity_by_key: dict[str, int] = {}
    for idx, it in enumerate(neo4j_items):
        if it.kind in (KIND_NEO4J_NODE, KIND_NEO4J_TRIPLE):
            entity_by_key[it.id] = idx

    # ---- 构建 LightRAG 侧（含跨源合并）----
    lr_items, _rel_idx, merged = _build_lightrag_items(
        lr, name_map, neo4j_items, entity_by_key, triple_index,
    )

    all_items = neo4j_items + lr_items

    # ---- 排序（先）----
    sorted_items = sort_evidence(all_items)
    total_before_truncation = len(sorted_items)

    # ---- 截断（后）----
    kept_items, dropped = truncate_evidence(sorted_items, cfg)

    # ---- 汇总 ----
    by_kind: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for it in kept_items:
        by_kind[it.kind] = by_kind.get(it.kind, 0) + 1
        for src in it.sources:
            by_source[src] = by_source.get(src, 0) + 1

    lr_entities = [e for e in (lr.get("entities") or []) if isinstance(e, dict)]
    lr_rels = [r for r in (lr.get("relationships") or []) if isinstance(r, dict)]
    lr_chunks = [c for c in (lr.get("chunks") or []) if isinstance(c, dict)]

    n4_nodes_raw = [n for n in (n4.get("nodes") or []) if isinstance(n, dict)]
    n4_edges_raw = [e for e in (n4.get("edges") or []) if isinstance(e, dict)]
    n4_triples_raw = _normalize_triples(n4)
    neo4j_has_evidence = bool(
        not n4.get(NEO4J_UNAVAILABLE_KEY) and (n4_triples_raw or n4_nodes_raw)
    )
    if n4.get(NEO4J_UNAVAILABLE_KEY):
        neo4j_has_evidence = False

    # ---- prompt_section（基于最终保留的 Neo4j 证据，先截断后渲染）----
    neo4j_kept = [it for it in kept_items
                  if it.kind in (KIND_NEO4J_TRIPLE, KIND_NEO4J_NODE)]
    prompt_section = _render_neo4j_section(neo4j_kept, cfg.max_prompt_chars) if neo4j_kept else None

    return {
        "status": "ok",
        "evidence_sources": {
            "lightrag_vector": len(lr_chunks),
            "lightrag_graph": {
                "entities": len(lr_entities),
                "relationships": len(lr_rels),
            },
            "neo4j_graph": {
                "nodes": len(n4_nodes_raw),
                "edges": len(n4_edges_raw),
                "triples": len(n4_triples_raw),
            },
        },
        "evidence": [it.to_dict() for it in kept_items],
        "summary": {
            "total": total_before_truncation,
            "deduplicated": merged,
            "truncated": dropped,
            "by_kind": by_kind,
            "by_source": by_source,
        },
        "coverage": {
            "neo4j_has_evidence": neo4j_has_evidence,
            "notes": notes,
        },
        "prompt_section": prompt_section,
    }
