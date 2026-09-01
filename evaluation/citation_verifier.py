"""Phase 5B B3 — CitationVerifier（引用真实性验证，D4 归一化 + D3 追溯，纯函数）。

职责：对 `CitationExtractor` 产出的 `CitationResult` 做真实性验证，判断每条引用
是否能追溯到检索到的实体 / 文本块。

设计原则（对齐 docs/phase5b_b1_design_audit.md 的 CitationAnalyzer 契约）：
- 复用 evaluation.metrics 的 `resolve_entity_name`（D4 别名归一化）与
  `map_entity_to_chunks`（D3 source_id↔chunk_id 映射），不重复实现。
- 三级追溯（优先级从高到低）：
    Tier 1 — 实体名归一化匹配（D4）: 引用实体名 → resolve → canonical → 命中检索实体
    Tier 2 — source_id→chunk_id 结构化映射（D3）: Tier 1 命中的实体，其 source_id
             指向某个检索 chunk（作为最强信号，仅作增强确认）
    Tier 3 — chunk 内容子串兜底: 候选串 / canonical 形式命中 chunk content
- 诚实记录：无法追溯的引用明确记入 ungrounded_entities，绝不伪造映射。
"""

from evaluation.citation_extractor import CitationResult, ReferenceItem, extract_citations
from evaluation.metrics import map_entity_to_chunks, resolve_entity_name


def _candidates_from_item(item: ReferenceItem) -> list[str]:
    """为单条引用生成候选匹配串（中文头 / 英文别名 / 全文）。"""
    candidates: list[str] = []
    raw = item.raw_text.strip()
    if item.entity_name:
        candidates.append(item.entity_name)
    # raw_text 中 ( 前的中文头
    paren = raw.find("(")
    if paren > 0:
        head = raw[:paren].strip()
        if head:
            candidates.append(head)
    # ( ) 内的英文别名
    if paren >= 0:
        close = raw.find(")", paren)
        if close > paren + 1:
            alias = raw[paren + 1 : close].strip()
            if alias:
                candidates.append(alias)
    # 全文兜底
    if raw and raw not in candidates:
        candidates.append(raw)
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


def _display_name(item: ReferenceItem, canonical: str) -> str:
    """选择可解释的展示名：canonical 优先，否则 entity_name，否则 raw_text 截断。"""
    if canonical:
        return canonical
    if item.entity_name:
        return item.entity_name
    return item.raw_text[:80]


def verify_citations(
    *,
    citation_result: CitationResult,
    retrieved_entities: list[dict],
    retrieved_chunks: list[dict],
    entity_name_map: dict[str, str],
) -> CitationResult:
    """对 CitationResult 做真实性验证（CitationVerifier 主入口）。

    就地回填 citation_grounded / grounded_entities / ungrounded_entities /
    grounded_ratio / total_references / grounding_method，并返回同一对象。

    Args:
        citation_result: extract_citations() 的产出。
        retrieved_entities: aquery_data().entities[] 列表。
        retrieved_chunks: aquery_data().chunks[] 列表。
        entity_name_map: load_entity_name_map() 的 D4 映射表。

    Returns:
        回填追溯字段后的同一 CitationResult 对象。
    """
    citation_result.total_references = len(citation_result.reference_items)
    if citation_result.total_references == 0:
        citation_result.citation_grounded = False
        citation_result.grounded_ratio = 0.0
        citation_result.grounding_method = "none"
        return citation_result

    # 预构建检索索引（一次）
    retrieved_by_canonical: dict[str, dict] = {}
    for ent in retrieved_entities:
        name = ent.get("entity_name", "")
        if not name:
            continue
        canon = resolve_entity_name(name, entity_name_map)
        retrieved_by_canonical.setdefault(canon.lower(), ent)

    chunk_contents = [c.get("content", "") for c in retrieved_chunks]
    chunk_files = [c.get("file_path", "") for c in retrieved_chunks]

    grounded_entities: list[str] = []
    ungrounded_entities: list[str] = []
    method_priority = {"source_id_chunk_mapping": 3, "entity_name_map": 2, "chunk_content_substring": 1}
    best_method = "none"

    for item in citation_result.reference_items:
        candidates = _candidates_from_item(item)
        matched_entity: dict | None = None
        matched_canonical = ""
        grounded = False
        method = "none"

        # Tier 1 — 实体名归一化匹配（D4）
        for cand in candidates:
            resolved = resolve_entity_name(cand, entity_name_map)
            ent = retrieved_by_canonical.get(resolved.lower())
            if ent is not None:
                matched_entity = ent
                matched_canonical = resolved
                grounded = True
                method = "entity_name_map"
                break

        # Tier 2 — source_id→chunk_id 结构化映射（D3，作为 Tier 1 命中的增强确认）
        if matched_entity is not None:
            linked_chunks = map_entity_to_chunks(matched_entity, retrieved_chunks)
            if linked_chunks:
                method = "source_id_chunk_mapping"
            # 回填 entity_type / source_file
            item.entity_type = matched_entity.get("entity_type")
            if item.source_file is None:
                item.source_file = matched_entity.get("file_path")

        # Tier 3 — chunk 内容子串兜底（Tier 1 未命中时）
        if not grounded:
            for cand in candidates:
                cand_lower = cand.lower()
                for content in chunk_contents:
                    if cand_lower and cand_lower in content.lower():
                        grounded = True
                        method = "chunk_content_substring"
                        break
                if grounded:
                    break
            # .md 文件名对 chunk file_path 兜底
            if not grounded and item.source_file:
                sf = item.source_file.lower()
                for fp in chunk_files:
                    if sf and sf in fp.lower():
                        grounded = True
                        method = "chunk_content_substring"
                        break

        display = _display_name(item, matched_canonical if matched_entity else "")
        if grounded:
            grounded_entities.append(display)
        else:
            ungrounded_entities.append(display)

        if method_priority.get(method, 0) > method_priority.get(best_method, 0):
            best_method = method

    total = citation_result.total_references
    citation_result.citation_grounded = len(grounded_entities) > 0
    citation_result.grounded_entities = grounded_entities
    citation_result.ungrounded_entities = ungrounded_entities
    citation_result.grounded_ratio = round(len(grounded_entities) / total, 4) if total else 0.0
    citation_result.grounding_method = best_method

    return citation_result


def analyze_citations(
    *,
    answer: str,
    retrieved_entities: list[dict],
    retrieved_chunks: list[dict],
    entity_name_map: dict[str, str],
) -> CitationResult:
    """一步完成引用提取 + 验证（便捷入口）。

    Args:
        answer: LLM 回答文本。
        retrieved_entities: aquery_data().entities[] 列表。
        retrieved_chunks: aquery_data().chunks[] 列表。
        entity_name_map: load_entity_name_map() 的 D4 映射表。

    Returns:
        完整回填的 CitationResult。
    """
    result = extract_citations(answer=answer)
    return verify_citations(
        citation_result=result,
        retrieved_entities=retrieved_entities,
        retrieved_chunks=retrieved_chunks,
        entity_name_map=entity_name_map,
    )
