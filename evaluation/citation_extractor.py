"""Phase 5B B3 — CitationExtractor（结构化引用提取，纯函数，零 API 费用）。

职责：从 LLM 回答文本中提取引用（citation/reference），产出结构化的
`CitationResult`（含 `ReferenceItem` 列表）。

设计原则（对齐 docs/phase5b_b1_design_audit.md 的 CitationAnalyzer 契约）：
- 提取（extract）与验证（verify）分离：本模块只负责「结构化提取」，不做
  任何实体归一化 / 追溯 —— 那由 citation_verifier.py 负责。
- 纯函数，零项目依赖（仅 stdlib re/dataclasses），不调用 LLM。
- 支持当前系统实际出现的引用格式（优先级从高到低）：
    1. `### References` / `## 参考文献` 区块 + `- [N]` / `* [N]` 列表项
    2. `[来源: xxx]` 中文来源标记
    3. 正文内 `[N] xxx` 内联引用
    4. `.md` 文件名引用
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReferenceItem:
    """单条引用的结构化表示。"""

    index: int = -1                 # 引用序号 [N]；source_tag/inline 无序号时递增
    raw_text: str = ""              # 原始条目文本（保留全文，供 verify 取括号内英文别名）
    entity_name: str = ""           # 提取的实体名候选（未归一化，归一化交给 verify）
    entity_type: Optional[str] = None   # verify 阶段从检索实体回填
    source_file: Optional[str] = None   # verify 阶段从 .md / file_path 回填


@dataclass
class CitationResult:
    """引用分析结果（extract 产出存在性 + 结构化条目，verify 回填追溯字段）。"""

    citation_present: bool = False
    citation_format: Optional[str] = None  # "references_section" | "inline" | "source_tag" | None
    reference_items: list[ReferenceItem] = field(default_factory=list)

    # --- 以下字段由 citation_verifier.verify_citations() 回填 ---
    citation_grounded: bool = False
    grounded_entities: list[str] = field(default_factory=list)
    ungrounded_entities: list[str] = field(default_factory=list)
    grounded_ratio: float = 0.0
    total_references: int = 0
    grounding_method: str = "none"  # "source_id_chunk_mapping" > "entity_name_map" > "chunk_content_substring" > "none"


# ------------------------------------------------------------
# 正则常量（模块级编译一次）
# ------------------------------------------------------------

# References 区块标题（兼容中英文）
_REF_SECTION_HEADER_RE = re.compile(
    r"(?im)^#{1,4}\s*(?:references|参考(?:资料|文献)?|参考文献)\s*:?\s*$"
)

# References 区块内的列表项：- [1] xxx / * [1] xxx
_LIST_ITEM_RE = re.compile(r"(?m)^[-*]\s*\[(\d+)\]\s*(.+)$")

# References 区块内无 bullet 的 [1] xxx（兜底）
_BARE_ITEM_RE = re.compile(r"(?m)^\[(\d+)\]\s*(.+)$")

# 中文来源标记 [来源: xxx]
_SOURCE_TAG_RE = re.compile(r"\[来源\s*[：:]\s*([^\]]+)\]")

# 正文内联 [N] xxx（仅当无 References 区块时使用）
_INLINE_REF_RE = re.compile(r"\[(\d+)\]\s*([^\[\]\n]+)")

# .md 文件名引用
_FILE_REF_RE = re.compile(r"[\w\-]+\.md")


def _extract_entity_candidate(raw_text: str) -> str:
    """从原始条目文本中提取实体名候选（未归一化）。

    策略：
    1. 去除行首 [N] 标记；
    2. 按首个分隔符切分，取左侧（去掉「— 药品知识文档」等后缀）；
    3. 若左侧含括号，取括号前的中文头部（如 "二甲双胍 (Metformin)" → "二甲双胍"），
       raw_text 保留全文供 verifier 提取括号内英文别名。
    """
    text = raw_text.strip()
    # 去掉行首 [N]
    text = re.sub(r"^\[(\d+)\]\s*", "", text)

    # 按首个分隔符切分取左侧
    for sep in ("—", "–", " - ", "|", "：", ":"):
        idx = text.find(sep)
        if idx > 0:
            text = text[:idx].strip()
            break

    # 若含括号，取括号前头部
    paren = text.find("(")
    if paren > 0:
        text = text[:paren].strip()

    return text


def _extract_reference_items_from_section(section_body: str) -> list[ReferenceItem]:
    """解析 References 区块 body，提取 - [N] / * [N] / [N] 列表项。"""
    items: list[ReferenceItem] = []
    # 优先带 bullet 的列表项
    matches = _LIST_ITEM_RE.findall(section_body)
    if not matches:
        # 无 bullet 兜底
        matches = _BARE_ITEM_RE.findall(section_body)
    for num, item_text in matches:
        raw = item_text.strip()
        items.append(
            ReferenceItem(
                index=int(num),
                raw_text=raw,
                entity_name=_extract_entity_candidate(raw),
            )
        )
    return items


def extract_citations(*, answer: str) -> CitationResult:
    """从回答文本提取引用（CitationExtractor 主入口）。

    Args:
        answer: LLM 回答文本。

    Returns:
        CitationResult，含 citation_present / citation_format / reference_items。
        空/纯空白输入返回全默认对象，不抛异常。
    """
    if not answer or not answer.strip():
        return CitationResult()

    # 1. References 区块（### References / ## 参考文献 + 列表项）
    header_match = _REF_SECTION_HEADER_RE.search(answer)
    if header_match:
        # 截取标题到下一个 # 标题 / 空行 / 文末之间作为 body
        body_start = header_match.end()
        rest = answer[body_start:]
        # 下一个 # 标题
        next_heading = re.search(r"(?m)^#{1,4}\s", rest)
        # 空行（\n\s*\n）
        blank_line = re.search(r"\n\s*\n", rest)
        cut = len(rest)
        for m in (next_heading, blank_line):
            if m is not None:
                cut = min(cut, m.start())
        section_body = rest[:cut]

        items = _extract_reference_items_from_section(section_body)
        if items:
            return CitationResult(
                citation_present=True,
                citation_format="references_section",
                reference_items=items,
            )
        # 有标题但无列表项 → 仍视为存在引用区块（可能是空区块或非列表格式），继续尝试内联

    # 2. 中文来源标记 [来源: xxx]
    source_tags = _SOURCE_TAG_RE.findall(answer)
    if source_tags:
        items = []
        for i, tag in enumerate(source_tags):
            tag = tag.strip()
            source_file = None
            m = _FILE_REF_RE.search(tag)
            if m:
                source_file = m.group(0)
            items.append(
                ReferenceItem(
                    index=-1 if len(source_tags) > 1 else i,
                    raw_text=tag,
                    entity_name=_extract_entity_candidate(tag),
                    source_file=source_file,
                )
            )
        return CitationResult(
            citation_present=True,
            citation_format="source_tag",
            reference_items=items,
        )

    # 3. 正文内联 [N] xxx（仅当无 References 区块时）
    inline_refs = _INLINE_REF_RE.findall(answer)
    if inline_refs:
        items = []
        for num, ctx in inline_refs:
            ctx = ctx.strip()
            if len(ctx) <= 1:
                continue
            items.append(
                ReferenceItem(
                    index=int(num),
                    raw_text=ctx,
                    entity_name=_extract_entity_candidate(ctx),
                )
            )
        if items:
            return CitationResult(
                citation_present=True,
                citation_format="inline",
                reference_items=items,
            )

    # 4. .md 文件名引用（兜底）
    file_refs = _FILE_REF_RE.findall(answer)
    if file_refs:
        items = []
        for i, fname in enumerate(file_refs):
            items.append(
                ReferenceItem(
                    index=i,
                    raw_text=fname,
                    entity_name="",
                    source_file=fname,
                )
            )
        return CitationResult(
            citation_present=True,
            citation_format="inline",
            reference_items=items,
        )

    # 无任何引用
    return CitationResult()
