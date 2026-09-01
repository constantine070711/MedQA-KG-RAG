"""Phase 6 F3 — curated 临床关系重建（.md 结构化字段 → 真实医疗关系）。

职责:
    - 从 curated 医疗 .md 的可信结构化章节（常见症状/并发症/治疗方法/适应症/禁忌症/常见不良反应）
      解析候选临床关系
    - 关系两端经现有 D4（load_entity_name_map / resolve_entity_name）解析到 curated 实体
    - 只写入两端均为 curated 实体的关系；占位符（"见正文"等）与无法解析目标严格跳过并记录
    - 参数化 UNWIND 批量写入 + MERGE 幂等
    - 不推断、不猜测、不为凑数量补关系；不修改原始 .md / entities.json / D4

边界（F3）:
    - 不实现 Neo4jGraphRetriever / Evidence Fusion / hybrid retrieval
    - curated 层与 LightRAG 原始层架构隔离，不做跨层链接
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from evaluation.metrics import load_entity_name_map, resolve_entity_name
from src.medical_core.relations import RelationType

from src.graph_import import _run_batched

logger = logging.getLogger(__name__)

# ---- 章节 → 关系语义映射 ----
# (md 实体类型, 章节标题) -> {rel_type, match_type, md_side}
#   md_side: "subject" = md 实体是关系主体；"object" = md 实体是关系客体
SECTION_MAP = {
    ("disease", "常见症状"): {"rel_type": "HAS_SYMPTOM", "match_type": "symptom", "md_side": "subject"},
    ("disease", "并发症"): {"rel_type": "HAS_COMPLICATION", "match_type": "disease", "md_side": "subject"},
    ("disease", "治疗方法"): {"rel_type": "TREATS", "match_type": "drug", "md_side": "object"},
    ("drug", "适应症"): {"rel_type": "TREATS", "match_type": "disease", "md_side": "subject"},
    ("drug", "禁忌症"): {"rel_type": "CONTRAINDICATED_FOR", "match_type": "disease", "md_side": "subject"},
    ("drug", "常见不良反应"): {"rel_type": "HAS_ADVERSE_REACTION", "match_type": "symptom", "md_side": "subject"},
}

# 关系类型 → Neo4j 节点标签（受控常量，非原始数据）
REL_LABELS = {
    "TREATS": ("Drug", "Disease"),
    "HAS_COMPLICATION": ("Disease", "Disease"),
    "HAS_SYMPTOM": ("Disease", "Symptom"),
    "CONTRAINDICATED_FOR": ("Drug", "Disease"),
    "HAS_ADVERSE_REACTION": ("Drug", "Symptom"),
}

# 允许的关系类型（来自 medical_core RelationType 注册表，受控）
VALID_REL_TYPES = {t.value for t in RelationType}

# 占位符特征（绝不能生成真实关系）
PLACEHOLDER_MARKERS = ("见正文", "详见正文", "见相关", "未知", "暂无", "占位")

# curated 实体类型（仅这些参与两端解析）
CURATED_TYPES = ("disease", "drug")


def _norm(s: str) -> str:
    """去除所有空白用于匹配。"""
    return re.sub(r"\s+", "", s or "")


def is_placeholder_text(text: str) -> bool:
    """判断章节文本是否含占位符内容。"""
    return any(m in (text or "") for m in PLACEHOLDER_MARKERS)


def _load_curated_entities(entities_path) -> list[dict]:
    """从 entities.json 加载 curated 实体候选（含别名 + 归一化名）。"""
    data = json.loads(Path(entities_path).read_text(encoding="utf-8"))
    entities = []
    for e in data.get("entities", []):
        names = [e.get("name", "")] + [
            a.get("name", "") for a in e.get("aliases", []) or [] if a.get("name")
        ]
        entities.append({
            "name": e.get("name", ""),
            "entity_type": e.get("entity_type", ""),
            "names": [n for n in names if n],
            "norm_names": [_norm(n) for n in names if n],
        })
    return entities


def _extract_sections(md_text: str) -> dict[str, str]:
    """按 ## 章节标题切分 .md 正文，返回 {标题: 内容}。"""
    sections: dict[str, str] = {}
    parts = re.split(r"^##\s+", md_text, flags=re.M)
    for part in parts[1:]:
        header = part.split("\n", 1)[0].strip()
        body = part.split("\n", 1)[1] if "\n" in part else ""
        sections[header] = body.strip()
    return sections


def _frontmatter_field(md_text: str, field: str) -> str:
    m = re.search(rf"^{field}:\s*\"([^\"]+)\"", md_text, re.M)
    return m.group(1) if m else ""


def parse_curated_relations(raw_dir, entities_path) -> dict:
    """解析 curated .md 结构化章节 → 候选真实关系（两端 D4 解析到 curated 实体）。

    Returns:
        {
            "relations": [{rel_type, subject, object, source, confidence}],
            "unresolved": [{source, section, rel_type, match_type, raw}],
            "placeholder_skipped": [{source, section, reason}],
            "stats": {source_md_files, candidate_relations, resolved_relations,
                      unresolved_mentions, placeholder_skipped, self_skipped,
                      by_rel_type, d4_resolve_success, d4_resolve_fail},
        }
    """
    entities = _load_curated_entities(entities_path)
    name_map = load_entity_name_map(str(entities_path))

    # canonical 名 → entity 信息（用于 D4 解析验证与两端标签）
    canonical_to_entity = {e["name"]: e for e in entities}
    # 归一化名 → canonical（用于匹配）
    norm_to_canonical: dict[str, str] = {}
    for e in entities:
        for nn in e["norm_names"]:
            norm_to_canonical.setdefault(nn, e["name"])

    relations: list[dict] = []
    unresolved: list[dict] = []
    placeholder_skipped: list[dict] = []
    self_skipped: list[dict] = []
    d4_success = 0
    d4_fail = 0

    md_files = sorted(Path(raw_dir).glob("**/*.md"))
    for md in md_files:
        text = md.read_text(encoding="utf-8")
        entity_type = _frontmatter_field(text, "entity_type")
        entity_name = _frontmatter_field(text, "name")
        if entity_type not in CURATED_TYPES or not entity_name:
            continue
        if entity_name not in canonical_to_entity:
            continue  # .md 实体不在 curated 集合，跳过

        sections = _extract_sections(text)
        for header, body in sections.items():
            spec = SECTION_MAP.get((entity_type, header))
            if spec is None:
                continue
            rel_type = spec["rel_type"]
            match_type = spec["match_type"]
            md_side = spec["md_side"]

            # 占位符隔离
            if is_placeholder_text(body):
                placeholder_skipped.append({
                    "source": f"{md.name}#{header}", "section": header,
                    "reason": "placeholder content",
                })
                continue

            body_norm = _norm(body)

            # 匹配 curated 候选（match_type 过滤，归一化子串）
            matched = set()
            for e in entities:
                if e["entity_type"] != match_type:
                    continue
                for nn in e["norm_names"]:
                    if nn and nn in body_norm:
                        matched.add(e["name"])
                        break

            for target in sorted(matched):
                # D4 解析目标到 canonical
                resolved = resolve_entity_name(target, name_map)
                if resolved and resolved in canonical_to_entity:
                    d4_success += 1
                else:
                    d4_fail += 1
                    unresolved.append({
                        "source": f"{md.name}#{header}", "section": header,
                        "rel_type": rel_type, "match_type": match_type,
                        "raw": target, "reason": "D4 resolve fail",
                    })
                    continue

                # 组装关系（md 实体为 subject 或 object）
                if md_side == "subject":
                    subj, obj = entity_name, resolved
                else:
                    subj, obj = resolved, entity_name

                # 自环剔除（如 "哮喘" 别名在 "哮喘持续状态" 中的假阳性）
                if subj == obj:
                    self_skipped.append({
                        "source": f"{md.name}#{header}", "section": header,
                        "rel_type": rel_type, "subject": subj, "object": obj,
                        "reason": "self-relation",
                    })
                    continue

                # 关系类型合法性校验
                if rel_type not in VALID_REL_TYPES:
                    continue

                relations.append({
                    "rel_type": rel_type,
                    "subject": subj,
                    "object": obj,
                    "source": f"{md.name}#{header}",
                    "confidence": 1.0,
                })

            # 未解析条目枚举：章节中的 bullet 项，未匹配任何 curated 实体
            for line in body.splitlines():
                line = line.strip()
                if not line.startswith("-"):
                    continue
                item = line.lstrip("- ").strip()
                item_norm = _norm(item)
                if not item_norm:
                    continue
                # 去掉行内 markdown 强调/括号注释，做粗匹配
                if any(nn in item_norm for e in entities
                       for nn in e["norm_names"] if nn):
                    continue  # 已命中 curated 实体
                unresolved.append({
                    "source": f"{md.name}#{header}", "section": header,
                    "rel_type": rel_type, "match_type": match_type,
                    "raw": item[:60], "reason": "no curated entity match",
                })

    # 去重关系（同一 (rel, subject, object) 仅保留一次，来源合并记首个）
    seen = set()
    unique_relations = []
    for rel in relations:
        key = (rel["rel_type"], rel["subject"], rel["object"])
        if key in seen:
            continue
        seen.add(key)
        unique_relations.append(rel)

    by_rel_type: dict[str, int] = {}
    for rel in unique_relations:
        by_rel_type[rel["rel_type"]] = by_rel_type.get(rel["rel_type"], 0) + 1

    return {
        "relations": unique_relations,
        "unresolved": unresolved,
        "placeholder_skipped": placeholder_skipped,
        "self_skipped": self_skipped,
        "stats": {
            "source_md_files": len(md_files),
            "candidate_relations": len(relations),
            "resolved_relations": len(unique_relations),
            "unresolved_mentions": len(unresolved),
            "placeholder_skipped": len(placeholder_skipped),
            "self_skipped": len(self_skipped),
            "by_rel_type": by_rel_type,
            "d4_resolve_success": d4_success,
            "d4_resolve_fail": d4_fail,
        },
    }


# ============================================================
# Neo4j 写入（参数化 UNWIND + MERGE 幂等）
# ============================================================

def import_relations(conn, relations: list[dict], batch_size: int = 200) -> dict:
    """按关系类型分组，参数化 UNWIND MERGE 写入。

    关系类型与节点标签来自受控常量 REL_LABELS / VALID_REL_TYPES（非原始数据拼接）。
    """
    grouped: dict[str, list[dict]] = {}
    for rel in relations:
        grouped.setdefault(rel["rel_type"], []).append(rel)

    results: dict[str, dict] = {}
    for rel_type, items in grouped.items():
        subj_label, obj_label = REL_LABELS.get(rel_type, ("Entity", "Entity"))
        cypher = f"""
        UNWIND $batch AS item
        MATCH (a:{subj_label} {{name: item.subject}})
        MATCH (b:{obj_label} {{name: item.object}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.source = item.source, r.confidence = item.confidence
        """
        results[rel_type] = _run_batched(conn, cypher, items, batch_size, f"rel_{rel_type}")
    return results


def count_relations_by_type(conn, rel_type: Optional[str] = None) -> int:
    if rel_type:
        rows = conn.execute(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c")
    else:
        rows = conn.execute("MATCH ()-[r]->() RETURN count(r) AS c")
    return int(rows[0]["c"]) if rows else 0
