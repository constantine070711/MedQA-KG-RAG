# -*- coding: utf-8 -*-
"""Phase 10-A.5 — 真实医学知识库（CMeIE）独立数据适配器（实验旁路）。

输入：CMeIE（CHIP2020，CBLUE 同源）原始 .jsonl
      {"text": "...", "spo_list": [{"subject","subject_type","predicate",
                                    "object":{"@value": "..."},"object_type"}]}
输出：统一医学知识格式（MedKBEntity / MedKBRelation，均带 provenance）。

与 cblue_adapter 的关系：
  - CMeIE 是 CBLUE 基准任务之一（CBLUE CMeIE-V2 由此发展而来）。
  - 复用 cblue_adapter 的 CMEIE_RELATION_MAP（关系映射）与 Entity Linking（L0-L3）。
  - 本模块为**独立旁路**：不改生产核心、不改 cblue_adapter、不写 Neo4j / 生产 LightRAG。

与 CBLUE 格式的差异（本 adapter 处理）：
  1. object / object_type 为字典 {"@value": "..."}（非字符串）→ 解包。
  2. 记录级无 sample_id → 由 ingest 注入记录序号。
  3. subject_type / object_type 提供实体类型（CMeIE 10/11 类）。

关系映射保守（高置信子集，禁止强行映射）：
  - 复用 cblue 高置信映射：并发症→HAS_COMPLICATION / 筛查→SCREENS_FOR / 鉴别诊断→DIFFERENTIAL_DIAGNOSIS。
  - 补充明确同义词：高危因素→HAS_RISK_FACTOR（=危险因素）。
  - 其余 40 种 CMeIE predicate 如实记 relation_type="UNMAPPED"（保留原词，不伪造）。

许可证：CMeIE — CC-BY-NC 4.0（OpenI 官方页面声明）。所有派生产物强制携带 provenance。
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .cblue_adapter import (
    _norm,
    KNOWN_ENTITY_TYPES,
    CMEIE_RELATION_MAP,
    link_entities,
    load_production_codes,
    load_production_name_map,
)

# ============================================================
# 常量
# ============================================================

MEDKB_LICENSE = "CC-BY-NC 4.0"
ADAPTER_NAME = "medical_kb"
INGEST_DATE = "2026-08-31"
DATASET_NAME = "CMeIE"
DATASET_SPLIT = "train"

# CMeIE subject/object_type → 本项目 EntityType（映射清晰者；未知如实保留 "unknown"）
# 注：手术治疗/其他治疗/社会学/流行病学/预后/其他 在本项目 EntityType 无对应，
#     如实映射为 "unknown" 并记录 warning（不强行归类）。
CMEIE_TYPE_MAP = {
    "疾病": "disease",
    "症状": "symptom",
    "药物": "drug",
    "检查": "examination",
    "部位": "anatomy",
}

# CMeIE predicate → 本项目 RelationType（高置信子集）。
# 复用 cblue 映射 + 明确同义词（高危因素=危险因素）。
# 未收录的 predicate 一律 relation_type="UNMAPPED"，如实保留原词（不伪造映射）。
MEDKB_RELATION_MAP = dict(CMEIE_RELATION_MAP)
MEDKB_RELATION_MAP["高危因素"] = "HAS_RISK_FACTOR"


# ============================================================
# 统一数据模型（MedKB 专用，含 provenance / quality）
# ============================================================

@dataclass
class MedKBEntity:
    """统一医学实体（CMeIE 派生）。"""

    entity_id: str
    name: str
    entity_type: str
    aliases: list = field(default_factory=list)          # [{"name","language"}]
    standard_codes: list = field(default_factory=list)    # [{"system","code"}]
    raw_text: str = ""
    provenance: dict = field(default_factory=dict)        # {dataset,split,sample_id,source_file,adapter,license,ingest_date}
    quality: dict = field(default_factory=lambda: {"level": "pass", "errors": [], "warnings": []})

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "aliases": self.aliases,
            "standard_codes": self.standard_codes,
            "raw_text": self.raw_text,
            "provenance": self.provenance,
            "quality": self.quality,
        }


@dataclass
class MedKBRelation:
    """统一医学关系三元组（CMeIE 派生）。"""

    subject: str
    relation: str            # RelationType.value（已映射）或原 predicate（未映射）
    object: str
    relation_type: str       # RelationType.value 或 "UNMAPPED"
    subject_type: str = ""   # 原始类型（保留审计）
    object_type: str = ""
    provenance: dict = field(default_factory=dict)
    confidence: float = 1.0
    quality: dict = field(default_factory=lambda: {"level": "pass", "errors": [], "warnings": []})

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "relation_type": self.relation_type,
            "subject_type": self.subject_type,
            "object_type": self.object_type,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "quality": self.quality,
        }

    def to_triple(self) -> dict:
        """简化三元组（triples.json 用）。"""
        return {
            "subject": self.subject,
            "predicate": self.relation,
            "object": self.object,
            "relation_type": self.relation_type,
            "provenance": self.provenance,
        }


# ============================================================
# 工具函数
# ============================================================

def _make_provenance(split: str, record_idx: int, source_file: str) -> dict:
    return {
        "dataset": DATASET_NAME,
        "split": split,
        "sample_id": str(record_idx),
        "source_file": str(source_file),
        "adapter": ADAPTER_NAME,
        "license": MEDKB_LICENSE,
        "ingest_date": INGEST_DATE,
    }


def _mk_entity_id(split: str, record_idx: int, ent_idx: int) -> str:
    return f"medkb:cmeie:{split}:{record_idx}:{ent_idx}"


def _unwrap_value(v) -> str:
    """解包 CMeIE 的 {"@value": "..."} 字典为字符串；已是字符串则原样返回。"""
    if isinstance(v, dict):
        return str(v.get("@value", "") or "").strip()
    return str(v or "").strip()


def _map_entity_type(raw_type: str) -> tuple[str, list[str]]:
    """CMeIE 类型 → 项目 EntityType；未知如实返回 "unknown" + warning。"""
    raw_type = str(raw_type or "").strip()
    etype = CMEIE_TYPE_MAP.get(raw_type, "unknown")
    warnings = []
    if etype == "unknown":
        warnings.append(f"unknown cmeie type: {raw_type!r}")
    if etype not in KNOWN_ENTITY_TYPES:
        warnings.append(f"type not in project EntityType: {etype}")
    return etype, warnings


# ============================================================
# 解析：CMeIE 单条记录 → MedKBEntity + MedKBRelation
# ============================================================

def parse_cmeie_record(record: dict, record_idx: int, split: str = "train",
                       source_file: str = "") -> dict:
    """解析单条 CMeIE 记录 → 实体候选 + 关系。

    Returns:
        {"entities": [MedKBEntity], "relations": [MedKBRelation],
         "skipped": [dict], "stats": {...}}
    """
    if not isinstance(record, dict):
        return {"entities": [], "relations": [], "skipped": [{"record": record_idx, "reason": "record 非 dict"}], "stats": {}}

    text = str(record.get("text", "") or "").strip()
    spo = record.get("spo_list", [])
    if not isinstance(spo, list):
        spo = []

    if not text:
        return {"entities": [], "relations": [], "skipped": [{"record": record_idx, "reason": "empty text"}], "stats": {}}

    relations: list[MedKBRelation] = []
    skipped: list[dict] = []
    provenance = _make_provenance(split, record_idx, source_file)

    # 实体名候选（subject/object 去重，按 (name,type) 保留首次）
    entity_cache: dict[tuple[str, str], MedKBEntity] = {}

    def _candidate_entity(name: str, raw_type: str) -> Optional[MedKBEntity]:
        name = _unwrap_value(name)
        if not name:
            return None
        etype, twarn = _map_entity_type(raw_type)
        key = (_norm(name), etype)
        if key in entity_cache:
            return entity_cache[key]
        e = MedKBEntity(
            entity_id=_mk_entity_id(split, record_idx, len(entity_cache)),
            name=name,
            entity_type=etype,
            raw_text=text[:500],
            provenance=provenance,
            quality={"level": "warn" if twarn else "pass", "errors": [], "warnings": twarn},
        )
        entity_cache[key] = e
        return e

    for i, s in enumerate(spo):
        if not isinstance(s, dict):
            skipped.append({"record": record_idx, "index": i, "reason": "spo 非 dict"})
            continue
        subj = _unwrap_value(s.get("subject"))
        obj = _unwrap_value(s.get("object"))
        pred = str(s.get("predicate", "") or "").strip()
        subj_raw_type = str(s.get("subject_type", "") or "").strip()
        obj_raw_type = _unwrap_value(s.get("object_type"))

        if not subj or not obj or not pred:
            skipped.append({"record": record_idx, "index": i, "reason": "spo 缺字段"})
            continue

        rel_type = MEDKB_RELATION_MAP.get(pred, "UNMAPPED")
        warnings = []
        if rel_type == "UNMAPPED":
            warnings.append(f"unmapped predicate: {pred!r}")
        confidence = 1.0 if rel_type != "UNMAPPED" else 0.5
        quality = {"level": "pass" if not warnings else "warn",
                   "errors": [], "warnings": warnings}

        relations.append(MedKBRelation(
            subject=subj,
            relation=rel_type if rel_type != "UNMAPPED" else pred,
            object=obj,
            relation_type=rel_type,
            subject_type=subj_raw_type,
            object_type=obj_raw_type,
            provenance=provenance,
            confidence=confidence,
            quality=quality,
        ))
        _candidate_entity(subj, subj_raw_type)
        _candidate_entity(obj, obj_raw_type)

    stats = {
        "records_processed": 1,
        "relations_extracted": len(relations),
        "entities_extracted": len(entity_cache),
        "skipped": len(skipped),
    }
    return {"entities": list(entity_cache.values()), "relations": relations,
            "skipped": skipped, "stats": stats}


# ============================================================
# 文件读取（.json / .jsonl，兼容 cblue _read_records 语义）
# ============================================================

def _read_records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        out = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, list):
                    out.extend(obj)
                elif isinstance(obj, dict):
                    out.append(obj)
            except json.JSONDecodeError:
                out.append({"_parse_error": True, "_line": ln[:100]})
        return out
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("records"), list):
            return data["records"]
        if "text" in data or "spo_list" in data:
            return [data]
        return []
    return []


# ============================================================
# ingest：读取 + 解析 + 注入 record_idx + 去重
# ============================================================

def ingest(input_path, split: str = DATASET_SPLIT, limit: Optional[int] = None) -> dict:
    """读取 CMeIE 文件/目录并解析为实体与关系候选。

    Args:
        input_path: 文件或目录（含 CMeIE 数据）。
        split: 数据划分（默认 train）。
        limit: 最多处理记录数（None=全部）。

    Returns:
        {"split", "records_loaded", "entities", "relations", "skipped", "stats"}
    """
    path = Path(input_path)
    if path.is_dir():
        # 优先 train（防泄漏）；也可指定文件
        files = sorted(path.glob("*train*.jsonl")) or sorted(path.glob("*.jsonl"))
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    entities: list[MedKBEntity] = []
    relations: list[MedKBRelation] = []
    skipped: list[dict] = []
    records_loaded = 0
    empty_text = 0
    malformed = 0

    for f in files:
        try:
            records = _read_records(f)
        except Exception as exc:
            skipped.append({"file": str(f), "reason": f"read error: {exc}"})
            continue

        for record_idx, r in enumerate(records):
            if limit is not None and records_loaded >= limit:
                break
            if "_parse_error" in r:
                malformed += 1
                skipped.append({"file": str(f), "reason": "json parse error"})
                continue
            records_loaded += 1
            res = parse_cmeie_record(r, record_idx, split=split, source_file=str(f))
            entities.extend(res["entities"])
            relations.extend(res["relations"])
            skipped.extend(res["skipped"])
            if not str(r.get("text", "") or "").strip():
                empty_text += 1

    # 去重（实体按 (norm_name, type)；关系按 (subject, predicate, object)）
    ent_dedup: dict[tuple[str, str], MedKBEntity] = {}
    for e in entities:
        ent_dedup.setdefault((_norm(e.name), e.entity_type), e)
    rel_dedup: dict[tuple[str, str, str], MedKBRelation] = {}
    for r_ in relations:
        key = (r_.subject, r_.relation_type, r_.object)
        rel_dedup.setdefault(key, r_)

    stats = {
        "split": split,
        "records_loaded": records_loaded,
        "files": [str(f) for f in files],
        "empty_text_records": empty_text,
        "malformed_records": malformed,
        "entities_raw": len(entities),
        "entities_dedup": len(ent_dedup),
        "relations_raw": len(relations),
        "relations_dedup": len(rel_dedup),
        "triples_dedup": len(rel_dedup),
        "skipped": len(skipped),
    }
    return {"split": split, "records_loaded": records_loaded,
            "entities": list(ent_dedup.values()), "relations": list(rel_dedup.values()),
            "skipped": skipped, "stats": stats}


# ============================================================
# 校验
# ============================================================

def validate_entity(e: MedKBEntity) -> dict:
    errors, warnings = [], []
    if not e.name:
        errors.append("missing name")
    if not e.entity_type:
        errors.append("missing entity_type")
    if not e.provenance:
        errors.append("missing provenance")
    if not e.entity_id:
        errors.append("missing entity_id")
    if not e.raw_text:
        warnings.append("empty raw_text")
    warnings.extend(e.quality.get("warnings", []))
    errors.extend(e.quality.get("errors", []))
    level = "fail" if errors else ("warn" if warnings else "pass")
    return {"level": level, "errors": errors, "warnings": warnings}


def validate_relation(r: MedKBRelation) -> dict:
    errors, warnings = [], []
    if not r.subject:
        errors.append("missing subject")
    if not r.object:
        errors.append("missing object")
    if not r.relation:
        errors.append("missing relation")
    if not r.provenance:
        errors.append("missing provenance")
    if r.relation_type not in (set(MEDKB_RELATION_MAP.values()) | {"UNMAPPED"}):
        errors.append(f"unknown relation_type: {r.relation_type!r}")
    warnings.extend(r.quality.get("warnings", []))
    errors.extend(r.quality.get("errors", []))
    level = "fail" if errors else ("warn" if warnings else "pass")
    return {"level": level, "errors": errors, "warnings": warnings}


def convert(ingest_result: dict) -> dict:
    """统一化 ingest 结果 → {entities, relations}（dict 形式，用于写 JSON）。"""
    entities = [e.to_dict() for e in ingest_result.get("entities", [])]
    relations = [r.to_dict() for r in ingest_result.get("relations", [])]
    for e in entities:
        q = validate_entity(MedKBEntity(**{k: e[k] for k in
            ("entity_id", "name", "entity_type", "aliases", "standard_codes",
             "raw_text", "provenance", "quality") if k in e}))
        e["quality"] = q
    for r in relations:
        q = validate_relation(MedKBRelation(**{k: r[k] for k in
            ("subject", "relation", "object", "relation_type", "subject_type",
             "object_type", "confidence", "provenance", "quality") if k in r}))
        r["quality"] = q
    return {"entities": entities, "relations": relations}


# ============================================================
# Entity Linking L4（相似度候选，补充 cblue 的 L0-L3）
# ============================================================

def link_entities_l4(unmatched: list[dict], prod_name_map: Optional[dict] = None,
                     threshold: float = 0.8) -> list[dict]:
    """对 L0-L3 未匹配实体做字符相似度候选（L4，弱信号 → 仅 candidate，不 merge）。

    相似度：difflib.SequenceMatcher（字符级）。
    命中且 score >= threshold → 追加为 candidate（confidence=score，不标记 matched）。
    unmatched 中未被 L4 捕获的保持原样。
    """
    if prod_name_map is None:
        prod_name_map = load_production_name_map()
    try:
        from difflib import SequenceMatcher
    except ImportError:  # pragma: no cover
        return unmatched

    out: list[dict] = []
    for link in unmatched:
        name = str(link.get("name", "") or "")
        best = None
        best_score = 0.0
        for pk, canonical in prod_name_map.items():
            score = SequenceMatcher(None, name, pk).ratio()
            if score > best_score:
                best_score, best = score, canonical
        if best and best_score >= threshold:
            cand = dict(link)
            cand.update({"matched": False, "canonical": best,
                         "match_level": "L4", "confidence": round(best_score, 4)})
            out.append(cand)
        else:
            out.append(link)
    return out


# ============================================================
# 汇总统计
# ============================================================

def summarize(convert_result: dict, ingest_stats: dict) -> dict:
    """生成可读统计：unresolved / duplicate / missing provenance / predicate 分布等。"""
    entities = convert_result.get("entities", [])
    relations = convert_result.get("relations", [])

    missing_prov = sum(1 for e in entities if not e.get("provenance"))
    missing_prov += sum(1 for r in relations if not r.get("provenance"))

    unmapped_rels = sum(1 for r in relations if r.get("relation_type") == "UNMAPPED")
    unknown_type_ents = sum(1 for e in entities
                            if e.get("entity_type") in ("unknown", "anatomy", "biomarker"))
    fail_ents = sum(1 for e in entities if e.get("quality", {}).get("level") == "fail")
    fail_rels = sum(1 for r in relations if r.get("quality", {}).get("level") == "fail")

    # 关系种类分布（原始 predicate vs 映射后 RelationType）
    pred_counter = Counter(r.get("relation") for r in relations)
    rel_type_counter = Counter(r.get("relation_type") for r in relations)

    return {
        "entities": len(entities),
        "relations": len(relations),
        "triples": len(relations),
        "unmapped_relations": unmapped_rels,
        "unmapped_relation_ratio": round(unmapped_rels / len(relations), 4) if relations else None,
        "unknown_type_entities": unknown_type_ents,
        "duplicates_merged_entities": ingest_stats.get("entities_raw", 0) - len(entities),
        "duplicates_merged_relations": ingest_stats.get("relations_raw", 0) - len(relations),
        "triples_input": ingest_stats.get("relations_raw", 0),
        "triples_unique": len(relations),
        "triples_duplicate_removed": ingest_stats.get("relations_raw", 0) - len(relations),
        "missing_provenance": missing_prov,
        "quality_fail_entities": fail_ents,
        "quality_fail_relations": fail_rels,
        "predicate_distribution": dict(pred_counter.most_common()),
        "relation_type_distribution": dict(rel_type_counter.most_common()),
    }
