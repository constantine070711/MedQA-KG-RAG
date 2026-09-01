# -*- coding: utf-8 -*-
"""Phase 10-A — CBLUE 数据适配器（独立实验旁路）。

输入：CBLUE 3.0 原始格式（CMeEE-V2 / CMeIE-V2，.json / .jsonl）
输出：统一医学知识格式（CBLUEEntity / CBLUERelation，均带 provenance）

本阶段范围（严格）：
  ✅ 解析 / 转换 / 校验 / Entity Linking 候选
  ❌ 不实现：自动 merge KG、Neo4j 写入、LightRAG ingest、渲染 RAG 文档

许可证：CBLUE 3.0 — CC BY-NC-SA 4.0（学术非商业，须署名，同方式共享）。
所有派生产物强制携带 provenance（dataset/split/sample_id/source_file/license）。

数据格式参考（CBLUE 官方 schema）：
  CMeEE-V2: {"text": "...", "entities": [{"start_idx","end_idx","type","entity"}]}
  CMeIE-V2: {"text": "...", "spo_list": [{"subject","predicate","object"}]}
"""

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ============================================================
# 常量
# ============================================================

CBLUE_LICENSE = "CC BY-NC-SA 4.0"
ADAPTER_NAME = "cblue"
INGEST_DATE = "2026-08-25"

SUBSETS = ("cmeee_v2", "cmeie_v2")

# CBLUE CMeEE-V2 9 类实体 → 本项目实体类型（复用既有 Stub 映射）
# 注：bod/mic 在本项目 EntityType 无对应，如实保留为字符串并记录 warning。
CMEEE_TYPE_MAP = {
    "dis": "disease",
    "sym": "symptom",
    "dru": "drug",
    "pro": "examination",
    "equ": "examination",
    "dep": "department",
    "bod": "anatomy",
    "ite": "examination",
    "mic": "biomarker",
}

# 本项目 EntityType 已知类型（用于 quality warning）
KNOWN_ENTITY_TYPES = {
    "disease", "drug", "symptom", "examination", "department",
    "anatomy", "biomarker", "guideline", "unknown",
}

# CMeIE-V2 关系（中文 predicate）→ 本项目 RelationType 高置信子集。
# 未收录的 predicate 一律 relation_type="UNMAPPED"，如实保留原词（不伪造映射）。
CMEIE_RELATION_MAP = {
    "治疗": "TREATS",
    "症状": "HAS_SYMPTOM",
    "并发症": "HAS_COMPLICATION",
    "禁忌": "CONTRAINDICATED_FOR",
    "不良反应": "HAS_ADVERSE_REACTION",
    "诊断": "DIAGNOSES",
    "筛查": "SCREENS_FOR",
    "鉴别诊断": "DIFFERENTIAL_DIAGNOSIS",
    "危险因素": "HAS_RISK_FACTOR",
    "药物相互作用": "DRUG_INTERACTION",
    "联合用药": "ADMINISTERED_WITH",
}

# RelationType 已知值（warning 用）
KNOWN_RELATION_TYPES = set(CMEIE_RELATION_MAP.values()) | {"UNMAPPED"}

# 标准编码系统
CODE_SYSTEMS = {"ICD-10", "ATC", "SNOMED_CT"}

_WS_RE = re.compile(r"[\s　]+")


# ============================================================
# 统一数据模型（CBLUE 专用，含 provenance / quality）
# ============================================================

@dataclass
class CBLUEEntity:
    """统一医学实体（CBLUE 派生）。"""

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
class CBLUERelation:
    """统一医学关系三元组（CBLUE 派生）。"""

    subject: str
    relation: str            # RelationType.value（已映射）或原 predicate（未映射）
    object: str
    relation_type: str       # RelationType.value 或 "UNMAPPED"
    provenance: dict = field(default_factory=dict)
    confidence: float = 1.0
    quality: dict = field(default_factory=lambda: {"level": "pass", "errors": [], "warnings": []})

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "relation_type": self.relation_type,
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

def _norm(text: str) -> str:
    """归一化：NFKC + 去空白，用于名称匹配（Entity Linking L3）。"""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = _WS_RE.sub("", s)
    return s.lower()


def _make_provenance(dataset: str, split: str, sample_id: str,
                     source_file: str) -> dict:
    return {
        "dataset": dataset,
        "split": split,
        "sample_id": sample_id,
        "source_file": str(source_file),
        "adapter": ADAPTER_NAME,
        "license": CBLUE_LICENSE,
        "ingest_date": INGEST_DATE,
    }


def _mk_entity_id(subset: str, split: str, sample_id: str, idx: int) -> str:
    return f"cblue:{subset}:{split}:{sample_id}:{idx}"


# ============================================================
# 解析：CMeEE-V2（命名实体识别）
# ============================================================

def parse_cmeee_record(record: dict, split: str = "train",
                       source_file: str = "", idx_base: int = 0) -> dict:
    """解析单条 CMeEE-V2 记录 → CBLUEEntity 列表。

    Returns:
        {"entities": [CBLUEEntity], "skipped": [dict], "stats": {...}}
    """
    if not isinstance(record, dict):
        return {"entities": [], "skipped": [{"reason": "record 非 dict"}], "stats": {}}

    sample_id = str(record.get("sample_id", "") or "")
    text = str(record.get("text", "") or "")
    anns = record.get("entities", [])
    if not isinstance(anns, list):
        anns = []

    if not sample_id:
        return {"entities": [], "skipped": [{"reason": "missing sample_id"}], "stats": {}}
    if not text.strip():
        return {"entities": [], "skipped": [{"reason": "empty text", "sample_id": sample_id}], "stats": {}}

    entities: list[CBLUEEntity] = []
    skipped: list[dict] = []
    provenance = _make_provenance("CMeEE-V2", split, sample_id, source_file)

    for i, a in enumerate(anns):
        if not isinstance(a, dict):
            skipped.append({"sample_id": sample_id, "reason": "entity 非 dict"})
            continue
        ent_name = str(a.get("entity", "") or "").strip()
        if not ent_name:
            # 回退：用 text span 截取
            try:
                ent_name = text[int(a["start_idx"]):int(a["end_idx"])].strip()
            except (KeyError, TypeError, ValueError):
                ent_name = ""
        if not ent_name:
            skipped.append({"sample_id": sample_id, "reason": "empty entity name", "index": i})
            continue

        raw_type = str(a.get("type", "") or "")
        etype = CMEEE_TYPE_MAP.get(raw_type, "unknown")
        warnings = []
        if etype == "unknown":
            warnings.append(f"unknown cblue type: {raw_type!r}")
        if etype not in KNOWN_ENTITY_TYPES:
            warnings.append(f"type not in project EntityType: {etype}")

        quality = {"level": "pass" if not warnings else "warn",
                   "errors": [], "warnings": warnings}
        entities.append(CBLUEEntity(
            entity_id=_mk_entity_id("cmeee_v2", split, sample_id, idx_base + i),
            name=ent_name,
            entity_type=etype,
            aliases=[],
            standard_codes=[],
            raw_text=text[:500],
            provenance=provenance,
            quality=quality,
        ))

    stats = {
        "records_processed": 1,
        "entities_extracted": len(entities),
        "skipped": len(skipped),
    }
    return {"entities": entities, "skipped": skipped, "stats": stats}


# ============================================================
# 解析：CMeIE-V2（关系抽取）
# ============================================================

def parse_cmeie_record(record: dict, split: str = "train",
                       source_file: str = "", idx_base: int = 0) -> dict:
    """解析单条 CMeIE-V2 记录 → CBLUERelation + 实体名候选。

    Returns:
        {"entities": [CBLUEEntity], "relations": [CBLUERelation],
         "skipped": [dict], "stats": {...}}
    """
    if not isinstance(record, dict):
        return {"entities": [], "relations": [], "skipped": [{"reason": "record 非 dict"}], "stats": {}}

    sample_id = str(record.get("sample_id", "") or "")
    text = str(record.get("text", "") or "")
    spo = record.get("spo_list", [])
    if not isinstance(spo, list):
        spo = []

    if not sample_id:
        return {"entities": [], "relations": [], "skipped": [{"reason": "missing sample_id"}], "stats": {}}
    if not text.strip():
        return {"entities": [], "relations": [], "skipped": [{"reason": "empty text", "sample_id": sample_id}], "stats": {}}

    relations: list[CBLUERelation] = []
    skipped: list[dict] = []
    provenance = _make_provenance("CMeIE-V2", split, sample_id, source_file)

    # 实体名候选（subject/object 去重，类型未知 → warning）
    entity_names: dict[str, CBLUEEntity] = {}

    def _candidate_entity(name: str) -> Optional[CBLUEEntity]:
        name = str(name or "").strip()
        if not name:
            return None
        if name in entity_names:
            return entity_names[name]
        e = CBLUEEntity(
            entity_id=_mk_entity_id("cmeie_v2", split, sample_id, len(entity_names)),
            name=name,
            entity_type="unknown",
            raw_text=text[:500],
            provenance=provenance,
            quality={"level": "warn", "errors": [],
                     "warnings": ["CMeIE 无类型标注，entity_type=unknown"]},
        )
        entity_names[name] = e
        return e

    for i, s in enumerate(spo):
        if not isinstance(s, dict):
            skipped.append({"sample_id": sample_id, "reason": "spo 非 dict", "index": i})
            continue
        subj = str(s.get("subject", "") or "").strip()
        obj = str(s.get("object", "") or "").strip()
        pred = str(s.get("predicate", "") or "").strip()
        if not subj or not obj or not pred:
            skipped.append({"sample_id": sample_id, "reason": "spo 缺字段", "index": i})
            continue

        rel_type = CMEIE_RELATION_MAP.get(pred, "UNMAPPED")
        warnings = []
        if rel_type == "UNMAPPED":
            warnings.append(f"unmapped predicate: {pred!r}")
        confidence = 1.0 if rel_type != "UNMAPPED" else 0.5
        quality = {"level": "pass" if not warnings else "warn",
                   "errors": [], "warnings": warnings}

        relations.append(CBLUERelation(
            subject=subj,
            relation=rel_type if rel_type != "UNMAPPED" else pred,
            object=obj,
            relation_type=rel_type,
            provenance=provenance,
            confidence=confidence,
            quality=quality,
        ))
        _candidate_entity(subj)
        _candidate_entity(obj)

    stats = {
        "records_processed": 1,
        "relations_extracted": len(relations),
        "entities_extracted": len(entity_names),
        "skipped": len(skipped),
    }
    return {"entities": list(entity_names.values()), "relations": relations,
            "skipped": skipped, "stats": stats}


# ============================================================
# 文件读取（.json 单对象/数组 / .jsonl）
# ============================================================

def _read_records(path: Path) -> list[dict]:
    """读取 CBLUE 数据文件，返回记录列表。

    支持：
      - .json: 顶层为单个对象（含 "records" 或直接是记录），或对象数组
      - .jsonl: 每行一条 JSON
    """
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
    # .json
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # 兼容 {"records": [...]} 或直接是单条记录
        if isinstance(data.get("records"), list):
            return data["records"]
        if "sample_id" in data or "text" in data or "spo_list" in data or "entities" in data:
            return [data]
        return []
    return []


# ============================================================
# 汇总 ingest / convert / 去重
# ============================================================

def ingest(input_path, subset: str, split: str = "train",
           limit: Optional[int] = None) -> dict:
    """读取目录/文件并解析为实体与关系候选。

    Args:
        input_path: 文件或目录（含 cblue 数据）。
        subset: "cmeee_v2" | "cmeie_v2"。
        split: 数据划分（默认 train）。
        limit: 最多处理记录数（None=全部）。

    Returns:
        {"subset", "split", "records_loaded", "entities", "relations",
         "skipped", "stats"}
    """
    if subset not in SUBSETS:
        raise ValueError(f"subset 必须是 {SUBSETS}，got {subset!r}")

    path = Path(input_path)
    if path.is_dir():
        files = sorted(path.glob(f"*{subset}*")) + sorted(path.glob(f"*.{subset}.json"))
        files = [f for f in files if f.suffix in (".json", ".jsonl")]
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    entities: list[CBLUEEntity] = []
    relations: list[CBLUERelation] = []
    skipped: list[dict] = []
    records_loaded = 0

    for f in files:
        try:
            records = _read_records(f)
        except Exception as exc:
            skipped.append({"file": str(f), "reason": f"read error: {exc}"})
            continue

        for r in records:
            if limit is not None and records_loaded >= limit:
                break
            if "_parse_error" in r:
                skipped.append({"file": str(f), "reason": "json parse error"})
                continue
            records_loaded += 1
            if subset == "cmeee_v2":
                res = parse_cmeee_record(r, split=split, source_file=str(f),
                                         idx_base=len(entities))
            else:
                res = parse_cmeie_record(r, split=split, source_file=str(f),
                                         idx_base=len(entities))
            entities.extend(res["entities"])
            relations.extend(res.get("relations", []))
            skipped.extend(res["skipped"])

    # 去重（基于 entity_id / triple 键）
    ent_dedup = {}
    for e in entities:
        ent_dedup.setdefault(e.entity_id, e)
    rel_dedup = {}
    for r_ in relations:
        key = (r_.subject, r_.relation_type, r_.object)
        rel_dedup.setdefault(key, r_)

    stats = {
        "subset": subset,
        "split": split,
        "records_loaded": records_loaded,
        "files": [str(f) for f in files],
        "entities_raw": len(entities),
        "entities_dedup": len(ent_dedup),
        "relations_raw": len(relations),
        "relations_dedup": len(rel_dedup),
        "skipped": len(skipped),
    }
    return {"subset": subset, "split": split, "records_loaded": records_loaded,
            "entities": list(ent_dedup.values()), "relations": list(rel_dedup.values()),
            "skipped": skipped, "stats": stats}


# ============================================================
# 校验
# ============================================================

def validate_entity(e: CBLUEEntity) -> dict:
    """校验实体字段完整性，返回 quality dict（不修改实体）。"""
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


def validate_relation(r: CBLUERelation) -> dict:
    """校验关系字段完整性，返回 quality dict（不修改关系）。"""
    errors, warnings = [], []
    if not r.subject:
        errors.append("missing subject")
    if not r.object:
        errors.append("missing object")
    if not r.relation:
        errors.append("missing relation")
    if not r.provenance:
        errors.append("missing provenance")
    if r.relation_type not in KNOWN_RELATION_TYPES:
        errors.append(f"unknown relation_type: {r.relation_type!r}")
    warnings.extend(r.quality.get("warnings", []))
    errors.extend(r.quality.get("errors", []))
    level = "fail" if errors else ("warn" if warnings else "pass")
    return {"level": level, "errors": errors, "warnings": warnings}


def convert(ingest_result: dict) -> dict:
    """统一化 ingest 结果 → {entities, relations}（dict 形式，用于写 JSON）。"""
    entities = [e.to_dict() for e in ingest_result.get("entities", [])]
    relations = [r.to_dict() for r in ingest_result.get("relations", [])]
    # 校验并回写 quality
    for e in entities:
        q = validate_entity(CBLUEEntity(**{k: e[k] for k in
            ("entity_id", "name", "entity_type", "aliases", "standard_codes",
             "raw_text", "provenance", "quality") if k in e}))
        e["quality"] = q
    for r in relations:
        q = validate_relation(CBLUERelation(**{k: r[k] for k in
            ("subject", "relation", "object", "relation_type",
             "confidence", "provenance", "quality") if k in r}))
        r["quality"] = q
    return {"entities": entities, "relations": relations}


# ============================================================
# Entity Linking 候选（L0-L3，只读 entities.json，不修改生产）
# ============================================================

def load_production_name_map(entities_json_path: Optional[str] = None) -> dict:
    """本地实现：读取生产 entities.json → {lower_name: canonical}。

    不 import evaluation/metrics（避免跨层耦合）。同名同逻辑。
    """
    if entities_json_path is None:
        entities_json_path = str(Path(__file__).resolve().parent.parent.parent
                                 / "data" / "entities" / "entities.json")
    with open(entities_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    name_map = {}
    for ent in data.get("entities", []):
        canonical = ent["name"]
        name_map[_norm(canonical)] = canonical
        for a in ent.get("aliases", []) or []:
            if isinstance(a, dict) and a.get("name"):
                name_map[_norm(a["name"])] = canonical
    return name_map


def load_production_codes(entities_json_path: Optional[str] = None) -> dict:
    """生产实体标准编码索引：{norm_code: canonical}（Entity Linking L0 用）。"""
    if entities_json_path is None:
        entities_json_path = str(Path(__file__).resolve().parent.parent.parent
                                 / "data" / "entities" / "entities.json")
    with open(entities_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    code_map = {}
    for ent in data.get("entities", []):
        canonical = ent["name"]
        for sc in ent.get("standard_codes", []) or []:
            if isinstance(sc, dict) and sc.get("code"):
                code_map[_norm(sc["code"])] = canonical
    return code_map


def link_entities(entities, prod_name_map: Optional[dict] = None,
                  prod_codes: Optional[dict] = None) -> dict:
    """生成 Entity Linking 候选（不 merge、不修改生产）。

    层级：
      L0 标准编码一致
      L1 规范名精确
      L2 别名匹配（name_map 含别名）
      L3 归一化匹配（弱信号 → candidate）

    Returns:
        {"linked": [...], "candidate": [...], "unmatched": [...]}
    """
    if prod_name_map is None:
        prod_name_map = load_production_name_map()
    if prod_codes is None:
        prod_codes = load_production_codes()

    linked, candidate, unmatched = [], [], []

    for e in entities:
        ent = e.to_dict() if hasattr(e, "to_dict") else e
        name = str(ent.get("name", "") or "")
        link = {
            "entity_id": ent.get("entity_id"),
            "name": name,
            "entity_type": ent.get("entity_type"),
            "matched": False,
            "canonical": None,
            "match_level": None,
            "confidence": 0.0,
        }
        if not name:
            unmatched.append(link)
            continue

        # L0: 标准编码一致（最强）
        code_hit = None
        for sc in ent.get("standard_codes", []) or []:
            if isinstance(sc, dict) and sc.get("code"):
                cand = prod_codes.get(_norm(sc["code"]))
                if cand:
                    code_hit = cand
                    break
        if code_hit:
            link.update(matched=True, canonical=code_hit, match_level="L0",
                        confidence=0.99)
            linked.append(link)
            continue

        # L1/L2: name_map（含规范名与别名）
        norm_name = _norm(name)
        mapped = prod_name_map.get(norm_name)
        if mapped:
            lvl = "L1" if _norm(mapped) == norm_name else "L2"
            link.update(matched=True, canonical=mapped, match_level=lvl,
                        confidence=0.95 if lvl == "L1" else 0.9)
            linked.append(link)
            continue

        # L3: 归一化包含/被包含（弱信号 → candidate）
        cand = None
        for pk in prod_name_map:
            if norm_name and (norm_name in pk or pk in norm_name) and len(norm_name) >= 2:
                cand = prod_name_map[pk]
                break
        if cand:
            link.update(matched=False, canonical=cand, match_level="L3",
                        confidence=0.5)
            candidate.append(link)
            continue

        unmatched.append(link)

    return {"linked": linked, "candidate": candidate, "unmatched": unmatched}


# ============================================================
# 汇总统计
# ============================================================

def summarize(convert_result: dict, ingest_stats: dict) -> dict:
    """生成可读统计：unresolved / duplicate / missing provenance 等。"""
    entities = convert_result.get("entities", [])
    relations = convert_result.get("relations", [])

    missing_prov = sum(1 for e in entities if not e.get("provenance"))
    missing_prov += sum(1 for r in relations if not r.get("provenance"))

    unmapped_rels = sum(1 for r in relations if r.get("relation_type") == "UNMAPPED")
    unknown_type_ents = sum(1 for e in entities
                            if e.get("entity_type") in ("unknown", "anatomy", "biomarker"))
    fail_ents = sum(1 for e in entities if e.get("quality", {}).get("level") == "fail")
    fail_rels = sum(1 for r in relations if r.get("quality", {}).get("level") == "fail")

    return {
        "entities": len(entities),
        "relations": len(relations),
        "triples": len(relations),
        "unmapped_relations": unmapped_rels,
        "unknown_type_entities": unknown_type_ents,
        "duplicates_merged_entities": ingest_stats.get("entities_raw", 0) - len(entities),
        "duplicates_merged_relations": ingest_stats.get("relations_raw", 0) - len(relations),
        "missing_provenance": missing_prov,
        "quality_fail_entities": fail_ents,
        "quality_fail_relations": fail_rels,
    }
