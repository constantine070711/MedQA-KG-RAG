# -*- coding: utf-8 -*-
"""Phase 10-A.5 Step B1 — CMeIE → LightRAG custom_kg 结构化构建器（旁路，零 LLM）。

核心原则：CMeIE 已提供的结构化知识（subject/predicate/object），
不再让 LLM 重复抽取。仅做数据结构转换 + 本地 embedding。

输入：CMeIE 记录（原始 train 前 N 条，含 text + spo_list）
输出：LightRAG 1.5.4 `ainsert_custom_kg` 可接受的结构：
    chunks:        [{content, source_id, file_path, chunk_order_index}]   （每记录一 chunk，content=原始 text）
    entities:      [{entity_name, entity_type, description, source_id, file_path}]
    relationships: [{src_id, tgt_id, description, keywords, weight, source_id, file_path}]

设计要点：
  1. 多 predicate 防覆盖：LightRAG 关系按 (src,tgt) 无向去重（后写覆盖前写）。
     故按 (subject, object) 聚合全部 predicate，合并进 keywords / description：
        keywords   = 映射后 RelationType 的并集（逗号分隔，供检索匹配）
        description = 原始 predicate 的并集（分号分隔，可追溯）
     同一实体对的多个 predicate 不丢失、不覆盖。
  2. Provenance：每个 chunk 的 source_id = "medkb_cmeie_{sample_id:05d}"；
     entity/relationship 的 source_id 指向其首次出现的记录 chunk。
     独立 provenance manifest 提供 chunk/entity/relation ID ↔ CMeIE sample_id ↔ 原始 text 追溯。
  3. 幂等 / deterministic / 不修改输入 / 不调用 LLM。

禁止：改 LightRAG 源码 / monkey-patch / 改生产 / 伪造关系 / 丢失原始 predicate。
"""

from collections import OrderedDict
from pathlib import Path

from .medical_kb_adapter import _unwrap_value, CMEIE_TYPE_MAP, MEDKB_RELATION_MAP

# 常量：与 Phase 10-A 产物对齐
DATASET = "CMeIE"
SPLIT = "train"
ADAPTER = "medkb_custom_kg"
LICENSE = "CC-BY-NC 4.0"
INGEST_DATE = "2026-09-01"
FILE_PATH = "cmeie"


def _chunk_source_id(sample_id: int) -> str:
    return f"medkb_cmeie_{sample_id:05d}"


def _map_entity_type(raw_type: str) -> str:
    """CMeIE 类型 → 项目 EntityType（映射清晰者；未知如实 "UNKNOWN"）。"""
    return CMEIE_TYPE_MAP.get(str(raw_type or "").strip(), "UNKNOWN")


def _map_relation_type(predicate: str) -> str:
    """predicate → 映射后 RelationType 或原词（UNMAPPED 时 keywords 用原词）。"""
    rt = MEDKB_RELATION_MAP.get(predicate)
    return rt if rt is not None else predicate


def build_custom_kg(records: list[dict], file_path: str = FILE_PATH) -> dict:
    """从 CMeIE 记录构建 custom_kg + 统计。

    Args:
        records: list[dict]，每条含 {sample_id:int, text:str, spo_list:list}
                 （spo_list 元素含 subject/subject_type/predicate/object/object_type）

    Returns:
        {"chunks": [...], "entities": [...], "relationships": [...],
         "provenance_manifest": {...}, "stats": {...}}
    """
    chunks: list[dict] = []
    entities: OrderedDict[str, dict] = OrderedDict()   # entity_name -> data
    pair_preds: OrderedDict[tuple[str, str], dict] = OrderedDict()  # (subj,obj) -> merged info
    chunk_to_sample: dict[str, int] = {}               # chunk source_id -> sample_id

    # ---- 第一遍：chunks + 聚合 (subject, object) 的所有 predicate ----
    for rec in records:
        sample_id = int(rec.get("sample_id", 0))
        text = str(rec.get("text", "") or "").strip()
        src_id = _chunk_source_id(sample_id)
        chunk_to_sample[src_id] = sample_id
        chunks.append({
            "content": text,
            "source_id": src_id,
            "file_path": file_path,
            "chunk_order_index": 0,
        })
        for spo in rec.get("spo_list", []) or []:
            if not isinstance(spo, dict):
                continue
            subj = _unwrap_value(spo.get("subject"))
            obj = _unwrap_value(spo.get("object"))
            pred = str(spo.get("predicate", "") or "").strip()
            if not subj or not obj or not pred:
                continue
            key = (subj, obj)
            info = pair_preds.setdefault(key, {
                "predicates": set(), "relation_types": set(),
                "records": set(), "sample_id_first": sample_id,
            })
            info["predicates"].add(pred)
            info["relation_types"].add(_map_relation_type(pred))
            info["records"].add(sample_id)

    # ---- 第二遍：entities（subject/object 去重，类型=首次出现记录的类型，source_id=首现 chunk）----
    first_type: dict[str, str] = {}
    for rec in records:
        for spo in rec.get("spo_list", []) or []:
            if not isinstance(spo, dict):
                continue
            subj = _unwrap_value(spo.get("subject"))
            obj = _unwrap_value(spo.get("object"))
            if subj and subj not in first_type:
                first_type[subj] = _map_entity_type(spo.get("subject_type"))
            if obj and obj not in first_type:
                first_type[obj] = _map_entity_type(_unwrap_value(spo.get("object_type")))

    for (subj, obj), info in pair_preds.items():
        first_src = _chunk_source_id(info["sample_id_first"])
        for name in (subj, obj):
            if name in entities:
                continue
            entities[name] = {
                "entity_name": name,
                "entity_type": first_type.get(name, "UNKNOWN"),
                "description": f"CMeIE 实体：{name}（首现记录 {first_src}）",
                "source_id": first_src,
                "file_path": file_path,
            }

    # ---- 第三遍：relationships（多 predicate 合并为单边）----
    relationships: list[dict] = []
    multi_pred_count = 0
    for (subj, obj), info in pair_preds.items():
        first_src = _chunk_source_id(info["sample_id_first"])
        predicates_sorted = sorted(info["predicates"])
        relation_types_sorted = sorted(info["relation_types"])
        if len(predicates_sorted) > 1:
            multi_pred_count += 1
        relationships.append({
            "src_id": subj,
            "tgt_id": obj,
            "description": "; ".join(predicates_sorted),          # 原始 predicate 并集（可追溯）
            "keywords": ", ".join(relation_types_sorted),          # 映射类型并集（检索匹配）
            "weight": 1.0,
            "source_id": first_src,
            "file_path": file_path,
        })

    # ---- Provenance manifest ----
    manifest = {
        "dataset": DATASET, "split": SPLIT, "adapter": ADAPTER,
        "license": LICENSE, "ingest_date": INGEST_DATE,
        "chunks": {src: {"sample_id": sid}
                   for src, sid in chunk_to_sample.items()},
        "entities": {name: {"source_id": e["source_id"],
                            "source_sample_id": chunk_to_sample.get(e["source_id"])}
                     for name, e in entities.items()},
        "relationships": [{"src_id": r["src_id"], "tgt_id": r["tgt_id"],
                           "predicates": r["description"], "source_id": r["source_id"],
                           "source_sample_id": chunk_to_sample.get(r["source_id"])}
                          for r in relationships],
    }

    stats = {
        "records": len(records),
        "chunks": len(chunks),
        "entities": len(entities),
        "relationships": len(relationships),
        "triples_input": sum(len(info["predicates"]) for info in pair_preds.values()),
        "unique_subject_object_pairs": len(pair_preds),
        "multi_predicate_pairs": multi_pred_count,
        "max_predicates_per_pair": max((len(i["predicates"]) for i in pair_preds.values()), default=0),
        "llm_extraction_calls": 0,  # 本 builder 零 LLM
    }

    return {"chunks": chunks, "entities": list(entities.values()),
            "relationships": relationships,
            "provenance_manifest": manifest, "stats": stats}


def build_custom_kg_from_file(records_path, limit: int = None, file_path: str = FILE_PATH) -> dict:
    """从 CMeIE 原始 train jsonl 读取前 N 条并构建 custom_kg。"""
    import json
    p = Path(records_path)
    records = []
    with open(p, encoding="utf-8") as f:
        for sample_id, line in enumerate(f):
            if limit is not None and sample_id >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append({
                "sample_id": sample_id,
                "text": rec.get("text", ""),
                "spo_list": rec.get("spo_list", []),
            })
    return build_custom_kg(records, file_path=file_path)
