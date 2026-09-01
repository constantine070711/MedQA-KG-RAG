"""
CBLUE 数据适配器 Stub (预留接口)

为 CBLUE 3.0 中文医疗数据集提供适配器接口定义。

许可证状态 (2026-07-26):
    - CBLUE 3.0: CC BY-NC-SA 4.0 — ✅ 可用于学术研究
    - 下载地址: https://tianchi.aliyun.com/dataset/177390
    - 数据子集:
        CMeEE-V2  — 中文医学命名实体识别 (9类实体)
        CMeIE-V2  — 中文医学实体关系抽取 (53种关系)
        CHIP-CDN  — 临床术语标准化 (ICD-10映射)
        CHIP-STS  — 疾病问答语义相似度
        IMCS-V2   — 智能诊疗对话

CMeKG 2.0 状态:
    - 许可证: CC-BY-ND (禁止演绎)
    - 限制: 不可分发修改后的版本
    - 建议: 仅限本地参考研究，不纳入项目分发

数据格式参考:
    CMeEE-V2: {
        "text": "患者3天前出现发热、咳嗽...",
        "entities": [
            {"start_idx": 3, "end_idx": 7, "type": "sym", "entity": "发热"},
            ...
        ]
    }

    CMeIE-V2: {
        "text": "阿司匹林用于治疗...",
        "spo_list": [
            {"subject": "阿司匹林", "predicate": "治疗", "object": "头痛"},
            ...
        ]
    }

使用前提:
    1. 从天池平台完成学生认证
    2. 下载数据集到 data/external/cblue/
    3. 在 SOURCES.md 中记录下载时间和许可证
    4. 不要将原始数据集提交到 Git

当前状态:
    ⚠️ STUB — 接口已定义，待数据下载后实现具体解析逻辑
"""

import logging
from pathlib import Path
from typing import Optional

from .base_adapter import DataSourceAdapter

logger = logging.getLogger(__name__)


class CBLUEAdapter(DataSourceAdapter):
    """
    CBLUE 3.0 数据适配器 Stub。

    为 CMeEE-V2（命名实体识别）和 CMeIE-V2（关系抽取）数据集
    提供适配器接口。当前为预留接口，待数据下载后实现。

    支持的数据子集:
        - "cmeee_v2": 命名实体识别 (text + entities)
        - "cmeie_v2": 关系抽取 (text + spo_list)
        - "chip_cdn": 术语标准化
    """

    name: str = "cblue"
    description: str = (
        "CBLUE 3.0 Chinese Biomedical Language Understanding Evaluation adapter"
    )
    supported_formats: list[str] = [".json", ".jsonl"]

    # CMeEE-V2 实体类型映射 (CBLUE → 本项目 EntityType)
    CMEEE_ENTITY_TYPE_MAP = {
        "dis": "disease",       # 疾病
        "sym": "symptom",       # 症状
        "dru": "drug",          # 药物
        "pro": "examination",   # 检查
        "equ": "examination",   # 医疗设备
        "dep": "department",    # 科室
        "bod": "anatomy",       # 身体部位
        "ite": "examination",   # 检验项目
        "mic": "biomarker",     # 微生物
    }

    # CMeIE-V2 关系类型映射 (CBLUE → 本项目 RelationType)
    # 完整 53 种关系待数据下载后逐项映射
    CMEIE_RELATION_TYPE_MAP = {}  # TODO: 根据实际数据填充

    def __init__(self, data_dir: Optional[str] = None, subset: str = "cmeee_v2"):
        super().__init__(source_path=data_dir)
        self.subset = subset

    def ingest(self, source_path: Optional[str] = None) -> list[dict]:
        """
        Stub — 暂不实现。

        需要在 data/external/cblue/ 下有实际数据文件后才能工作。

        Raises:
            NotImplementedError: CBLUE 数据未下载
        """
        raise NotImplementedError(
            "CBLUE 数据尚未下载。请先从天池平台获取数据:\n"
            "  1. 访问 https://tianchi.aliyun.com/dataset/177390\n"
            "  2. 完成学生认证\n"
            "  3. 下载 CMeEE-V2 / CMeIE-V2 数据\n"
            "  4. 放置到 data/external/cblue/\n"
            "  5. 在 data/SOURCES.md 中记录下载信息和许可证\n"
            "  6. 重新调用此方法"
        )

    def get_license_info(self) -> dict:
        """返回许可证信息"""
        return {
            "dataset": "CBLUE 3.0",
            "license": "CC BY-NC-SA 4.0",
            "attribution": (
                "CBLUE: A Chinese Biomedical Language Understanding "
                "Evaluation Benchmark. Tianchi & CHIP."
            ),
            "download_url": "https://tianchi.aliyun.com/dataset/177390",
            "paper_citation": (
                "Zhang N, Chen M, Bi Z, et al. CBLUE: A Chinese Biomedical "
                "Language Understanding Evaluation Benchmark. ACL 2022."
            ),
            "restrictions": [
                "仅限非商业学术研究使用",
                "衍生作品需沿用相同协议",
                "必须注明出处",
            ],
            "can_redistribute": "Yes (under same license CC BY-NC-SA 4.0)",
            "status": "NOT YET DOWNLOADED",
        }


class CMeKGAdapter(DataSourceAdapter):
    """
    CMeKG / CPubMed-KG 2.0 数据适配器 Stub。

    许可证: CC-BY-ND (署名-禁止演绎)
    ⚠️ 重要限制: 不可分发修改后的数据版本。

    建议使用方式:
        - 仅用于本地研究参考
        - 不纳入项目代码仓库
        - 通过对比检验本项目知识图谱的覆盖度
        - 在论文中引用 CMeKG 作为 Baseline 对比

    数据格式 (CPubMed-KG 2.0):
        三元组格式 (subject, predicate, object)
        覆盖: 疾病 / 症状 / 药物 / 诊疗 / 解剖 等
    """

    name: str = "cmekg"
    description: str = (
        "CMeKG / CPubMed-KG 2.0 adapter (CC-BY-ND, local reference only)"
    )
    supported_formats: list[str] = [".json", ".csv", ".txt"]

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(source_path=data_dir)

    def ingest(self, source_path: Optional[str] = None) -> list[dict]:
        """
        Stub — 暂不实现。

        ⚠️ CMeKG 使用 CC-BY-ND 许可证，不可分发修改后的数据。
        此适配器仅用于本地研究参考。
        """
        raise NotImplementedError(
            "CMeKG 使用 CC-BY-ND 许可证，仅限于本地参考。\n"
            "CMeKG 官方: http://cmekg.pcl.ac.cn/\n"
            "CPubMed-KG 2.0: https://cpubmed.openi.org.cn/graph/wiki\n"
            "建议仅用于本地对比研究，不纳入项目分发。"
        )

    def get_license_info(self) -> dict:
        """返回许可证信息"""
        return {
            "dataset": "CMeKG / CPubMed-KG 2.0",
            "license": "CC-BY-ND",
            "attribution": (
                "CPubMed-KG 2.0 — HIT(SZ) ICRC, Chen Qingcai et al."
            ),
            "download_url": "https://cpubmed.openi.org.cn/graph/wiki",
            "restrictions": [
                "⚠️ 禁止演绎 (No Derivatives)",
                "不可分发修改后的数据版本",
                "可用于本地研究参考",
                "可用于论文对比实验引用",
            ],
            "can_redistribute": "NO — CC-BY-ND prohibits distribution of modified versions",
            "recommendation": (
                "仅本地参考，不纳入项目代码仓库。"
                "在论文中引用作为 Baseline 对比。"
            ),
        }
