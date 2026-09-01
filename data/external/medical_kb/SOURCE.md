# 数据来源说明 — CMeIE (Medical Knowledge Base)

> 本目录存放 **真实原始医学数据**，保持原样、不做任何清洗/修改。
> 所有派生产物（清洗、转换、Entity Linking、Experiment KB）一律输出到 `data/raw/medical_kb/`、`data/processed/medical_kb/`、实验 cache 目录。

## 1. 数据集信息（Provenance）

| 字段 | 值 |
|------|------|
| **数据集名称** | CMeIE — 中文医学文本实体关系抽取数据集 |
| **官方来源** | OpenI 启智社区（鹏城实验室），CHIP 2020 评测任务二镜像 |
| **官方 URL** | https://openi.pcl.ac.cn/ZhangbuDong/CMeIE |
| **原始评测任务** | CHIP 2020（中国健康信息处理会议）评测任务二：中文医学文本实体关系抽取 — http://cips-chip.org.cn/2020/eval2 |
| **构建方** | 郑州大学自然语言处理实验室、北京大学计算语言学教育部重点实验室、鹏城实验室、哈尔滨工业大学（深圳） |
| **学术引用** | Guan, T., Zan, H., Zhou, X., Xu, H., Zhang, K. "CMeIE: Construction and Evaluation of Chinese Medical Information Extraction Dataset", NLPCC 2020 |
| **数据版本** | 2021-04-08（train/dev/test 文件）；README 更新于 2022-02-10 |
| **License** | **CC-BY-NC 4.0**（知识共享-署名-非商业使用）——声明于 OpenI 官方数据集页；**包内无独立 LICENSE 文件**，以页面声明为准 |
| **下载时间** | 2026-08-31 23:44（本地） |
| **原始文件名** | `CMeIE.zip`（原始压缩包存档为 `CMeIE_original.zip`） |
| **数据规模** | train **14,339** 条 / dev 3,585 条 / test 4,482 条；约 7.5 万三元组；53 种 schema 关系 |
| **本阶段使用** | **仅 train 14,339 条**（防测试集泄漏；dev/test 保留但不用） |

## 2. 数据格式（原始）

每条记录为 JSONL 一行：

```json
{
  "text": "医学文本原文（多句拼接时以 @疾病名 分隔前缀）",
  "spo_list": [
    {
      "Combined": false,
      "predicate": "鉴别诊断",
      "subject": "产后抑郁症",
      "subject_type": "疾病",
      "object": {"@value": "轻度情绪失调"},
      "object_type": {"@value": "疾病"}
    }
  ]
}
```

要点：
- `object` / `object_type` 为字典 `{"@value": "..."}`（非字符串）——与现有 `cblue_adapter` 期望的字符串 object 不同，需独立 adapter 或轻量 shim。
- 记录级**无 `sample_id`** 字段（现有 `cblue_adapter` 强制要求，需注入）。
- 53 种 schema 中部分与项目 `RelationType` 可映射（治疗/症状/并发症/禁忌/不良反应/诊断/鉴别诊断/危险因素等），其余如实记 `UNMAPPED`。

## 3. 文件清单

```
data/external/medical_kb/
├── SOURCE.md                    # 本文件
├── CMeIE_original.zip           # 原始压缩包（原样存档）
└── CMeIE/                       # 解压后的原始文件（原样）
    ├── README.txt               # 官方说明（文件清单 + 提交要求，无 License 文本）
    ├── 53_schemas.jsonl         # SPO 关系约束表（53 种）
    ├── CMeIE_train.jsonl        # 训练集 14,339 条 ← 本阶段唯一数据源
    ├── CMeIE_dev.jsonl          # 验证集 3,585 条（不使用）
    ├── CMeIE_test.jsonl         # 测试集 4,482 条（不使用）
    ├── example_gold.jsonl       # 标准答案示例
    └── example_pred.jsonl       # 提交格式示例
```

## 4. 与 CBLUE 的关系

CMeIE 是 **CBLUE（中文医疗信息处理挑战榜）** 基准任务之一（CBLUE 的 CMeIE-V2 即由此发展而来）。
本项目既有 `src/data_adapter/cblue_adapter.py` 的 `CMEIE_RELATION_MAP` 关系映射可复用。

## 5. 合规声明

- 仅用于**学术研究与项目演示**，不用于任何商业用途（CC-BY-NC 4.0）。
- 原始数据不修改、不覆盖、不二次分发；派生产物强制携带 provenance。
- 测试集泄漏红线：**dev/test 不入语料、不进 Experiment KB、不进 Benchmark**。
