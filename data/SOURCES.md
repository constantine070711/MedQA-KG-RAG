# 数据来源与许可证记录

> 更新日期: 2026-07-25
> Phase 3: 医疗知识库工程化建设

---

## 数据目录结构

```
data/
├── raw/
│   ├── diseases/       # 疾病知识 (9 个 .md 文件)
│   ├── drugs/          # 药物知识 (5 个 .md 文件)
│   ├── symptoms/       # 症状知识 (预留)
│   └── guidelines/     # 临床指南 (预留)
├── processed/          # 处理后的数据 (预留, 待 Neo4j 导入)
├── metadata/
│   └── index.json      # 元数据索引 (自动生成)
└── SOURCES.md          # 本文件
```

---

## 已导入数据清单

### 疾病知识 (data/raw/diseases/)

| 文件 | 实体 | ICD-10 | 类别 | 字符数 |
|------|------|:--:|------|:--:|
| diabetes-type2.md | 2型糖尿病 | E11 | 内分泌与代谢 | ~550 |
| hypertension.md | 高血压 | I10 | 心血管 | ~520 |
| common-cold.md | 感冒 | J00-J06 | 呼吸系统 | ~480 |
| hyperlipidemia.md | 高脂血症 | E78 | 内分泌与代谢 | ~550 |
| coronary-heart-disease.md | 冠心病 | I20-I25 | 心血管 | ~650 |
| copd.md | 慢性阻塞性肺疾病 | J44 | 呼吸系统 | ~630 |
| chronic-gastritis.md | 慢性胃炎 | K29 | 消化系统 | ~620 |
| stroke.md | 脑卒中 | I60-I69 | 神经系统 | ~750 |
| bronchial-asthma.md | 支气管哮喘 | J45 | 呼吸系统 | ~580 |

**小计**: 9 个疾病, ~5,330 字符

### 药物知识 (data/raw/drugs/)

| 文件 | 实体 | ATC | 类型 | 字符数 |
|------|------|:--:|------|:--:|
| metformin.md | 二甲双胍 | A10BA02 | 双胍类口服降糖药 | ~420 |
| aspirin.md | 阿司匹林 | N02BA01 | 非甾体抗炎药 | ~470 |
| amoxicillin.md | 阿莫西林 | J01CA04 | 青霉素类抗生素 | ~430 |
| omeprazole.md | 奥美拉唑 | A02BC01 | 质子泵抑制剂 | ~500 |
| nifedipine.md | 硝苯地平 | C08CA05 | 钙通道阻滞剂 | ~500 |

**小计**: 5 个药物, ~2,320 字符

### 总计

| 指标 | 数值 |
|------|------|
| 总文件数 | 14 |
| 总字符数 | ~7,650 |
| 元数据字段 | source / entity_type / name / aliases / standard_code / category / license / dates |
| 格式 | Markdown + YAML frontmatter |

---

## 数据来源详情

### 当前数据: 公有领域医学教科书知识

- **来源**: 基于公开医学教科书知识人工整理
- **许可证**: 公有领域（医学知识点本身不受版权保护，文字表述为自主编写）
- **覆盖**: 内分泌/心血管/呼吸/消化/神经系统常见疾病及常用药物
- **格式**: Markdown + YAML frontmatter（每个文件包含结构化元数据 + 自由文本正文）

### 元数据字段标准

每个 .md 文件包含 YAML frontmatter:

```yaml
---
source: "Public medical textbook knowledge — human-curated"
entity_type: "disease"          # disease | drug | symptom | guideline
name: "2型糖尿病"
icd10_code: "E11"              # ICD-10 (疾病) 或 ATC (药物)
category: "内分泌与代谢疾病"
created_date: "2026-07-25"
updated_date: "2026-07-25"
license: "Public Domain — factual medical knowledge, independently written"
aliases: ["Type 2 Diabetes Mellitus", "T2DM"]
---
```

---

## 推荐数据源（待后续获取）

> ⚠️ License 审查日期: 2026-07-26 (Phase 4)
> 审查结论: 仅 CBLUE 3.0 (CC BY-NC-SA 4.0) 可安全用于学术研究

### 高优先级

| 数据源 | 描述 | 获取方式 | 许可证 | 审查状态 |
|--------|------|----------|--------|:--:|
| **CMeKG 2.0** | 中文医学知识图谱，含疾病/症状/药物/诊疗等 1万+ 实体 | http://cmekg.pcl.ac.cn/ | **CC-BY-ND** | ⚠️ 仅本地参考 |
| **CBLUE 3.0** | 中文医疗信息处理评测基准，含 18 个 NLP 数据集 | https://tianchi.aliyun.com/dataset/177390 | **CC BY-NC-SA 4.0** | ✅ 可用于学术 |
| **CMeEE-V2** | 中文医学命名实体识别数据集（9大类实体） | CBLUE 子集，天池下载 | CC BY-NC-SA 4.0 | ✅ 可用于学术 |
| **CMeIE-V2** | 中文医学文本实体关系抽取（53种关系） | CBLUE 子集，天池下载 | CC BY-NC-SA 4.0 | ✅ 可用于学术 |

### CMeKG 2.0 (CC-BY-ND) 限制说明

- ⚠️ **禁止演绎 (No Derivatives)**: 不可分发修改后的版本
- ✅ 可用于**本地研究参考**
- ✅ 可用于**论文对比实验引用**
- ❌ **不可**将转换/修改后的 CMeKG 数据纳入项目代码仓库
- 建议: 通过对比检验本项目知识图谱的覆盖度

### CBLUE 3.0 (CC BY-NC-SA 4.0) 使用条件

- ✅ 必须**署名** (BY): 引用 CBLUE 论文和天池页面
- ✅ 仅限**非商业**使用 (NC): 学术研究符合要求
- ✅ **相同方式共享** (SA): 衍生作品需沿用相同协议
- ✅ 可下载、修改、分发（同协议下）
- 📎 论文引用: Zhang N, Chen M, Bi Z, et al. CBLUE: A Chinese Biomedical Language Understanding Evaluation Benchmark. ACL 2022.

### 中优先级

| 数据源 | 描述 | 获取方式 | 许可证 |
|--------|------|----------|--------|
| **CHIP-CDN** | 临床术语标准化（ICD-10映射） | CBLUE 子集，天池下载 | CC BY-NC-SA 4.0 |
| **CHIP-STS** | 疾病问答语义相似度 | CBLUE 子集，天池下载 | CC BY-NC-SA 4.0 |
| **IMCS-V2** | 智能诊疗对话系统数据集 | CBLUE 子集，天池下载 | CC BY-NC-SA 4.0 |

### 后续参考

| 数据源 | 描述 | 链接 |
|--------|------|------|
| QASystemOnMedicalKG | 中文医疗 KG 问答系统（含数据） | https://github.com/longyaoyoudu/Multilevel_Decomposition_Medical_Agent |
| CMeKG Tools | CMeKG 工具链（NER/RE模型） | https://github.com/king-yyf/CMeKG |
| OpenKG 医疗数据集 | 开放知识图谱 — 医疗类 | http://data.openkg.cn/ |

---

## 数据使用原则

1. **优先使用公有领域知识**：医学教科书级别的知识属于公有领域
2. **标注来源**：所有外部数据集使用时必须在 YAML frontmatter 中标注 `source` 和 `license`
3. **合规使用**：仅使用允许学术研究用途的数据集（CC BY-NC-SA 等）
4. **不爬取受版权保护的内容**：不爬取百度百科、维基百科等
5. **敏感数据脱敏**：不使用包含真实患者信息的临床数据
6. **免责声明**：系统明确标注"不能替代医生诊断"
7. **每条知识可追溯**：通过 metadata 系统记录完整的来源链

---

## 变更记录

| 日期 | 变更 |
|------|------|
| 2026-07-25 | 初始创建，记录 MVP 数据和推荐数据源 |
| 2026-07-25 | Phase 3: 重建目录结构，14 个 .md 文件 + YAML frontmatter 元数据 |
| 2026-07-26 | Phase 4: License 审查完成 — CMeKG (CC-BY-ND, 仅本地) / CBLUE (CC BY-NC-SA 4.0, 可用) |
