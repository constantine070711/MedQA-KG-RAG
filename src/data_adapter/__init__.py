# -*- coding: utf-8 -*-
"""Phase 10-A — 独立数据适配器包（CBLUE 实验旁路）。

本包是 CBLUE 数据的**实验旁路管道**，不参与生产链路：
  - 不修改生产知识库 / Neo4j / Agent / RAG / evaluation 基线
  - 输出统一医学知识格式到 data/processed/cblue/（独立隔离目录）
  - 本阶段仅实现：解析 / 转换 / 校验 / Entity Linking 候选
  - 不实现：自动 merge KG / Neo4j 写入 / LightRAG ingest

为避免耦合，本包不 export 到 src/__init__.py（核心包保持冻结不动）。
"""
