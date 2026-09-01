# -*- coding: utf-8 -*-
"""Phase 10-B B2 — 旁路检索级 Reranker 包（实验性，不进入生产链路）。

原则（docs/phase10_b_reranker_design_audit.md）：
  - 旁路：不修改生产 Agent / RAG / Fusion / Neo4j；Neo4j canonical evidence 不进入 rerank
  - 可回退：模型不可用 / 推理异常 / score 异常 → 返回原始候选顺序（截取 top_m）
  - 不修改输入对象
  - CPU 可运行
"""
