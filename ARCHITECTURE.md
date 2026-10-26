# ARCHITECTURE.md — 系统架构文档

> 版本: v2（Next.js + FastAPI SSE + MCP + 语音/OCR）+ 已冻结的 v1 检索/证据/安全核心；
> CBLUE 数据适配、Reranker 重排、日志监控保持独立实验/旁路边界。
> 日期: 2026-10-19
> 底座: LightRAG 1.5.4

---

## 0. v2 应用架构

v2 不重写医疗检索核心，而是将 v1 已有的安全、检索、证据融合和可靠性决策
暴露给现代 Web 应用。

```
Next.js 16 + TypeScript evidence workspace
  ├── conversation and streamed answer
  ├── workflow state and tool trace
  ├── evidence inspector and graph view
  ├── Web Speech API voice input / playback
  └── Tesseract.js local image OCR
                  │
                  │ SSE + HTTP JSON
                  ▼
FastAPI v2 gateway
  ├── POST /api/chat/stream
  ├── GET  /api/tools
  ├── POST /api/tools/execute
  └── GET  /api/health, /api/meta
                  │
                  ▼
MedQAAgent LangGraph StateGraph
  safety_check
    -> neo4j_retrieve
    -> build_prompt
    -> generate
    -> analyze
    -> decide
    -> finalize
                  │
        ┌─────────┼──────────┐
        ▼         ▼          ▼
    LightRAG    Neo4j      MedicalToolbox
                         / MCP stdio server
```

### 0.1 Application Components

| Layer | Component | Responsibility |
|---|---|---|
| Browser | `web/src/components/medqa-workspace.tsx` | Conversation, streaming, input controls, evidence drawer |
| Browser | `web/src/components/evidence-panel.tsx` | Safety, reliability, tool trace, sources, citations, triples |
| Browser | `web/src/hooks/use-speech.ts` | Mandarin speech recognition and answer playback |
| Browser | `web/src/components/image-ocr-dialog.tsx` | Local OCR, editable text, supported-entity lookup |
| API | `api/runtime.py` | LangGraph node updates to SSE events and answer chunks |
| API | `api/toolbox.py` | Controlled HTTP tool execution |
| MCP | `mcp_server/medqa_mcp.py` | Five read-only tools over stdio |
| Core | `src/agent.py` | Inherited seven-node medical question workflow |
| Core | `src/tools/medical_tools.py` | Whitelisted LightRAG/Neo4j tool implementation |

### 0.2 SSE Event Contract

```text
status         safety / retrieval / fusion / generation / analysis / decision
tool           tool name, status, summary, elapsed time
answer_delta   streamed answer text
final          complete Agent response envelope
error          sanitized user-facing error
```

The server also emits SSE heartbeat comments while long generation is running.

### 0.3 Tool and MCP Boundary

- Tool names and argument names are whitelisted.
- Limits, enums, and string lengths are validated before execution.
- Cypher remains parameterized and read-only.
- No tool accepts arbitrary Cypher or file paths.
- The browser uses SSE traces for display; external agents can use the same
  retrieval tools through MCP.

### 0.4 Multimodal Boundary

- Voice recognition and speech synthesis run in the browser.
- OCR runs locally with Tesseract.js `chi_sim+eng`.
- Source images are not sent to the FastAPI service.
- OCR text is editable and does not automatically become medical evidence.
- Supported disease/drug entities can generate general knowledge questions,
  but no tool diagnoses an individual from an image.

### 0.5 Version Boundary

- `v1.0.0`: frozen Streamlit application and its verified retrieval core.
- `v2`: Next.js/FastAPI application, tools, MCP, voice, and image OCR.
- The historical sections below document the inherited v1 core and remain
  authoritative for retrieval, safety, evidence fusion, and evaluation.

---

## 1. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         医疗智能问答系统 (MedQA)                              │
│                     Medical Knowledge Graph + RAG System                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     Web UI Layer (Phase 8)                             │ │
│  │                        Streamlit App                                   │ │
│  │              ┌─────────────┬─────────────┬────────────┐               │ │
│  │              │  对话界面    │  来源引用    │  KG可视化   │               │ │
│  │              └─────────────┴─────────────┴────────────┘               │ │
│  └────────────────────────────────┬──────────────────────────────────────┘ │
│                                   │                                         │
│  ┌────────────────────────────────┼──────────────────────────────────────┐ │
│  │                    Agent Layer (Phase 7)                               │ │
│  │                       LangGraph Workflow                               │ │
│  │                                                                        │ │
│  │     ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │ │
│  │     │ 意图判断  │───▶│ 路由分发  │───▶│ 多路检索  │───▶│ 结果融合  │     │ │
│  │     │ (分类器)  │    │ (Router) │    │ (并行)    │    │ (Merge)  │     │ │
│  │     └──────────┘    └──────────┘    └──────────┘    └──────────┘     │ │
│  └────────────────────────────────┬──────────────────────────────────────┘ │
│                                   │                                         │
│  ┌────────────────────────────────┼──────────────────────────────────────┐ │
│  │                    Retrieval Layer (当前核心)                           │ │
│  │                       LightRAG Engine                                  │ │
│  │                                                                        │ │
│  │     ┌────────────────────────────────────────────────────────────┐    │ │
│  │     │                    Query Processing                         │    │ │
│  │     │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │    │ │
│  │     │  │  naive   │ │  local   │ │  global  │ │   mix    │      │    │ │
│  │     │  │(纯向量)   │ │(实体邻居) │ │(全局关系) │ │(全混合)  │      │    │ │
│  │     │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │    │ │
│  │     └────────────────────────────────────────────────────────────┘    │ │
│  │                                                                        │ │
│  │     ┌────────────────────────────────────────────────────────────┐    │ │
│  │     │                  Indexing Pipeline                          │    │ │
│  │     │  文本 → 分块 → 实体抽取 → 关系抽取 → 向量化 → 图合并 → 索引  │    │ │
│  │     └────────────────────────────────────────────────────────────┘    │ │
│  └────────────────────────────────┬──────────────────────────────────────┘ │
│                                   │                                         │
│  ┌────────────────────────────────┼──────────────────────────────────────┐ │
│  │                      Storage Layer                                    │ │
│  │                                                                        │ │
│  │     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │ │
│  │     │  Graph Store │  │ Vector Store │  │   KV Store   │             │ │
│  │     │              │  │              │  │              │             │ │
│  │     │  [MVP]       │  │  [MVP]       │  │  [MVP]       │             │ │
│  │     │  NetworkX    │  │  NanoVectorDB│  │  JSON        │             │ │
│  │     │              │  │              │  │              │             │ │
│  │     │  [Target]    │  │  [Long-term] │  │  [Target]    │             │ │
│  │     │  Neo4j       │  │  NanoVectorDB│  │  JSON        │             │ │
│  │     └──────────────┘  └──────────────┘  └──────────────┘             │ │
│  └────────────────────────────────┬──────────────────────────────────────┘ │
│                                   │                                         │
│  ┌────────────────────────────────┼──────────────────────────────────────┐ │
│  │                      External Services                                │ │
│  │                                                                        │ │
│  │     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │ │
│  │     │  DeepSeek    │  │  bge-small   │  │  HuggingFace │             │ │
│  │     │  API (LLM)   │  │  (Embedding) │  │  (模型下载)   │             │ │
│  │     └──────────────┘  └──────────────┘  └──────────────┘             │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 数据流 (Data Flow)

### 2.1 文档索引流程 (Indexing)

```
                          ┌─────────────┐
                          │  医疗文档     │
                          │ .txt/.pdf    │
                          └──────┬──────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Document Parser       │
                    │   LightRAG native/docling│
                    │   提取纯文本              │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Text Chunker          │
                    │   token_size=1200       │
                    │   overlap=100            │
                    └────────────┬───────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
            ┌──────────┐ ┌──────────┐ ┌──────────┐
            │  Chunk 1 │ │  Chunk 2 │ │  Chunk N │
            └────┬─────┘ └────┬─────┘ └────┬─────┘
                 │             │             │
                 ▼             ▼             ▼
         ┌───────────────────────────────────────────┐
         │         LLM Entity & Relation Extraction   │
         │         DeepSeek API (extract role)        │
         │                                            │
         │  Input:  text chunk                        │
         │  Output: entities[] + relations[]          │
         │  Cache:  KV Store (避免重复调用)            │
         └───────────────────┬───────────────────────┘
                             │
                 ┌───────────┼───────────┐
                 ▼           ▼           ▼
         ┌──────────┐ ┌──────────┐ ┌──────────┐
         │ Entities │ │Relations │ │  Chunks  │
         │ bge-small│ │ bge-small│ │ bge-small│
         │ Embedding│ │ Embedding│ │ Embedding│
         └────┬─────┘ └────┬─────┘ └────┬─────┘
              │             │             │
              ▼             ▼             ▼
    ┌─────────────────────────────────────────────┐
    │              Storage Layer                   │
    │                                              │
    │  Graph Store  ◀── Entities + Relations       │
    │  Vector Store ◀── Embedding Vectors          │
    │  KV Store     ◀── LLM Cache + Metadata       │
    └─────────────────────────────────────────────┘
```

### 2.2 查询检索流程 (Query)

```
                          ┌─────────────┐
                          │  用户问题     │
                          │  "糖尿病症状" │
                          └──────┬──────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Keyword Extraction    │
                    │   LLM (keyword role)    │
                    │   hl_keywords: 糖尿病    │
                    │   ll_keywords: 症状      │
                    └────────────┬───────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ▼                    ▼                    ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │  Local Graph │    │ Global Graph │    │  Naive Vector│
   │  Retrieval   │    │  Retrieval   │    │  Retrieval   │
   │              │    │              │    │              │
   │ 1. 向量搜实体  │    │ 1. 向量搜关系  │    │ 1. 向量搜文本块│
   │ 2. 遍历1-hop  │    │ 2. 获取全局主题│    │ 2. Top-K返回  │
   │   邻居实体+关系│    │ 3. 扩展上下文  │    │              │
   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
          │                   │                    │
          └───────────────────┼────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Round-Robin Merge     │
                 │   去重 + 截断 (token限额)│
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   (Optional) Rerank     │
                 │   bge-reranker-v2-m3   │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Context Assembly      │
                 │   entities + relations  │
                 │   + chunks → prompt     │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   LLM Generation        │
                 │   DeepSeek (query role) │
                 │   System Prompt:        │
                 │   "你是医疗知识助手..."  │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Final Answer          │
                 │   + Source References   │
                 └────────────────────────┘
```

### 2.3 MVP 已验证的数据流

```text
data/raw/medical_knowledge.txt  (5条医疗知识, 1375字符)
    │
    ▼
LightRAG.ainsert(text)
    │
    ├── Chunk: 2个文本块 (token_size=1200, overlap=100)
    ├── Extract: 38个实体 + 33条关系 (DeepSeek LLM)
    ├── Embed: bge-small-zh-v1.5 (512d) → 向量化
    └── Store: NetworkX (图) + NanoVectorDB (向量) + JSON (KV)
    │
    ▼
LightRAG.aquery("2型糖尿病有哪些常见症状?", mode=mix)
    │
    ├── Keywords: "糖尿病" + "症状" (LLM提取)
    ├── Local: 实体"2型糖尿病"向量匹配 → 1-hop邻居实体+关系
    ├── Global: 关系"症状"向量匹配 → 全局关系上下文
    ├── Naive: 块向量匹配 → top-2 chunks
    ├── Merge: round-robin → 去重 → 截断
    └── LLM: DeepSeek 生成回答 + 来源引用
    │
    ▼
"2型糖尿病的常见症状主要包括... (附References)"
```

### 2.4 Phase 7 Agent 工作流（已实现，src/agent.py MedQAAgent）

```
用户问题 → [safety_check] SafetyChecker + 分层安全 prompt
               │ 条件边 route_after_safety
               ├── BLOCKED → [decide] 5C ReliabilityDecision → [finalize] 拒答
               └── 正常 → [neo4j_retrieve] F5 mentions + _retrieve_neo4j（只读, 不可达降级）
                          → [build_prompt] F5 build_fusion_prompt（安全 prompt + 证据段）
                          → [generate] RAGEngine.query_with_context（唯一一次 LLM generation, 缓存开关）
                          → [analyze] detect_refusal / citation / B4 信号 / M8-M12（零 LLM）
                          → [decide] 5C ReliabilityDecision（终端门, 复用不重写）
                          → [finalize] F5 fuse_evidence + 展示层输出信封
```

约束（G1/G2 已验证）: 非 BLOCKED 恰 1 次 generation；BLOCKED 0 次 generation + 0 次 Neo4j 检索；
无 intent/rewrite/reflection/retry/judge LLM；不自动重检索。工作流实现见 `src/agent.py`。

### 2.5 Phase 8 Web UI 层（已实现，app/）

```
用户浏览器
  → app/streamlit_app.py（薄渲染层：chat + 分区展示 + KG expander）
      → MedQAAgent.run(question)（唯一 Agent 调用；持久事件循环驱动）
          → [内部] LightRAG + Neo4j + LLM + 5C 决策
  → final_response → app/ui_helpers.py（纯函数格式化）
      → 回答 / 安全 / 可靠性 / 证据与来源（overview·LightRAG·Neo4j·KG·引用·覆盖）/ 系统状态 / 调试
```

边界与关键设计：
- **UI 只调用 MedQAAgent**，不直接接触 LightRAG / Neo4j / LLM / ReliabilityDecision / Evidence Fusion。
- **Neo4j 四态**（`format_neo4j_evidence`）：not_checked（BLOCKED 未检索）> unavailable > hit > no_hit。
- **KG 可视化**（`build_kg_graph_data` + pyvis 0.3.2，**in_line 内联 vis.js，零 CDN**）：仅 hit 渲染，≤20/20，只消费 `sources` 含 `neo4j_graph` 的 evidence；networkx 确定性布局 + 关闭物理 + 延迟 fit（居中不漂移）。
- **持久事件循环**（`run_agent_sync`）：解决 Streamlit 同步脚本下 naive `asyncio.run` 的跨循环 bug。

### 2.6 Phase 9 多轮对话 Memory（已实现，方案 B）

```
用户第 N 轮提问 Q_N
  → app/streamlit_app.py
      memory = build_conversation_context(st.session_state["history"])   # 最近≤3轮合格Q/A，脱敏，≤600字符
      fr = run_agent_sync(agent, Q_N, conversation_context=memory)
          → MedQAAgent.run(Q_N, conversation_context=memory)
              → _init_state(Q_N)
              → ainvoke(input, config={"configurable": {"conversation_context": memory}})  # 不进 AgentState
                  safety_check(Q_N)          # 只当前问题
                  → (BLOCKED 短路 decide) / neo4j_retrieve(Q_N)      # 只当前问题
                  → build_prompt: base_safety_prompt + Neo4j证据段 + [<conversation_context> 背景段]
                  → generate(Q_N, system_prompt)   # 恰 1 次 generation
                  → analyze / decide / finalize     # 全当前轮
  → history 追加 {Q_N, final_response}（供展示 + 下一轮 memory）
```

边界与关键设计：
- **Memory = 对话背景，不是 evidence/citation/retrieval/state 事实**：唯一进入位置是 `_build_prompt` 在 generation 前追加的 system_prompt `<conversation_context>` 段；经 invoke config 传递，**不新增 AgentState 字段**。
- **当前轮为唯一事实源**：SafetyChecker / LightRAG / Neo4j / Evidence Fusion / Citations / Coverage / ReliabilityDecision 全部只针对当前问题。
- **过滤**（`_memory_turn_is_eligible`）：error / blocked / refusal 轮不进入 Memory。
- **SAFE 无证据 + memory**：`_MEMORY_SAFE_BASE` 打底并保留 `{context_data}` 占位符，保证当前轮向量检索仍进入 prompt。
- **隔离**：memory 来自 `session_state["history"]`（每 tab 隔离）；清空对话即清空 memory；无 thread_id / checkpointer / MemorySaver。

---

## 3. 模块关系图

```
┌─────────────────────────────────────────────────────────────┐
│                        app.py (Phase 8)                      │
│                      Streamlit Web UI                        │
│                    调用 agent.query()                         │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────┐
│                   src/agent.py (Phase 7)                      │
│                   LangGraph Workflow                          │
│                                                               │
│   StateGraph:                                                 │
│   user_input → classify_intent → route → retrieve → merge    │
│                                     │                        │
│                    ┌────────────────┼────────────┐           │
│                    ▼                ▼            ▼           │
│              Vector RAG      KG Retrieval   Hybrid(LightRAG) │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────┐
│                  src/rag_engine.py (Phase 2)                  │
│                      LightRAG Wrapper                         │
│                                                               │
│   class RAGEngine:                                            │
│     - __init__(working_dir, llm, embedder)                   │
│     - initialize()              # 初始化存储                  │
│     - insert_documents(texts)   # 索引文档                   │
│     - query(question, mode)     # 检索+生成                   │
│     - get_cache_stats()         # 缓存统计                   │
└──────────┬──────────────────────┬────────────────────────────┘
           │                      │
    ┌──────┼──────────┐    ┌──────┼──────────┐
    ▼      ▼          ▼    ▼      ▼          ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│embed │ │ llm  │ │data  │ │vector│ │ kg   │ │config│
│der.py│ │client│ │loader│ │store │ │build │ │.py   │
│      │ │.py   │ │.py   │ │.py   │ │er.py │ │      │
├──────┤ ├──────┤ ├──────┤ ├──────┤ ├──────┤ ├──────┤
│bge   │ │Deep  │ │PDF   │ │Chroma│ │Neo4j │ │.env  │
│small │ │Seek  │ │TXT   │ │DB    │ │Cypher│ │加载  │
│zh    │ │API   │ │Chunk │ │      │ │实体/ │ │配置  │
│512d  │ │OpenAI│ │      │ │      │ │关系  │ │常量  │
│      │ │兼容  │ │      │ │      │ │抽取  │ │      │
└──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
     │        │        │         │         │
     └────────┴────────┴─────────┴─────────┘
                      │
            外部服务 / 本地模型
     ┌────────────────┼────────────────┐
     ▼                ▼                ▼
┌──────────┐  ┌──────────────┐  ┌──────────┐
│ DeepSeek │  │ bge-small-zh │  │ Neo4j    │
│ API      │  │ (本地CPU)     │  │ (后续)    │
└──────────┘  └──────────────┘  └──────────┘
```

### 当前模块状态

| 模块 | 文件 | 状态 | 说明 |
|------|------|:--:|------|
| Config | `src/config.py` | ✅ | 统一配置管理 (dataclass + 单例) |
| Embedder | `src/embedder.py` | ✅ | bge-small-zh-v1.5, 512d |
| LLM Client | `src/llm_client.py` | ✅ | DeepSeek API (v4-flash) |
| Data Loader | `src/data_loader.py` | ✅ | 多格式 + YAML frontmatter + metadata |
| RAG Engine | `src/rag_engine.py` | ✅ | LightRAG 封装 (索引/查询/缓存) |
| Metadata | `src/metadata.py` | ✅ | 元数据注册表 + JSON/Cypher 导出 |
| Neo4j Utils | `src/neo4j_utils.py` | ✅ | Schema DDL + 节点/关系 Cypher 导出 |
| Medical Core | `src/medical_core/` | ✅ | 实体/关系/安全规则 |
| Vector Store | NanoVectorDB (默认) | ✅ | JSON 文件存储 |
| Graph Store | NetworkX (默认) | ✅ | GraphML 文件 (380节点 + 422边) |
| **数据管道** | **src/pipeline/** | ✅ | Adapter→Transformer→Validator→Writer |
| **EntityRegistry** | medical_core/entities.py | ✅ | 已填充 14 个领域实体 (Phase 4) |
| VectorStore 抽象 | `src/vector_store.py` | ✅ | 抽象接口 (Phase 4) |
| ChromaDB | — | ⚫ | 已放弃 (LightRAG 1.5.4 已弃用) |
| Neo4j 连接 | `src/neo4j_utils.py` | ✅ | F1: Neo4jConnection 实现 (connect/execute/事务/批量/异常) + get_neo4j_connection 工厂 |
| Neo4j Schema | `src/neo4j_utils.py` | ✅ | F1: bootstrap() 幂等 + verify_schema() (SHOW 校验) |
| Neo4j 容器 | `docker-compose.yml` | ✅ | F1: medqa-neo4j (neo4j:5.26-community, volume, 7474/7687, 认证 .env 注入) |
| **图数据导入** | `src/graph_import.py` + 2 scripts | ✅ | F2: graphml 原始层 380/422 + curated 临床层 9+5, 参数化 UNWIND 幂等, 占位符关系不导入 |
| **关系重建** | `src/relation_rebuild.py` + script | ✅ | F3: curated 临床关系 9 条 (TREATS 6 + HAS_COMPLICATION 3), D4 两端解析, 自环/占位符/unresolved 隔离, 幂等 |
| **图检索** | `src/neo4j_graph.py` | ✅ | F4: Neo4jGraphRetriever（6 API）, D4 解析 + 白名单 + 参数化 + 有界 path, 只读 |
| **证据融合** | `src/evidence_fusion.py` | ✅ | F5: 纯函数融合（D4 mentions / 跨源去重 / provenance / priority 排序 / 截断 / coverage 如实 / build_fusion_prompt）, 零 LLM/DB |
| **融合查询编排** | `scripts/run_graph_query.py` | ✅ | F5: 一次 LLM generation, Neo4j 不可达降级, 复用信号链路 + ReliabilityDecision |
| **端到端对比评估** | `scripts/eval_phase6.py` | ✅ | F6: 双模式 (baseline vs Neo4j), --fresh 修复缓存吞证据, M7 +0.0714, 33 单测 |
| **Agent** | `src/agent.py` | ✅ | Phase 7 G1: MedQAAgent 7 节点 StateGraph (safety_check→neo4j_retrieve→build_prompt→generate→analyze→decide→finalize), 条件边 route_after_safety, 一次 generation 约束, F5 融合 + 5C 决策层复用, F6 cache 运行时开关, 验证 6/6; **Phase 9 P2: 加法式 conversation_context（经 invoke config，不进 AgentState）+ _append_conversation_context**, tests 42 |
| **UI** | `app/streamlit_app.py` + `app/ui_helpers.py` | ✅ | Phase 8 P1-P6: 对话 + 证据展示 + KG 可视化（pyvis 0.3.2）+ Neo4j 四态; UI 只调 MedQAAgent, 持久事件循环; **Phase 9 P0: KG 居中（networkx 布局+延迟 fit+vis.js 内联）、心理危机热线 12356 展示层校正; Phase 9 P2: build_conversation_context 多轮 Memory; Phase 9 P4: 「参考前文」提示（memory_used_hint）、首轮加载提示（_has_queried）、示例跨实体题、侧边栏「关于本系统」**, tests 179 |
| **Memory** | `app/ui_helpers.py` `build_conversation_context` | ✅ | Phase 9 P2/P3: 多轮 Memory（≤3 轮/≤600 字符/error·blocked·refusal 跳过/脱敏）；仅 LLM prompt 背景，不进 Safety/检索/证据/决策/AgentState；验证通过；**P4: memory_used_hint 展示提示（实质回答才提示）** |
| **CBLUE 数据适配** | `src/data_adapter/cblue_adapter.py` | 🔬 **实验模块（未接入生产链路）** | Phase 10-A: 独立包解析/转换/校验/Entity Linking 候选（CBLUEEntity/CBLUERelation，provenance 强制）；`scripts/cblue_prepare.py`/`cblue_validate.py`；三层隔离（external/raw/processed/cblue + 独立 working_dir）；**不触碰生产 Neo4j/entities.json/Agent/evaluation 基线**；tests 34 |
| **Reranker** | `src/retrieval/rerank.py` | 🔬 **实验评估（未接入生产链路）** | Phase 10-B: bge-reranker-base cross-encoder（CPU）+ fallback；`scripts/reranker_benchmark.py`（M1/M2/MRR/latency，零 LLM）；实验显示当前语料下每查询恰好 14 chunks，rerank 无增删空间 → 暂不接入生产；tests 16 |
| **观测模块** | `src/monitoring.py` | 🔬 **独立观测（不进入生产核心）** | Phase 10-C: request_id + JSONL（`data/logs/medqa.jsonl`）+ metrics 白名单 + sanitize/禁止清单 + 日志轮转（DEFAULT 5MB/3 备份，`_rotate_if_needed`，失败静默）；经 `app/ui_helpers.run_agent_sync` 接入（final_response 原样返回）；异常静默降级；**C3: UI 多轮+Memory 日志补验 + 轮转**；tests 24 |
| 评估指标 | `evaluation/metrics.py` | ✅ | A3-A5: M1-M7 + LLM judge (Phase 5A); B5: M8-M12 (Phase 5B) |
| 评估执行器 | `evaluation/runner.py` | ✅ | A4-A5: 单模式评估 (28 题); **C3: Step E5 决策层集成 + --no-decision-layer (Phase 5C)** |
| 模式对比器 | `evaluation/mode_comparator.py` | ✅ | A6: 4 模式 × 28 题对比 |
| 评估 CLI | `scripts/eval_retrieval.py` | ✅ | A7: --full / --compare-modes / --refusal-only / --mode |
| Phase 5B 报告 | `scripts/eval_phase5b.py` | ✅ | B6: 完整评估 + phase5b_report.json (Phase 5B) |
| 引用提取器 | `evaluation/citation_extractor.py` | ✅ | B3: 结构化引用提取 (Phase 5B) |
| 引用验证器 | `evaluation/citation_verifier.py` | ✅ | B3: 三级追溯验证 (Phase 5B) |
| 不确定性信号 | `evaluation/uncertainty_signals.py` | ✅ | B4: 11 确定性信号提取 (Phase 5B, 供 5C 决策) |
| **可靠性决策** | `evaluation/reliability_decision.py` | ✅ | C1: R1-R9 决策层 (allow/revise/refuse), 纯函数零 LLM (Phase 5C) |
| 决策层单测 | `evaluation/test_reliability_decision.py` | ✅ | C2: 36 单测, 28 题回归 M7=0.9286 (Phase 5C) |
| Phase 5C 报告 | `scripts/eval_phase5c.py` | ✅ | C4: 完整评估 + phase5c_report.json (含 decision_layer_summary, Phase 5C) |

---

## 4. 为什么选择 LightRAG

### 4.1 决策背景

项目进行了 6 个候选项目的全面调研：

| 候选项目 | 定位 | 匹配度 |
|----------|------|:--:|
| **LightRAG** | 通用 Graph+Vector RAG 框架 | **8/10** |
| neo4j-graphrag-python | Neo4j 官方 GraphRAG SDK | 7/10 |
| QASystemOnMedicalKG | 中文医疗 KG 问答系统 | 6/10 |
| Neo4j LLM Graph Builder | 全栈 KG 构建应用 | 5/10 |
| Medical-Graph-RAG | 医学 GraphRAG (ACL 2025) | 4/10 |
| Microsoft GraphRAG | 端到端 GraphRAG | 4/10 |

### 4.2 选择 LightRAG 的核心原因

```
1. 检索内核最成熟
   ├── 5 种检索模式: naive / local / global / hybrid / mix
   ├── Graph + Vector 原生混合 (不是后拼接)
   └── round-robin 融合 + 去重 + token 截断

2. 技术栈完美匹配
   ├── Neo4j 后端: ✅ (支持, 后续切换)
   ├── 向量后端: NanoVectorDB (长期保留；不迁移 ChromaDB)
   ├── bge-m3 Embedding: ✅ (原生支持)
   ├── DeepSeek LLM: ✅ (OpenAI 兼容接口)
   └── Windows 兼容: ✅ (pip install 即可)

3. 工程优势
   ├── 增量更新: 添加文档不重建索引
   ├── LLM 缓存: 相同 prompt 不重复调用 (省钱)
   ├── 模块化: 存储 / LLM / Embedding 均可独立替换
   └── 资源友好: 4GB RAM + CPU 可运行

4. 社区活跃
   ├── 36,000+ GitHub Stars
   ├── MIT License (无商用限制)
   ├── 持续更新 (2026年仍在活跃开发)
   └── 完善的文档和示例

5. MVP 适配
   ├── 默认文件存储 (零配置启动)
   ├── NetworkX + NanoVectorDB (无需额外安装数据库)
   ├── 图迁移到 Neo4j；向量长期使用 NanoVectorDB
   └── pip install lightrag-hku 一行安装
```

### 4.3 LightRAG 在本项目中的角色

```
LightRAG = 检索内核 (不做 Agent, 不做 UI)

本项目:
  ├── LightRAG 负责: 文档索引 + 实体抽取 + 关系抽取 + 5种检索模式 + LLM调用
  ├── 自建 LangGraph 负责: Agent 工作流编排 (意图判断 → 路由)
  └── 自建 Streamlit 负责: Web UI (对话界面 + 引用展示)
```

---

## 5. 为什么不用 Microsoft GraphRAG

### 5.1 对比分析

| 维度 | LightRAG | Microsoft GraphRAG | 本项目偏好 |
|------|----------|-------------------|:--:|
| **索引速度** | 快 (无需社区检测) | 慢 (Leiden 社区检测) | LightRAG |
| **API 调用量** | 低 (每文档 ~10次) | 极高 (数百次/文档) | LightRAG |
| **Token 消耗** | 低 | $6-7 索引一本3万字书 | LightRAG |
| **增量更新** | ✅ 支持 | ❌ 需全量重建 | LightRAG |
| **存储后端** | 10+ 后端可选 | 基于文件 (Parquet/CSV) | LightRAG |
| **部署复杂度** | pip install | 复杂配置 | LightRAG |
| **硬件要求** | CPU 可运行 | GPU 推荐 | LightRAG |
| **Windows兼容** | ✅ | ⚠️ | LightRAG |
| **中文支持** | ✅ | ⚠️ 英文优化 | LightRAG |
| **社区摘要** | ❌ 不生成 | ✅ Leiden + 层次化摘要 | GraphRAG |
| **全局推理** | 中等 (global mode) | 强 (community reports) | 本项目够用 |

### 5.2 MS GraphRAG 不适合本项目的具体原因

```
1. 资源消耗与硬件不匹配
   ├── 用户硬件: RTX 5060 8GB + 16GB RAM
   ├── GraphRAG 需求: 强烈推荐 GPU + 大内存
   └── 索引 32K 字书籍需 $6-7 token 费用 (GPT-4o)
       → 构建医疗知识库将产生巨大 API 开销

2. 更新机制不适合迭代开发
   ├── GraphRAG: 全量重建 (新数据 → 重新跑整个 pipeline)
   ├── LightRAG: 增量更新 (新文档 → 合并到现有图)
   └── MVP 阶段需频繁添加/调整医疗数据 → 全量重建不可接受

3. 部署复杂度影响快速验证
   ├── GraphRAG: yaml 配置 → 多阶段 pipeline → CLI 运行
   ├── LightRAG: pip install → Python API → 3行代码运行
   └── 本项目目标"先跑通再完善" → LightRAG 明显更适合

4. 中文医疗场景未优化
   ├── GraphRAG: 英文实体抽取优化, 中文效果未经验证
   ├── LightRAG: LLM 驱动的抽取, 中英文均可
   └── 已有 bge-small-zh-v1.5 中文 Embedding 配合

5. 存储架构限制
   ├── GraphRAG: 输出为 Parquet/CSV 文件 (不是图数据库)
   ├── LightRAG: 支持 Neo4j 原生图数据库
   └── 本项目需 Neo4j 做知识图谱可视化 → LightRAG 直接支持
```

### 5.3 但 MS GraphRAG 并非被完全放弃

```
MS GraphRAG 的社区检测和层次化摘要思路值得借鉴:

- "Global Search" 的 Map-Reduce 模式
  → 后续可在 Neo4j 中实现社区检测 (GDS 库)
  → 不通过 GraphRAG 框架，而是直接在图数据库中实现

- "DRIFT Search" 的迭代精炼思路
  → 可参考用于 LangGraph Agent 的多轮检索设计

结论: GraphRAG 的思想可以参考, 但框架本身不适合作为底座
```

---

## 6. v1 架构演进路线

```
Phase 1-3 (当前)               Phase 4-5                    Phase 7-8
    MVP + 工程化                  生产存储                     完整系统

┌──────────────┐         ┌──────────────┐           ┌──────────────┐
│  LightRAG    │         │  LightRAG    │           │  LightRAG    │
│  + NetworkX  │   →     │  + Neo4j     │    →      │  + Neo4j     │
│  + NanoVDB   │         │  + NanoVDB   │           │  + NanoVDB   │
│  + src/*.py  │         │  + src/*.py  │           │  + LangGraph │
│  + metadata  │         │  + metadata  │           │  + Streamlit │
└──────────────┘         └──────────────┘           └──────────────┘
 模块化 + 工程化           模块化 + 持久化              完整应用系统
```

| 阶段 | 图存储 | 向量存储 | Pipeline | Agent | UI | 代码结构 |
|------|--------|----------|----------|-------|----|----------|
| Phase 1-3 (MVP) | NetworkX | NanoVectorDB | ❌ | ❌ | ❌ | 模块化 src/ |
| Phase 4 | NetworkX | NanoVectorDB | ✅ Pipeline | ❌ | ❌ | + src/pipeline/ |
| Phase 5-6 | NetworkX→Neo4j | NanoVectorDB | ✅ Pipeline | ❌ | ❌ | + 检索评估 |
| Phase 7 | Neo4j | NanoVectorDB | ✅ Pipeline | ✅ LangGraph | ❌ | + Agent 工作流 (src/agent.py) |
| Phase 8 | Neo4j | NanoVectorDB | ✅ Pipeline | ✅ LangGraph | ✅ Streamlit | + Web UI (app/) + KG 可视化 |
| Phase 9 | Neo4j | NanoVectorDB | ✅ Pipeline | ✅ LangGraph | ✅ Streamlit | + 多轮对话 Memory + KG 可视化增强（内联 vis.js、确定性布局居中）+ UI 打磨（「参考前文」提示/首轮加载/关于本系统） |
| Phase 10 / v2 | Neo4j | NanoVectorDB | ✅ Pipeline | ✅ LangGraph | ✅ Next.js + FastAPI | + SSE + MCP/工具轨迹 + 语音 + 本地 OCR + CI |

---

## 7. 关键设计决策

| 决策 | 说明 |
|------|------|
| **底座选型** | LightRAG — 检索内核最成熟、增量更新、资源友好 |
| **存储渐进式** | 图存储迁移到 Neo4j；向量长期保留 NanoVectorDB（ChromaDB 路线已放弃） |
| **LLM 云端优先** | DeepSeek API (v4-flash) — 不占本地资源、中文好、便宜 |
| **Embedding 本地优先** | bge-small-zh-v1.5 — 94MB、512d、CPU可运行 |
| **Agent 自建** | LangGraph — 任何框架都缺Agent，统一自建方案 |
| **UI 自建** | v1 Streamlit 保留；v2 使用 Next.js + TypeScript + FastAPI SSE |
| **数据格式标准化** | Markdown + YAML frontmatter — 人机可读、元数据可追溯 |
| **Neo4j 离线导出** | 先 Cypher 后连接 — 降低环境复杂度 |
| **先跑通再优化** | 每阶段产出可运行代码，不过度工程化 |

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-07-25 | v0.1 — 初始架构文档（MVP，LightRAG 底座） |
| 2026-07-25 | v0.3 — 数据工程化层（`data/` 结构化 + 元数据追溯） |
| 2026-07-26 | v0.4 — 数据管道 Pipeline 层（Adapter→Transformer→Validator→Writer），移除 ChromaDB 路线 |
| 2026-08-12 | v0.5 — 评估框架模块（`evaluation/runner.py` / `mode_comparator.py` / `eval_retrieval.py`） |
| 2026-08-12 | v0.6 — 安全拦截层：SafetyChecker 接入 query flow，分层 system_prompt（BLOCKED / CAUTION+RESTRICTED / SAFE） |
| 2026-08-12 | v0.7 — 引用分析：citation_extractor（结构化提取）+ citation_verifier（三级追溯） |
| 2026-08-12 | v0.8 — 不确定性信号：uncertainty_signals（11 个确定性信号） |
| 2026-08-12 | v0.9 — 新指标 M8-M12：uncertainty_score / citation_quality / safety_signal_presence / combined_reliability |
| 2026-08-13 | v1.0 — Phase 5B 完整评估 |
| 2026-08-13 | v1.1 — 可靠性决策层 reliability_decision.py（R1-R9）+ 36 单测，runner Step E5 集成（28 题 M7=0.9286 ON / 0.8214 OFF） |
| 2026-08-13 | v1.2 — Phase 5C 完整评估（28/28，M7=0.9286，decision 分布 refuse=2 / revise=7 / allow=19） |
| 2026-08-16 | v1.3 — Neo4j Community 5.26 容器（docker-compose，认证 .env 注入）+ Neo4jConnection + Schema bootstrap（幂等） |
| 2026-08-16 | v1.4 — 图数据导入（graphml 原始层 + curated 临床层，参数化 UNWIND 幂等） |
| 2026-08-16 | v1.5 — curated 临床关系重建（TREATS / HAS_COMPLICATION，D4 两端解析，幂等） |
| 2026-08-16 | v1.6 — 图检索器 Neo4jGraphRetriever（6 API，D4 解析 + 白名单 + 参数化 + 有界 path，DB 只读） |
| 2026-08-16 | v1.7 — 证据融合 evidence_fusion.py（纯函数）+ run_graph_query 编排（一次 LLM generation） |
| 2026-08-17 | v1.8 — 端到端对比评估（双模式 baseline vs Neo4j，M7 +0.0714） |
| 2026-08-17 | v1.9 — LangGraph Agent 工作流 src/agent.py（MedQAAgent 7 节点 StateGraph，验证 6/6） |
| 2026-08-17 | v2.0 — Streamlit Web UI（对话 / 证据展示 / KG 可视化）+ 持久事件循环修复 |
| 2026-08-22 | v2.1 — KG 图谱居中（networkx 确定性布局 + vis.js 内联零 CDN）+ 心理危机热线展示校正 + 多轮对话 Memory（UI history → conversation_context → prompt 背景，经 config 不进 AgentState） |
| 2026-08-25 | v2.2 — UI 打磨：「参考前文」提示、首轮加载提示（10-40s 冷启动）、示例跨实体题、侧边栏「关于本系统」 |
| 2026-08-25 | v2.3 — CBLUE 独立数据适配管道（src/data_adapter，三层隔离，Entity Linking 候选；不集成生产链路） |
| 2026-08-25 | v2.4 — Reranker 实验评估（src/retrieval/rerank.py，实验阶段未接入生产）+ 日志监控模块（request_id + JSONL + metrics 白名单 + 脱敏） |
| 2026-08-26 | v2.5 — 日志轮转（src/monitoring.py `_rotate_if_needed`，5MB/3 备份，失败静默）+ UI 多轮 + Memory 日志补验 |
| 2026-10-19 | v2.6 — Next.js 16 证据工作台 + FastAPI SSE + MCP 工具/轨迹 + 语音 + 本地 OCR + CI |
