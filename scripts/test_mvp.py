"""
MVP 最小闭环验证脚本
文档输入 -> LightRAG索引 -> 用户问题 -> LLM回答

用法：
    cd MedQA-GitHub
    venv\Scripts\activate
    python scripts/test_mvp.py
"""

import os
import sys
import io
import asyncio

# 强制 UTF-8 输出（Windows GBK 兼容）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 配置（先于所有其他导入）
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

# 验证 API Key
api_key = os.getenv("DEEPSEEK_API_KEY", "")
if not api_key or "your-deepseek" in api_key:
    print("[ERROR] 请先在 .env 文件中配置 DEEPSEEK_API_KEY")
    print("   获取地址：https://platform.deepseek.com")
    sys.exit(1)

base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Embedding 模型配置（MVP 使用轻量中文模型 bge-small-zh-v1.5）
# 优先从本地 models/ 加载（已预下载），否则从 HuggingFace 下载
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBED_LOCAL_PATH = str(project_root / "models" / "bge-small-zh-v1.5")
EMBED_DIM = 512  # bge-small-zh-v1.5 输出维度

import os as _os
if _os.path.exists(EMBED_LOCAL_PATH):
    EMBED_MODEL_PATH = EMBED_LOCAL_PATH
    print(f"[CONFIG] Embedding: {EMBED_MODEL_NAME} (本地预下载, {EMBED_DIM}d)")
else:
    EMBED_MODEL_PATH = EMBED_MODEL_NAME
    print(f"[CONFIG] Embedding: {EMBED_MODEL_NAME} (在线加载, {EMBED_DIM}d)")

print(f"[CONFIG] 配置信息:")
print(f"   LLM: {model_name}")
print(f"   API: {base_url}")
print(f"   Embedding: {EMBED_MODEL_PATH}")
print()

# ============================================================
# Step 1: 初始化 Embedding 函数（本地轻量中文模型）
# ============================================================
print(f"[LOADING] 加载 Embedding 模型: {EMBED_MODEL_PATH} ...")
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(EMBED_MODEL_PATH)

async def embedding_func(texts: list[str]):
    """将文本列表转换为向量数组（返回 numpy array，LightRAG 需要 .size 属性）"""
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings  # 直接返回 numpy array，不要 .tolist()

print(f"   [OK] Embedding 模型加载成功 (维度: {EMBED_DIM})")
print()

# ============================================================
# Step 2: 初始化 DeepSeek LLM 函数
# ============================================================
from lightrag.llm.openai import openai_complete_if_cache

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """调用 DeepSeek API 生成回复"""
    return await openai_complete_if_cache(
        model=model_name,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

WORKING_DIR = str(project_root / "lightrag_cache")


# ============================================================
# 主流程
# ============================================================
async def main():
    # --- Step 3: 初始化 LightRAG ---
    print("[LOADING] 初始化 LightRAG...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
            func=embedding_func,
        ),
        llm_model_func=llm_model_func,
    )
    await rag.initialize_storages()
    print("   [OK] LightRAG 初始化完成")
    print(f"   存储目录: {WORKING_DIR}")
    print()

    # --- Step 4: 加载并插入医疗知识文档 ---
    data_dir = project_root / "data" / "raw"
    txt_files = list(data_dir.glob("*.txt"))

    if not txt_files:
        print("[ERROR] data/raw/ 目录下没有 .txt 文件")
        return

    print(f"[FILE] 找到 {len(txt_files)} 个文档文件")
    for file_path in txt_files:
        print(f"   [LOADING] 处理: {file_path.name}")
        text = file_path.read_text(encoding="utf-8")
        await rag.ainsert(text)
        print(f"   [OK] 已索引: {file_path.name} ({len(text)} 字符)")
    print()

    # --- Step 5: 测试问答 ---
    questions = [
        "2型糖尿病有哪些常见症状？",
        "高血压的治疗方法是什么？",
        "二甲双胍有什么不良反应？",
        "感冒了应该怎么治疗？",
    ]

    print("=" * 60)
    print("[TEST] 开始测试问答 (使用 mix 模式：Graph+Vector混合)")
    print("=" * 60)

    for i, q in enumerate(questions, 1):
        print(f"\n{'-' * 50}")
        print(f"[QUESTION] 问题 {i}: {q}")
        print(f"{'-' * 50}")
        result = await rag.aquery(q, param=QueryParam(mode="mix"))
        print(f"[ANSWER] 回答: {result}")
        print()

    print("\n[OK] MVP 验证完成！")
    print(f"   LightRAG 缓存目录: {WORKING_DIR}")
    print("   下次运行将复用缓存，如需重建请删除该目录")

if __name__ == "__main__":
    asyncio.run(main())
