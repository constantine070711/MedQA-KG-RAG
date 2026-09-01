"""
统一配置管理模块

职责:
    - 加载 .env 环境变量
    - 提供类型化的配置访问
    - 管理所有路径常量和模型参数

用法:
    from src.config import get_config
    cfg = get_config()
    print(cfg.deepseek_api_key)
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv


# ============================================================
# 项目根目录检测
# ============================================================

def _get_project_root() -> Path:
    """自动检测项目根目录（查找 .env 或 CLAUDE.md）"""
    # 优先使用当前文件向上推算
    current = Path(__file__).resolve().parent.parent

    # 验证：检查是否存在标志文件
    markers = [".env", "CLAUDE.md", "requirements.txt"]
    for marker in markers:
        if (current / marker).exists():
            return current

    # 回退：从环境变量或 CWD 推断
    env_root = os.getenv("PROJECT_ROOT", "")
    if env_root:
        return Path(env_root)

    return Path.cwd()


PROJECT_ROOT = _get_project_root()


# ============================================================
# 配置数据类
# ============================================================

@dataclass
class Config:
    """项目全局配置"""

    # --- 路径 ---
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "lightrag_cache")
    src_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "src")

    # --- DeepSeek API ---
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # --- Embedding 模型 ---
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_local_path: str = ""
    embedding_dim: int = 512  # bge-small-zh-v1.5 输出维度
    embedding_max_token_size: int = 8192

    # --- Neo4j（后续） ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # --- 应用 ---
    app_host: str = "127.0.0.1"
    app_port: int = 8501
    log_level: str = "INFO"

    # --- LightRAG 参数 ---
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    llm_model_max_token_size: int = 32768

    def __post_init__(self):
        """自动推导路径"""
        if not self.embedding_local_path:
            local_path = self.models_dir / "bge-small-zh-v1.5"
            if local_path.exists():
                self.embedding_local_path = str(local_path)
            else:
                self.embedding_local_path = self.embedding_model_name

    def validate(self) -> list[str]:
        """
        验证配置完整性，返回错误列表。
        返回空列表表示配置有效。
        """
        errors = []

        # API Key 检查
        if not self.deepseek_api_key or "your-deepseek" in self.deepseek_api_key:
            errors.append(
                "DEEPSEEK_API_KEY 未配置。请编辑 .env 文件，"
                "从 https://platform.deepseek.com 获取 API Key"
            )

        # 数据目录检查
        if not self.data_dir.exists():
            errors.append(f"数据目录不存在: {self.data_dir}")

        # Embedding 模型检查（允许在线加载，只做路径检查）
        if self.embedding_local_path != self.embedding_model_name:
            if not Path(self.embedding_local_path).exists():
                errors.append(
                    f"本地 Embedding 模型路径不存在: {self.embedding_local_path}"
                )

        return errors

    def summary(self) -> str:
        """返回配置摘要（不包含敏感信息）"""
        embed_source = (
            "本地预下载" if self.embedding_local_path != self.embedding_model_name
            else "在线加载"
        )
        return (
            f"Config:\n"
            f"  项目根目录: {self.project_root}\n"
            f"  数据目录:   {self.data_dir}\n"
            f"  缓存目录:   {self.cache_dir}\n"
            f"  LLM:        {self.deepseek_model} @ {self.deepseek_base_url}\n"
            f"  Embedding:  {self.embedding_model_name} ({self.embedding_dim}d, {embed_source})\n"
            f"  Neo4j:      {self.neo4j_uri} ({'已配置' if self.neo4j_password else '待配置'})\n"
        )


# ============================================================
# 单例工厂
# ============================================================

_config_instance: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    获取全局配置单例。

    Args:
        reload: 是否强制重新加载 .env 和重建 Config

    Returns:
        Config 实例
    """
    global _config_instance

    if _config_instance is not None and not reload:
        return _config_instance

    # 加载 .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
    else:
        # 尝试 .env.example 作为回退（仅用于结构参考）
        pass

    # 构建配置
    _config_instance = Config(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=int(os.getenv("APP_PORT", "8501")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    return _config_instance


# ============================================================
# 便捷常量（兼容旧代码）
# ============================================================

def get_working_dir() -> str:
    """获取 LightRAG 工作目录"""
    return str(get_config().cache_dir)


def get_embedding_config() -> dict:
    """获取 Embedding 模型配置"""
    cfg = get_config()
    return {
        "model_name": cfg.embedding_model_name,
        "local_path": cfg.embedding_local_path,
        "dim": cfg.embedding_dim,
    }


def get_llm_config() -> dict:
    """获取 LLM API 配置"""
    cfg = get_config()
    return {
        "api_key": cfg.deepseek_api_key,
        "base_url": cfg.deepseek_base_url,
        "model": cfg.deepseek_model,
    }
