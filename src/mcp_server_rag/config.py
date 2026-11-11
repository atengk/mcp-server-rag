"""
运行时配置管理模块。

解析并校验所有以 MCP_RAG_ 为前缀的环境变量，提供强类型配置访问与安全缺省值回退。

@author Ateng
@since 2026-11-11
"""

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _parse_int_env(key: str, default: int, min_val: int = 1) -> int:
    """安全解析整型环境变量，若未提供或格式非法则安全回退到默认值。"""
    val = _parse_optional_int_env(key, min_val=min_val)
    return val if val is not None else default


def _parse_optional_int_env(key: str, min_val: int = 1) -> int | None:
    """安全解析可选整型环境变量，未设置或非法时返回 None。"""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return None
    try:
        val = int(raw.strip())
        return val if val >= min_val else None
    except ValueError:
        return None


def _parse_bool_env(key: str, default: bool = False) -> bool:
    """安全解析布尔型环境变量，支持 1/true/yes/on 等常见真值。"""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_path_env(key: str, default: Path) -> Path:
    """安全解析文件路径环境变量并自动解析为绝对路径。"""
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default.resolve()
    return Path(raw.strip()).resolve()


class Settings(BaseModel):
    """MCP Server RAG 全局运行时配置模型。"""

    data_dir: Path = Field(
        default_factory=lambda: _parse_path_env("MCP_RAG_DATA_DIR", Path.cwd() / ".rag_data")
    )
    vector_store: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_VECTOR_STORE", "chromadb").strip().lower()
    )
    embedding_provider: str = Field(
        default_factory=lambda: (
            os.getenv("MCP_RAG_EMBEDDING_PROVIDER", "fastembed").strip().lower() or "fastembed"
        )
    )
    embedding_base_url: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_EMBEDDING_BASE_URL", "").strip()
    )
    embedding_api_key: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_EMBEDDING_API_KEY", "").strip()
    )
    embedding_model: str = Field(
        default_factory=lambda: (
            os.getenv("MCP_RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5").strip()
            or "BAAI/bge-small-zh-v1.5"
        )
    )
    embedding_dimension: int | None = Field(
        default_factory=lambda: _parse_optional_int_env("MCP_RAG_EMBEDDING_DIMENSION", min_val=1)
    )
    read_only: bool = Field(default_factory=lambda: _parse_bool_env("MCP_RAG_READ_ONLY", False))
    chunk_size: int = Field(
        default_factory=lambda: _parse_int_env("MCP_RAG_CHUNK_SIZE", 600, min_val=1)
    )
    chunk_overlap: int = Field(
        default_factory=lambda: _parse_int_env("MCP_RAG_CHUNK_OVERLAP", 80, min_val=0)
    )
    max_files_batch: int = Field(
        default_factory=lambda: _parse_int_env("MCP_RAG_MAX_FILES_BATCH", 200, min_val=1)
    )
    milvus_uri: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_MILVUS_URI", "http://localhost:19530").strip()
    )
    milvus_token: str = Field(default_factory=lambda: os.getenv("MCP_RAG_MILVUS_TOKEN", "").strip())


def get_settings() -> Settings:
    """获取当前运行时配置实例。"""
    return Settings()
