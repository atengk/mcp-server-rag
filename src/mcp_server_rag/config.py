"""
运行时配置管理模块。

解析并校验所有以 MCP_RAG_ 为前缀的环境变量，提供强类型配置访问。

@author Ateng
@since 2026-11-11
"""

import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """MCP Server RAG 全局运行时配置模型。"""

    data_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("MCP_RAG_DATA_DIR", Path.cwd() / ".rag_data")
        ).resolve()
    )
    vector_store: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_VECTOR_STORE", "chromadb").lower()
    )
    embedding_provider: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_EMBEDDING_PROVIDER", "fastembed").lower()
    )
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    read_only: bool = Field(
        default_factory=lambda: (
            os.getenv("MCP_RAG_READ_ONLY", "false").lower() in ("1", "true", "yes", "on")
        )
    )
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("MCP_RAG_CHUNK_SIZE", "600")))
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("MCP_RAG_CHUNK_OVERLAP", "80"))
    )
    max_files_batch: int = Field(
        default_factory=lambda: int(os.getenv("MCP_RAG_MAX_FILES_BATCH", "200"))
    )
    milvus_uri: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_MILVUS_URI", "http://localhost:19530")
    )
    milvus_token: str = Field(default_factory=lambda: os.getenv("MCP_RAG_MILVUS_TOKEN", ""))


def get_settings() -> Settings:
    """获取当前运行时配置实例。"""
    return Settings()
