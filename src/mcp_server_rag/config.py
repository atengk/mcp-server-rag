"""
运行时配置管理模块。

解析并校验所有以 MCP_RAG_ 为前缀的环境变量，提供强类型配置访问与安全缺省值回退。

@author Ateng
@since 2026-11-11
"""

import os
from pathlib import Path
from urllib.parse import quote

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
    metadata_store: str = Field(
        default_factory=lambda: (
            os.getenv("MCP_RAG_METADATA_STORE", "sqlite").strip().lower() or "sqlite"
        )
    )
    database_url: str = Field(default_factory=lambda: os.getenv("MCP_RAG_DATABASE_URL", "").strip())
    db_host: str = Field(default_factory=lambda: os.getenv("MCP_RAG_DB_HOST", "").strip())
    db_port: int | None = Field(
        default_factory=lambda: _parse_optional_int_env("MCP_RAG_DB_PORT", min_val=1)
    )
    db_user: str = Field(default_factory=lambda: os.getenv("MCP_RAG_DB_USER", "").strip())
    db_password: str = Field(default_factory=lambda: os.getenv("MCP_RAG_DB_PASSWORD", "").strip())
    db_name: str = Field(default_factory=lambda: os.getenv("MCP_RAG_DB_NAME", "").strip())
    storage_type: str = Field(
        default_factory=lambda: (
            os.getenv("MCP_RAG_STORAGE_TYPE", "local").strip().lower() or "local"
        )
    )
    s3_bucket: str = Field(default_factory=lambda: os.getenv("MCP_RAG_S3_BUCKET", "").strip())
    s3_endpoint: str = Field(default_factory=lambda: os.getenv("MCP_RAG_S3_ENDPOINT", "").strip())
    s3_region: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_S3_REGION", "us-east-1").strip() or "us-east-1"
    )
    s3_access_key: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_S3_ACCESS_KEY", "").strip()
    )
    s3_secret_key: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_S3_SECRET_KEY", "").strip()
    )
    s3_prefix: str = Field(default_factory=lambda: os.getenv("MCP_RAG_S3_PREFIX", "assets").strip())
    s3_public_base_url: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_S3_PUBLIC_BASE_URL", "").strip()
    )
    s3_timeout_seconds: int = Field(
        default_factory=lambda: _parse_int_env("MCP_RAG_S3_TIMEOUT_SECONDS", 30, min_val=1)
    )
    archive_remote_images: bool = Field(
        default_factory=lambda: _parse_bool_env("MCP_RAG_ARCHIVE_REMOTE_IMAGES", False)
    )
    remote_image_timeout_seconds: int = Field(
        default_factory=lambda: _parse_int_env(
            "MCP_RAG_REMOTE_IMAGE_TIMEOUT_SECONDS", 15, min_val=1
        )
    )
    remote_image_max_bytes: int = Field(
        default_factory=lambda: _parse_int_env(
            "MCP_RAG_REMOTE_IMAGE_MAX_BYTES", 10 * 1024 * 1024, min_val=1024
        )
    )
    rerank_enabled: bool = Field(
        default_factory=lambda: _parse_bool_env("MCP_RAG_RERANK_ENABLED", False)
    )
    rerank_provider: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_RERANK_PROVIDER", "fastembed").strip().lower()
    )
    rerank_model: str = Field(
        default_factory=lambda: (
            (os.getenv("MCP_RAG_RERANK_MODEL") or "").strip() or "BAAI/bge-reranker-base"
        )
    )
    rerank_base_url: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_RERANK_BASE_URL", "").strip()
    )
    rerank_api_key: str = Field(
        default_factory=lambda: os.getenv("MCP_RAG_RERANK_API_KEY", "").strip()
    )
    rerank_timeout_seconds: int = Field(
        default_factory=lambda: _parse_int_env("MCP_RAG_RERANK_TIMEOUT_SECONDS", 30, min_val=1)
    )

    def resolved_database_url(self) -> str:
        """解析最终生效的数据库连接 DSN。

        优先使用显式设置的 ``MCP_RAG_DATABASE_URL``；否则在指定了 ``MCP_RAG_DB_HOST`` 等拆分变量时，
        按当前 ``metadata_store`` 方言拼接标准化连接串（对用户名与密码执行 URL 编码）。

        Returns:
            数据库连接 URL 字符串；若未配置外部连接信息且使用内置 SQLite，则返回默认本地路径连接串。
        """
        if self.database_url:
            return self.database_url

        if self.metadata_store == "sqlite":
            if not self.db_name:
                return f"sqlite:///{(self.data_dir / 'metadata.db').as_posix()}"
            if self.db_name.startswith("sqlite:"):
                return self.db_name
            db_candidate = Path(self.db_name)
            resolved_db = (
                db_candidate if db_candidate.is_absolute() else (self.data_dir / db_candidate)
            )
            return f"sqlite:///{resolved_db.as_posix()}"

        if not self.db_host:
            return ""

        scheme_defaults: dict[str, tuple[str, int]] = {
            "postgresql": ("postgresql", 5432),
            "mysql": ("mysql", 3306),
        }
        scheme, default_port = scheme_defaults.get(self.metadata_store, (self.metadata_store, 5432))
        port = self.db_port if self.db_port is not None else default_port

        auth = ""
        if self.db_user:
            encoded_user = quote(self.db_user, safe="")
            if self.db_password:
                encoded_password = quote(self.db_password, safe="")
                auth = f"{encoded_user}:{encoded_password}@"
            else:
                auth = f"{encoded_user}@"

        db_name_part = f"/{quote(self.db_name, safe='')}" if self.db_name else ""
        return f"{scheme}://{auth}{self.db_host}:{port}{db_name_part}"


def get_settings() -> Settings:
    """获取当前运行时配置实例。"""
    return Settings()
