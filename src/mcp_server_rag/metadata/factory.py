"""
元数据目录适配器工厂。

根据运行时配置实例化对应的元数据目录后端引擎（当前默认实现内置 SQLite，
并为 PostgreSQL/MySQL 预留后端路由分支与清晰配置校验）。

@author Ateng
@since 2026-11-11
"""

from pathlib import Path
from urllib.parse import unquote, urlparse

from mcp_server_rag.config import Settings
from mcp_server_rag.metadata.base import MetadataCatalogAdapter
from mcp_server_rag.metadata.mysql import MySQLMetadataCatalog
from mcp_server_rag.metadata.postgres import PostgresMetadataCatalog
from mcp_server_rag.metadata.sqlite import SQLiteMetadataCatalog

SUPPORTED_METADATA_STORES: frozenset[str] = frozenset({"sqlite", "postgresql", "mysql"})


def _resolve_sqlite_db_path(settings: Settings) -> Path | None:
    """从配置中解析自定义 SQLite 文件路径，未显式覆盖时返回 ``None``。"""
    resolved_url = settings.resolved_database_url()
    if resolved_url.startswith("sqlite:///"):
        raw_path = unquote(resolved_url.removeprefix("sqlite:///"))
        if raw_path:
            # 兼容形如 sqlite:////var/data/meta.db 与 sqlite:///D:/data/meta.db
            return Path(raw_path)
    if settings.database_url:
        parsed = urlparse(settings.database_url)
        if parsed.scheme == "sqlite" and parsed.path:
            raw_path = unquote(parsed.path)
            is_windows_drive = (
                len(raw_path) >= 3
                and raw_path[0] == "/"
                and raw_path[1].isalpha()
                and raw_path[2] == ":"
            )
            return Path(raw_path[1:] if is_windows_drive else raw_path)
    return None


def get_metadata_catalog(settings: Settings) -> MetadataCatalogAdapter:
    """根据全局配置构造并返回对应的元数据目录适配器实例。

    Args:
        settings: 全局运行时配置实例，通过 ``metadata_store`` 与数据库连接配置决定后端类型。

    Returns:
        满足 ``MetadataCatalogAdapter`` 协议的元数据目录实例。

    Raises:
        ValueError: 当 ``settings.metadata_store`` 不支持或外部数据库缺少连接参数时抛出。
        ImportError: 当选择 ``postgresql`` 或 ``mysql`` 但对应可选依赖驱动未安装时抛出。
    """
    store = (settings.metadata_store or "sqlite").strip().lower()
    if store == "sqlite":
        custom_db_path = _resolve_sqlite_db_path(settings)
        return SQLiteMetadataCatalog(data_dir=settings.data_dir, db_path=custom_db_path)

    if store == "postgresql":
        return PostgresMetadataCatalog(dsn=settings.resolved_database_url())

    if store == "mysql":
        return MySQLMetadataCatalog(dsn=settings.resolved_database_url())

    raise ValueError(
        f"不支持的元数据存储引擎 '{settings.metadata_store}'，"
        f"可选值: {', '.join(sorted(SUPPORTED_METADATA_STORES))}"
    )
