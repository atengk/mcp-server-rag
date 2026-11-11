"""
可插拔元数据目录模块包。

统一导出抽象协议 ``MetadataCatalogAdapter``、默认内置 ``SQLiteMetadataCatalog`` 实现
（含向后兼容别名 ``MetadataCatalog``）、企业级 ``PostgresMetadataCatalog`` 与
``MySQLMetadataCatalog`` 实现以及工厂函数 ``get_metadata_catalog``。

@author Ateng
@since 2026-11-11
"""

from mcp_server_rag.metadata.base import MetadataCatalogAdapter
from mcp_server_rag.metadata.factory import get_metadata_catalog
from mcp_server_rag.metadata.mysql import MySQLMetadataCatalog
from mcp_server_rag.metadata.postgres import PostgresMetadataCatalog
from mcp_server_rag.metadata.sqlite import SQLiteMetadataCatalog

# 保持旧版本类名向后兼容
MetadataCatalog = SQLiteMetadataCatalog

__all__ = [
    "MetadataCatalog",
    "MetadataCatalogAdapter",
    "MySQLMetadataCatalog",
    "PostgresMetadataCatalog",
    "SQLiteMetadataCatalog",
    "get_metadata_catalog",
]
