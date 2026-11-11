"""
MySQL 元数据目录适配器实现。

基于可选依赖 ``pymysql`` 实现兼容 UTF8MB4 字符集的企业级 MySQL 元数据目录，
提供幂等 InnoDB DDL 建表、``ON DUPLICATE KEY UPDATE`` 更新与缺失驱动引导提示。

@author Ateng
@since 2026-11-11
"""

import importlib
import logging
from typing import Any
from urllib.parse import unquote, urlparse

from mcp_server_rag.fts import build_mysql_ft_query
from mcp_server_rag.metadata.base import normalize_kb_names
from mcp_server_rag.metadata.relational import _RelationalMetadataCatalog

logger = logging.getLogger(__name__)


def _load_pymysql() -> Any:
    """惰性导入 pymysql 驱动；未安装时返回明确的安装指引。"""
    try:
        return importlib.import_module("pymysql")
    except ImportError as exc:
        raise ImportError(
            "未检测到 pymysql 依赖，无法使用 MySQL 元数据目录。"
            "请运行 `pip install atengk-mcp-server-rag[mysql]` "
            '或 `uv pip install "pymysql>=1.1.0"` 进行安装。'
        ) from exc


def _mask_dsn(dsn: str) -> str:
    """脱敏连接字符串中的明文密码，防止异常日志泄露凭据。"""
    parsed = urlparse(dsn)
    if parsed.password is None:
        return dsn
    user_part = f"{parsed.username}:***@" if parsed.username is not None else ":***@"
    host_part = parsed.hostname or ""
    port_part = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{user_part}{host_part}{port_part}{parsed.path}"


def _is_duplicate_key_error(exc: Exception) -> bool:
    """精准判断驱动异常是否为主键/唯一键重复冲突（MySQL 1062 / Duplicate entry）。"""
    if exc.args and exc.args[0] == 1062:
        return True
    msg = str(exc)
    return "Duplicate entry" in msg or "UNIQUE constraint failed" in msg


def _is_duplicate_index_error(exc: Exception) -> bool:
    """精准判断驱动异常是否为重复创建全文索引（MySQL 1061 / Duplicate key name）。"""
    if exc.args and exc.args[0] == 1061:
        return True
    return "Duplicate key name" in str(exc)


def _is_ngram_unsupported_error(exc: Exception) -> bool:
    """判断 MySQL 实例是否因未加载 ngram 分词插件而无法解析 ``WITH PARSER ngram``。"""
    if exc.args and exc.args[0] == 1128:
        return True
    lowered = str(exc).lower()
    return "ngram" in lowered or "parser" in lowered


class MySQLMetadataCatalog(_RelationalMetadataCatalog):
    """基于 pymysql 的 MySQL 元数据目录适配器。"""

    _backend_name = "mysql"

    _ddl_statements = (
        """
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            name VARCHAR(128) PRIMARY KEY,
            description TEXT NOT NULL,
            embedding_model VARCHAR(255) NOT NULL DEFAULT '',
            embedding_dimension INTEGER NOT NULL DEFAULT 0,
            owner VARCHAR(128) NOT NULL DEFAULT 'default',
            visibility VARCHAR(32) NOT NULL DEFAULT 'public',
            created_at VARCHAR(64) NOT NULL,
            updated_at VARCHAR(64) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        """
        CREATE TABLE IF NOT EXISTS documents (
            id VARCHAR(512) PRIMARY KEY,
            kb_name VARCHAR(128) NOT NULL,
            title VARCHAR(512) NOT NULL,
            content_hash VARCHAR(128) NOT NULL,
            source_uri TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            file_size BIGINT NOT NULL DEFAULT 0,
            created_at VARCHAR(64) NOT NULL,
            updated_at VARCHAR(64) NOT NULL,
            INDEX idx_documents_kb_name (kb_name),
            FOREIGN KEY (kb_name) REFERENCES knowledge_bases(name) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id VARCHAR(512) PRIMARY KEY,
            kb_name VARCHAR(128) NOT NULL,
            doc_id VARCHAR(512) NOT NULL,
            chunk_index INTEGER NOT NULL,
            content LONGTEXT NOT NULL,
            prev_chunk_id VARCHAR(512) NULL,
            next_chunk_id VARCHAR(512) NULL,
            images_json TEXT NOT NULL,
            created_at VARCHAR(64) NOT NULL,
            INDEX idx_chunks_kb_doc_idx (kb_name, doc_id, chunk_index),
            FOREIGN KEY (kb_name) REFERENCES knowledge_bases(name) ON DELETE CASCADE,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
        """
        CREATE TABLE IF NOT EXISTS image_assets (
            asset_id VARCHAR(255) PRIMARY KEY,
            kb_name VARCHAR(128) NOT NULL,
            doc_id VARCHAR(512) NOT NULL,
            filename VARCHAR(512) NOT NULL,
            storage_uri TEXT NOT NULL,
            mime_type VARCHAR(128) NOT NULL DEFAULT 'application/octet-stream',
            byte_size BIGINT NOT NULL DEFAULT 0,
            created_at VARCHAR(64) NOT NULL,
            INDEX idx_image_assets_kb_doc (kb_name, doc_id),
            FOREIGN KEY (kb_name) REFERENCES knowledge_bases(name) ON DELETE CASCADE,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """,
    )

    def _parse_connect_kwargs(self) -> dict[str, Any]:
        """将标准 MySQL 连接 URL 解析并 URL 解码为 pymysql.connect 关键字参数。"""
        parsed = urlparse(self.dsn)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        database = unquote(parsed.path.lstrip("/"))
        if scheme not in {"mysql", "mysql+pymysql"} or not host or not database:
            raise ValueError(
                f"无效的 MySQL 连接字符串（仅支持 mysql:// 且须包含主机名与数据库名）: "
                f"'{_mask_dsn(self.dsn)}'"
            )
        kwargs: dict[str, Any] = {
            "host": host,
            "port": int(parsed.port or 3306),
            "database": database,
            "charset": "utf8mb4",
            "connect_timeout": 10,
            "read_timeout": 30,
            "write_timeout": 30,
            "autocommit": False,
        }
        if parsed.username is not None:
            kwargs["user"] = unquote(parsed.username)
        if parsed.password is not None:
            kwargs["password"] = unquote(parsed.password)
        return kwargs

    def _validate_dsn(self) -> None:
        """在建立连接前校验 MySQL DSN 解析完整性。"""
        self._connect_kwargs = self._parse_connect_kwargs()

    def _ensure_driver(self) -> Any:
        """校验 pymysql 驱动已安装并返回模块引用。"""
        return _load_pymysql()

    def _open_connection(self) -> Any:
        """通过 pymysql 打开配置了 UTF8MB4 与读写超时的 MySQL 数据库连接。"""
        pymysql = _load_pymysql()
        return pymysql.connect(**self._connect_kwargs)

    def _insert_kb_row(self, cursor: Any, params: tuple[Any, ...]) -> bool:
        """插入知识库记录；主键重复时捕获唯一键冲突并返回 ``False``。"""
        try:
            cursor.execute(
                """
                INSERT INTO knowledge_bases (
                    name, description, embedding_model, embedding_dimension,
                    owner, visibility, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                params,
            )
            return True
        except Exception as exc:
            if _is_duplicate_key_error(exc):
                return False
            raise

    def _upsert_document(self, cursor: Any, params: tuple[Any, ...]) -> None:
        """使用兼容 MySQL 8.0/8.4+ 的 ``ON DUPLICATE KEY UPDATE`` 幂等写入文档记录。"""
        _, _, _, content_hash, _, total_chunks, file_size, _, updated_at = params
        cursor.execute(
            """
            INSERT INTO documents (
                id, kb_name, title, content_hash, source_uri,
                chunk_count, file_size, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                content_hash = %s,
                chunk_count = %s,
                file_size = %s,
                updated_at = %s
            """,
            (*params, content_hash, total_chunks, file_size, updated_at),
        )

    def _upsert_image_asset(self, cursor: Any, params: tuple[Any, ...]) -> None:
        """使用 ``ON DUPLICATE KEY UPDATE`` 幂等写入图片资产记录。"""
        _, kb_name, doc_id, filename, storage_uri, mime_type, byte_size, _ = params
        cursor.execute(
            """
            INSERT INTO image_assets (
                asset_id, kb_name, doc_id, filename, storage_uri,
                mime_type, byte_size, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                kb_name = %s,
                doc_id = %s,
                filename = %s,
                storage_uri = %s,
                mime_type = %s,
                byte_size = %s
            """,
            (*params, kb_name, doc_id, filename, storage_uri, mime_type, byte_size),
        )

    def _post_init_schema(self, cursor: Any) -> None:
        """幂等创建 MySQL FULLTEXT 全文索引，优先启用 ngram 中文分词插件并在缺失时降级。"""
        cursor.execute("SHOW INDEX FROM chunks WHERE Key_name = 'idx_chunks_content_ft'")
        if cursor.fetchone() is not None:
            return

        try:
            cursor.execute(
                "ALTER TABLE chunks ADD FULLTEXT INDEX "
                "idx_chunks_content_ft (content) WITH PARSER ngram"
            )
        except Exception as exc:
            if _is_duplicate_index_error(exc):
                logger.debug("MySQL 全文索引 idx_chunks_content_ft 已存在，跳过创建")
                return
            if _is_ngram_unsupported_error(exc):
                logger.warning(
                    "MySQL 当前实例不支持 ngram 全文分词插件，降级创建标准 FULLTEXT 索引: %s",
                    exc,
                )
                try:
                    cursor.execute(
                        "ALTER TABLE chunks ADD FULLTEXT INDEX idx_chunks_content_ft (content)"
                    )
                except Exception as fb_exc:
                    if not _is_duplicate_index_error(fb_exc):
                        raise
            else:
                raise

    def search_chunks_fts(
        self,
        kb_names: list[str],
        query: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """基于 MySQL FULLTEXT 倒排索引执行全文检索并按相关度降序返回。"""
        clean_kbs = normalize_kb_names(kb_names)
        if not clean_kbs or top_k <= 0:
            return []

        ft_query = build_mysql_ft_query(query)
        if not ft_query:
            return []

        return self._execute_fts_query(
            clean_kbs=clean_kbs,
            score_expr="MATCH(c.content) AGAINST (%s IN NATURAL LANGUAGE MODE)",
            score_params=(ft_query,),
            match_clause="MATCH(c.content) AGAINST (%s IN NATURAL LANGUAGE MODE)",
            where_params=(ft_query,),
            order_clause="fts_score DESC, c.id ASC",
            top_k=top_k,
        )
