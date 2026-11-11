"""
PostgreSQL 元数据目录适配器实现。

基于可选依赖 ``psycopg`` 实现面向多实例并发共享场景的企业级 PostgreSQL 元数据目录，
提供幂等 DDL 建表、``ON CONFLICT`` 冲突写入及优雅的缺失驱动引导提示。

@author Ateng
@since 2026-11-11
"""

import importlib
import logging
from typing import Any

from mcp_server_rag.fts import build_postgres_tsquery, tokenize_fts_text
from mcp_server_rag.metadata.base import normalize_kb_names
from mcp_server_rag.metadata.relational import _RelationalMetadataCatalog

logger = logging.getLogger(__name__)


def _load_psycopg() -> Any:
    """惰性导入 psycopg 驱动；未安装时返回明确的安装指引。"""
    try:
        return importlib.import_module("psycopg")
    except ImportError as exc:
        raise ImportError(
            "未检测到 psycopg 依赖，无法使用 PostgreSQL 元数据目录。"
            "请运行 `pip install atengk-mcp-server-rag[postgres]` "
            '或 `uv pip install "psycopg[binary]>=3.1.0"` 进行安装。'
        ) from exc


class PostgresMetadataCatalog(_RelationalMetadataCatalog):
    """基于 psycopg 的 PostgreSQL 元数据目录适配器。"""

    _backend_name = "postgresql"

    _ddl_statements = (
        """
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            name VARCHAR(128) PRIMARY KEY,
            description TEXT NOT NULL DEFAULT '',
            embedding_model VARCHAR(255) NOT NULL DEFAULT '',
            embedding_dimension INTEGER NOT NULL DEFAULT 0,
            owner VARCHAR(128) NOT NULL DEFAULT 'default',
            visibility VARCHAR(32) NOT NULL DEFAULT 'public',
            created_at VARCHAR(64) NOT NULL,
            updated_at VARCHAR(64) NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS documents (
            id VARCHAR(512) PRIMARY KEY,
            kb_name VARCHAR(128) NOT NULL,
            title VARCHAR(512) NOT NULL,
            content_hash VARCHAR(128) NOT NULL,
            source_uri TEXT NOT NULL DEFAULT '',
            chunk_count INTEGER NOT NULL DEFAULT 0,
            file_size BIGINT NOT NULL DEFAULT 0,
            created_at VARCHAR(64) NOT NULL,
            updated_at VARCHAR(64) NOT NULL,
            FOREIGN KEY (kb_name) REFERENCES knowledge_bases(name) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id VARCHAR(512) PRIMARY KEY,
            kb_name VARCHAR(128) NOT NULL,
            doc_id VARCHAR(512) NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            prev_chunk_id VARCHAR(512) DEFAULT NULL,
            next_chunk_id VARCHAR(512) DEFAULT NULL,
            images_json TEXT NOT NULL DEFAULT '[]',
            created_at VARCHAR(64) NOT NULL,
            FOREIGN KEY (kb_name) REFERENCES knowledge_bases(name) ON DELETE CASCADE,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        )
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
            FOREIGN KEY (kb_name) REFERENCES knowledge_bases(name) ON DELETE CASCADE,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_kb_doc_idx
        ON chunks (kb_name, doc_id, chunk_index)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_image_assets_kb_doc
        ON image_assets (kb_name, doc_id)
        """,
    )

    def _ensure_driver(self) -> Any:
        """校验 psycopg 驱动已安装并返回模块引用。"""
        return _load_psycopg()

    def _open_connection(self) -> Any:
        """通过 psycopg 打开配置连接超时与语句执行超时的 PostgreSQL 数据库连接。"""
        psycopg = _load_psycopg()
        return psycopg.connect(
            self.dsn,
            connect_timeout=10,
            options="-c statement_timeout=30000",
        )

    def _insert_kb_row(self, cursor: Any, params: tuple[Any, ...]) -> bool:
        """使用 PostgreSQL ``ON CONFLICT DO NOTHING`` 幂等插入知识库记录。"""
        cursor.execute(
            """
            INSERT INTO knowledge_bases (
                name, description, embedding_model, embedding_dimension,
                owner, visibility, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) DO NOTHING
            """,
            params,
        )
        return int(getattr(cursor, "rowcount", 0)) > 0

    def _upsert_document(self, cursor: Any, params: tuple[Any, ...]) -> None:
        """使用 PostgreSQL ``ON CONFLICT DO UPDATE`` 幂等写入文档记录。"""
        cursor.execute(
            """
            INSERT INTO documents (
                id, kb_name, title, content_hash, source_uri,
                chunk_count, file_size, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                content_hash = EXCLUDED.content_hash,
                chunk_count = EXCLUDED.chunk_count,
                file_size = EXCLUDED.file_size,
                updated_at = EXCLUDED.updated_at
            """,
            params,
        )

    def _upsert_image_asset(self, cursor: Any, params: tuple[Any, ...]) -> None:
        """使用 PostgreSQL ``ON CONFLICT DO UPDATE`` 幂等写入图片资产记录。"""
        cursor.execute(
            """
            INSERT INTO image_assets (
                asset_id, kb_name, doc_id, filename, storage_uri,
                mime_type, byte_size, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (asset_id) DO UPDATE SET
                kb_name = EXCLUDED.kb_name,
                doc_id = EXCLUDED.doc_id,
                filename = EXCLUDED.filename,
                storage_uri = EXCLUDED.storage_uri,
                mime_type = EXCLUDED.mime_type,
                byte_size = EXCLUDED.byte_size
            """,
            params,
        )

    def _post_init_schema(self, cursor: Any) -> None:
        """幂等扩展 PostgreSQL ``chunks.content_tsv`` 列及 GIN 倒排索引并按需回填存量数据。"""
        cursor.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR")
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv
            ON chunks USING GIN (content_tsv)
            """
        )
        while True:
            cursor.execute(
                """
                SELECT id, content
                FROM chunks
                WHERE content_tsv IS NULL AND content <> ''
                LIMIT 500
                """
            )
            pending = cursor.fetchall()
            if not pending:
                break
            updates: list[tuple[str, str]] = []
            for raw_row in pending:
                row = self._row_to_dict(cursor, raw_row)
                tokenized = tokenize_fts_text(str(row.get("content") or ""))
                updates.append((tokenized, str(row["id"])))
            cursor.executemany(
                "UPDATE chunks SET content_tsv = to_tsvector('simple', %s) WHERE id = %s",
                updates,
            )
            logger.info("已为存量 %d 条 PostgreSQL 切片补建 tsvector 倒排索引", len(pending))

    def _upsert_chunk_row(self, cursor: Any, params: tuple[Any, ...]) -> None:
        """原子写入切片正文并同步维护 ``content_tsv`` 倒排向量列。"""
        tokenized = tokenize_fts_text(str(params[4]))
        cursor.execute(
            """
            INSERT INTO chunks (
                id, kb_name, doc_id, chunk_index, content,
                prev_chunk_id, next_chunk_id, images_json, created_at, content_tsv
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))
            """,
            (*params, tokenized),
        )

    def search_chunks_fts(
        self,
        kb_names: list[str],
        query: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """基于 PostgreSQL GIN ``tsvector`` 执行倒排检索并按 ``ts_rank_cd`` 排序。"""
        clean_kbs = normalize_kb_names(kb_names)
        if not clean_kbs or top_k <= 0:
            return []

        tsquery = build_postgres_tsquery(query)
        if not tsquery:
            return []

        return self._execute_fts_query(
            clean_kbs=clean_kbs,
            score_expr="ts_rank_cd(c.content_tsv, to_tsquery('simple', %s))",
            score_params=(tsquery,),
            match_clause="c.content_tsv @@ to_tsquery('simple', %s)",
            where_params=(tsquery,),
            order_clause="fts_score DESC, c.id ASC",
            top_k=top_k,
        )
