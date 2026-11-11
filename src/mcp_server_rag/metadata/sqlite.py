"""
SQLite 元数据目录引擎实现。

负责维护知识库注册表（含 owner/visibility/embedding_dimension）、文档内容指纹、
切片上下文链表（content/prev_chunk_id/next_chunk_id/images_json）及图片资产登记表的事务性读写，
并提供对存量旧版 SQLite 数据库的幂等平滑迁移能力。

@author Ateng
@since 2026-11-11
"""

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp_server_rag.assets import make_asset_doc_key, make_asset_id
from mcp_server_rag.fts import build_sqlite_fts5_query, tokenize_fts_text
from mcp_server_rag.metadata.base import (
    CHUNK_QUERY_BATCH_SIZE,
    CHUNK_SELECT_COLUMNS,
    deduplicate_chunk_ids,
    merge_fts_batch_hits,
    normalize_chunk_row,
    normalize_fts_hit_row,
    normalize_kb_names,
)

logger = logging.getLogger(__name__)

VALID_VISIBILITIES: frozenset[str] = frozenset({"public", "private"})

_ALLOWED_MIGRATION_TABLES: frozenset[str] = frozenset({"knowledge_bases", "chunks"})
_ALLOWED_MIGRATION_COLUMNS: frozenset[str] = frozenset(
    {
        "embedding_dimension",
        "owner",
        "visibility",
        "content",
        "prev_chunk_id",
        "next_chunk_id",
        "images_json",
        "created_at",
    }
)

_KB_SUMMARY_SELECT: str = """
    SELECT kb.name, kb.description, kb.embedding_model, kb.embedding_dimension,
           kb.owner, kb.visibility, kb.created_at, kb.updated_at,
           COUNT(DISTINCT d.id) AS document_count,
           COALESCE(SUM(d.chunk_count), 0) AS chunk_count
    FROM knowledge_bases kb
    LEFT JOIN documents d ON kb.name = d.kb_name
"""


class SQLiteMetadataCatalog:
    """基于内置 SQLite 的元数据目录实现。"""

    def __init__(self, data_dir: Path, db_path: Path | None = None) -> None:
        """初始化 SQLite 元数据目录并自动建表与迁移存量字段。

        Args:
            data_dir: 运行时数据根目录，当未指定 ``db_path`` 时在此目录下创建 ``metadata.db``。
            db_path: 可选自定义 SQLite 文件路径；指定时优先使用该路径。
        """
        self.data_dir = data_dir
        self.db_path = db_path if db_path is not None else (self.data_dir / "metadata.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """获取带超时保护的 SQLite 连接，并在退出上下文时自动提交与关闭连接。"""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        with closing(conn), conn:
            yield conn

    def _ensure_columns(
        self,
        cursor: sqlite3.Cursor,
        table: str,
        columns: list[tuple[str, str]],
    ) -> None:
        """为存量 SQLite 表幂等补充缺失字段（严格校验表名与列名白名单）。"""
        if table not in _ALLOWED_MIGRATION_TABLES:
            raise ValueError(f"非法的迁移目标表: {table}")
        cursor.execute(f"PRAGMA table_info({table})")
        existing = {str(row[1]) for row in cursor.fetchall()}
        for col_name, col_def in columns:
            if col_name not in _ALLOWED_MIGRATION_COLUMNS:
                raise ValueError(f"非法的迁移列名: {col_name}")
            if col_name not in existing:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
                logger.info("已为存量表 %s 自动迁移新增列 %s", table, col_name)

    def _init_schema(self) -> None:
        """初始化元数据数据表结构并兼容迁移存量字段。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    name TEXT PRIMARY KEY,
                    description TEXT NOT NULL DEFAULT '',
                    embedding_model TEXT NOT NULL DEFAULT '',
                    embedding_dimension INTEGER NOT NULL DEFAULT 0,
                    owner TEXT NOT NULL DEFAULT 'default',
                    visibility TEXT NOT NULL DEFAULT 'public',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    kb_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    source_uri TEXT NOT NULL DEFAULT '',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (kb_name) REFERENCES knowledge_bases(name) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    kb_name TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    prev_chunk_id TEXT DEFAULT NULL,
                    next_chunk_id TEXT DEFAULT NULL,
                    images_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS image_assets (
                    asset_id TEXT PRIMARY KEY,
                    kb_name TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    storage_uri TEXT NOT NULL,
                    mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
                    byte_size INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (kb_name) REFERENCES knowledge_bases(name) ON DELETE CASCADE,
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_kb_doc_idx
                ON chunks (kb_name, doc_id, chunk_index)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_image_assets_kb_doc
                ON image_assets (kb_name, doc_id, filename)
                """
            )

            # 对存量旧版数据库自动执行幂等 ALTER TABLE 迁移
            self._ensure_columns(
                cursor,
                "knowledge_bases",
                [
                    ("embedding_dimension", "INTEGER NOT NULL DEFAULT 0"),
                    ("owner", "TEXT NOT NULL DEFAULT 'default'"),
                    ("visibility", "TEXT NOT NULL DEFAULT 'public'"),
                ],
            )
            self._ensure_columns(
                cursor,
                "chunks",
                [
                    ("content", "TEXT NOT NULL DEFAULT ''"),
                    ("prev_chunk_id", "TEXT DEFAULT NULL"),
                    ("next_chunk_id", "TEXT DEFAULT NULL"),
                    ("images_json", "TEXT NOT NULL DEFAULT '[]'"),
                    ("created_at", "TEXT NOT NULL DEFAULT ''"),
                ],
            )
            self._ensure_fts_schema(cursor)
            conn.commit()

    def _ensure_fts_schema(self, cursor: sqlite3.Cursor) -> None:
        """创建 SQLite FTS5 全文检索虚表并幂等回填存量切片倒排索引。"""
        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                kb_name UNINDEXED,
                doc_id UNINDEXED,
                tokens,
                tokenize="unicode61 tokenchars '_'"
            )
            """
        )
        cursor.execute("SELECT chunk_id FROM chunks_fts")
        indexed_ids = {str(row[0]) for row in cursor.fetchall()}
        cursor.execute("SELECT id, kb_name, doc_id, content FROM chunks WHERE content != ''")
        missing_rows = [
            (str(row[0]), str(row[1]), str(row[2]), tokenize_fts_text(str(row[3])))
            for row in cursor.fetchall()
            if str(row[0]) not in indexed_ids
        ]
        if missing_rows:
            cursor.executemany(
                "INSERT INTO chunks_fts (chunk_id, kb_name, doc_id, tokens) VALUES (?, ?, ?, ?)",
                missing_rows,
            )
            logger.info("已为存量 %d 条切片自动补建 SQLite FTS5 倒排索引", len(missing_rows))

    def create_kb(
        self,
        name: str,
        description: str = "",
        embedding_model: str = "",
        embedding_dimension: int = 0,
        owner: str = "default",
        visibility: str = "public",
    ) -> bool:
        """创建新知识库记录。

        Args:
            name: 知识库唯一名称。
            description: 描述信息。
            embedding_model: 绑定的嵌入模型。
            embedding_dimension: 绑定的向量维度，默认 ``0``。
            owner: 所属用户或角色标识，默认 ``"default"``。
            visibility: 可见性（``"public"`` 或 ``"private"``），默认 ``"public"``。

        Returns:
            创建成功返回 ``True``；若知识库名称已存在则返回 ``False``。

        Raises:
            ValueError: 当 ``visibility`` 不是 ``"public"`` 或 ``"private"`` 时抛出。
        """
        normalized_visibility = visibility.strip().lower()
        if normalized_visibility not in VALID_VISIBILITIES:
            raise ValueError(
                f"非法的知识库可见性 visibility='{visibility}'，仅支持 'public' 或 'private'"
            )
        normalized_owner = owner.strip() or "default"
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO knowledge_bases (
                        name, description, embedding_model, embedding_dimension,
                        owner, visibility, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name,
                        description,
                        embedding_model,
                        int(embedding_dimension or 0),
                        normalized_owner,
                        normalized_visibility,
                        now,
                        now,
                    ),
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False

    def get_kb(self, name: str) -> dict[str, Any] | None:
        """按名称查找知识库。未找到时返回 ``None``。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                {_KB_SUMMARY_SELECT}
                WHERE kb.name = ?
                GROUP BY kb.name
                """,
                (name,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    def list_kbs(self) -> list[dict[str, Any]]:
        """获取所有知识库列表与切片统计。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                {_KB_SUMMARY_SELECT}
                GROUP BY kb.name
                ORDER BY kb.created_at ASC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def clear_kb(self, name: str) -> bool:
        """清空指定知识库的所有文档、切片与关联图片资产记录。"""
        if not self.get_kb(name):
            return False
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks_fts WHERE kb_name = ?", (name,))
            cursor.execute("DELETE FROM image_assets WHERE kb_name = ?", (name,))
            cursor.execute("DELETE FROM chunks WHERE kb_name = ?", (name,))
            cursor.execute("DELETE FROM documents WHERE kb_name = ?", (name,))
            cursor.execute(
                "UPDATE knowledge_bases SET updated_at = ? WHERE name = ?",
                (now, name),
            )
            conn.commit()
            return True

    def update_kb_embedding_model(
        self,
        name: str,
        embedding_model: str,
        embedding_dimension: int = 0,
    ) -> None:
        """更新知识库绑定的嵌入模型标识与维度。"""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE knowledge_bases
                SET embedding_model = ?, embedding_dimension = ?, updated_at = ?
                WHERE name = ?
                """,
                (embedding_model, int(embedding_dimension or 0), now, name),
            )
            conn.commit()

    def delete_kb(self, name: str) -> bool:
        """物理删除指定知识库及其关联的文档、切片与图片资产记录。"""
        if not self.get_kb(name):
            return False
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks_fts WHERE kb_name = ?", (name,))
            cursor.execute("DELETE FROM image_assets WHERE kb_name = ?", (name,))
            cursor.execute("DELETE FROM chunks WHERE kb_name = ?", (name,))
            cursor.execute("DELETE FROM documents WHERE kb_name = ?", (name,))
            cursor.execute("DELETE FROM knowledge_bases WHERE name = ?", (name,))
            conn.commit()
            return True

    def get_document_by_title(self, kb_name: str, title: str) -> dict[str, Any] | None:
        """根据知识库名称与文档标题查找文档记录。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, kb_name, title, content_hash, source_uri, chunk_count, file_size,
                       created_at, updated_at
                FROM documents
                WHERE kb_name = ? AND title = ?
                """,
                (kb_name, title),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_document_by_id(self, kb_name: str, doc_id: str) -> dict[str, Any] | None:
        """根据文档唯一 ID 查找文档记录。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, kb_name, title, content_hash, source_uri, chunk_count, file_size,
                       created_at, updated_at
                FROM documents
                WHERE kb_name = ? AND id = ?
                """,
                (kb_name, doc_id),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def find_document(self, kb_name: str, doc_identifier: str) -> dict[str, Any] | None:
        """多策略查找文档：支持按文档 ID、标题、源路径智能匹配。"""
        clean_id = doc_identifier.strip()
        if not clean_id:
            return None

        # 1. 精确匹配主键 ID
        doc = self.get_document_by_id(kb_name, clean_id)
        if doc:
            return doc

        # 2. 尝试本地路径解析匹配
        try:
            resolved_str = str(Path(clean_id).resolve())
            doc = self.get_document_by_id(kb_name, resolved_str)
            if doc:
                return doc
        except Exception as exc:
            logger.debug("按路径解析文档标识 '%s' 时跳过路径匹配: %s", clean_id, exc)

        # 3. 按 source_uri 或 title 匹配
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, kb_name, title, content_hash, source_uri, chunk_count, file_size,
                       created_at, updated_at
                FROM documents
                WHERE kb_name = ? AND (source_uri = ? OR title = ?)
                """,
                (kb_name, clean_id, clean_id),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_document(self, kb_name: str, doc_id: str) -> list[str]:
        """删除指定文档及其切片与图片资产记录，返回被删除的切片 ID 列表。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM chunks WHERE kb_name = ? AND doc_id = ?",
                (kb_name, doc_id),
            )
            chunk_ids = [str(row[0]) for row in cursor.fetchall()]

            cursor.execute(
                "DELETE FROM chunks_fts WHERE kb_name = ? AND doc_id = ?",
                (kb_name, doc_id),
            )
            cursor.execute(
                "DELETE FROM image_assets WHERE kb_name = ? AND doc_id = ?",
                (kb_name, doc_id),
            )
            cursor.execute(
                "DELETE FROM chunks WHERE kb_name = ? AND doc_id = ?",
                (kb_name, doc_id),
            )
            cursor.execute(
                "DELETE FROM documents WHERE kb_name = ? AND id = ?",
                (kb_name, doc_id),
            )
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                "UPDATE knowledge_bases SET updated_at = ? WHERE name = ?",
                (now, kb_name),
            )
            conn.commit()
            return chunk_ids

    def register_document_and_chunks(
        self,
        kb_name: str,
        doc_id: str,
        title: str,
        content_hash: str,
        source_uri: str,
        chunk_ids: list[str],
        file_size: int,
        chunk_contents: list[str] | None = None,
        chunk_images: list[list[str]] | None = None,
    ) -> None:
        """原子注册文档与其有序切片链表记录。"""
        now = datetime.now(timezone.utc).isoformat()
        total_chunks = len(chunk_ids)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. 插入或更新文档主记录
            cursor.execute(
                """
                INSERT INTO documents (
                    id, kb_name, title, content_hash, source_uri,
                    chunk_count, file_size, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    chunk_count = excluded.chunk_count,
                    file_size = excluded.file_size,
                    updated_at = excluded.updated_at
                """,
                (
                    doc_id,
                    kb_name,
                    title,
                    content_hash,
                    source_uri,
                    total_chunks,
                    file_size,
                    now,
                    now,
                ),
            )
            # 2. 先清理旧切片及 FTS 索引再重写，避免缩减分片或更新文本时残留脏数据
            cursor.execute(
                "DELETE FROM chunks_fts WHERE kb_name = ? AND doc_id = ?",
                (kb_name, doc_id),
            )
            cursor.execute(
                "DELETE FROM chunks WHERE kb_name = ? AND doc_id = ?",
                (kb_name, doc_id),
            )
            fts_rows: list[tuple[str, str, str, str]] = []
            for idx, c_id in enumerate(chunk_ids):
                prev_chunk_id = chunk_ids[idx - 1] if idx > 0 else None
                next_chunk_id = chunk_ids[idx + 1] if idx + 1 < total_chunks else None
                chunk_text = (
                    chunk_contents[idx]
                    if chunk_contents is not None and idx < len(chunk_contents)
                    else ""
                )
                images_list = (
                    chunk_images[idx]
                    if chunk_images is not None and idx < len(chunk_images)
                    else []
                )
                images_json = json.dumps(images_list, ensure_ascii=False)
                cursor.execute(
                    """
                    INSERT INTO chunks (
                        id, kb_name, doc_id, chunk_index, content,
                        prev_chunk_id, next_chunk_id, images_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        chunk_index = excluded.chunk_index,
                        content = excluded.content,
                        prev_chunk_id = excluded.prev_chunk_id,
                        next_chunk_id = excluded.next_chunk_id,
                        images_json = excluded.images_json,
                        created_at = excluded.created_at
                    """,
                    (
                        c_id,
                        kb_name,
                        doc_id,
                        idx,
                        chunk_text,
                        prev_chunk_id,
                        next_chunk_id,
                        images_json,
                        now,
                    ),
                )
                fts_rows.append((c_id, kb_name, doc_id, tokenize_fts_text(chunk_text)))
            if fts_rows:
                cursor.executemany(
                    "INSERT INTO chunks_fts (chunk_id, kb_name, doc_id, tokens) "
                    "VALUES (?, ?, ?, ?)",
                    fts_rows,
                )
            cursor.execute(
                "UPDATE knowledge_bases SET updated_at = ? WHERE name = ?",
                (now, kb_name),
            )
            conn.commit()

    def get_chunks_by_ids(
        self,
        kb_name: str,
        chunk_ids: list[str],
    ) -> list[dict[str, Any]]:
        """按切片 ID 列表批量查询持久化切片正文、双向链表指针与图片清单。"""
        unique_ids = deduplicate_chunk_ids(chunk_ids)
        if not unique_ids:
            return []

        results: list[dict[str, Any]] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for start in range(0, len(unique_ids), CHUNK_QUERY_BATCH_SIZE):
                batch = unique_ids[start : start + CHUNK_QUERY_BATCH_SIZE]
                placeholders = ", ".join("?" for _ in batch)
                cursor.execute(
                    f"""
                    SELECT {CHUNK_SELECT_COLUMNS}
                    FROM chunks
                    WHERE kb_name = ? AND id IN ({placeholders})
                    ORDER BY doc_id ASC, chunk_index ASC
                    """,
                    (kb_name, *batch),
                )
                results.extend(normalize_chunk_row(dict(row)) for row in cursor.fetchall())
        return results

    def get_chunks_by_doc(
        self,
        kb_name: str,
        doc_id: str,
    ) -> list[dict[str, Any]]:
        """按文档 ID 查询全部有序切片记录（按 ``chunk_index`` 升序）。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT {CHUNK_SELECT_COLUMNS}
                FROM chunks
                WHERE kb_name = ? AND doc_id = ?
                ORDER BY chunk_index ASC
                """,
                (kb_name, doc_id),
            )
            return [normalize_chunk_row(dict(row)) for row in cursor.fetchall()]

    def search_chunks_fts(
        self,
        kb_names: list[str],
        query: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """基于 SQLite FTS5 虚表执行倒排关键词召回并按 BM25 相关性排序。"""
        clean_kbs = normalize_kb_names(kb_names)
        if not clean_kbs or top_k <= 0:
            return []

        fts_query = build_sqlite_fts5_query(query)
        if not fts_query:
            return []

        hits: list[dict[str, Any]] = []
        batch_count = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for start in range(0, len(clean_kbs), CHUNK_QUERY_BATCH_SIZE):
                batch_count += 1
                kb_batch = clean_kbs[start : start + CHUNK_QUERY_BATCH_SIZE]
                placeholders = ", ".join("?" for _ in kb_batch)
                cursor.execute(
                    f"""
                    SELECT c.id, c.kb_name, c.doc_id, c.chunk_index, c.content,
                           c.prev_chunk_id, c.next_chunk_id, c.images_json, c.created_at,
                           d.title AS doc_title, d.source_uri AS doc_source_uri,
                           -bm25(chunks_fts) AS fts_score
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.chunk_id
                    LEFT JOIN documents d ON c.doc_id = d.id AND c.kb_name = d.kb_name
                    WHERE chunks_fts MATCH ?
                      AND chunks_fts.kb_name IN ({placeholders})
                    ORDER BY bm25(chunks_fts) ASC, c.id ASC
                    LIMIT ?
                    """,
                    (fts_query, *kb_batch, top_k),
                )
                hits.extend(normalize_fts_hit_row(dict(row)) for row in cursor.fetchall())

        return merge_fts_batch_hits(hits, batch_count=batch_count, top_k=top_k)

    def register_image_assets(
        self,
        kb_name: str,
        doc_id: str,
        assets: list[dict[str, Any]],
    ) -> None:
        """批量幂等登记文档关联图片资产。"""
        if not assets:
            return
        now = datetime.now(timezone.utc).isoformat()
        doc_key = make_asset_doc_key(doc_id)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for asset in assets:
                filename = str(asset.get("filename") or "").strip()
                storage_uri = str(asset.get("storage_uri") or "").strip()
                if not filename or not storage_uri:
                    raise ValueError("图片资产的 filename 与 storage_uri 不能为空")
                mime_type = str(asset.get("mime_type") or "application/octet-stream")
                byte_size = max(0, int(asset.get("byte_size") or 0))
                created_at = str(asset.get("created_at") or now)
                asset_id = str(asset.get("asset_id") or make_asset_id(kb_name, doc_key, filename))

                # 先按 (kb_name, doc_id, filename) 清理旧记录以保障重复写入幂等
                cursor.execute(
                    """
                    DELETE FROM image_assets
                    WHERE kb_name = ? AND doc_id = ? AND filename = ? AND asset_id != ?
                    """,
                    (kb_name, doc_id, filename, asset_id),
                )
                cursor.execute(
                    """
                    INSERT INTO image_assets (
                        asset_id, kb_name, doc_id, filename, storage_uri,
                        mime_type, byte_size, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(asset_id) DO UPDATE SET
                        kb_name = excluded.kb_name,
                        doc_id = excluded.doc_id,
                        filename = excluded.filename,
                        storage_uri = excluded.storage_uri,
                        mime_type = excluded.mime_type,
                        byte_size = excluded.byte_size
                    """,
                    (
                        asset_id,
                        kb_name,
                        doc_id,
                        filename,
                        storage_uri,
                        mime_type,
                        byte_size,
                        created_at,
                    ),
                )
            conn.commit()

    def get_image_asset(
        self,
        kb_name: str,
        doc_id: str,
        filename: str,
    ) -> dict[str, Any] | None:
        """按文档 ID（或单段目录键 ``doc_key``）与文件名查询单条图片资产记录。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT asset_id, kb_name, doc_id, filename, storage_uri,
                       mime_type, byte_size, created_at
                FROM image_assets
                WHERE kb_name = ? AND doc_id = ? AND filename = ?
                """,
                (kb_name, doc_id, filename),
            )
            row = cursor.fetchone()
            if row is None and doc_id:
                candidate_asset_id = make_asset_id(kb_name, doc_id, filename)
                cursor.execute(
                    """
                    SELECT asset_id, kb_name, doc_id, filename, storage_uri,
                           mime_type, byte_size, created_at
                    FROM image_assets
                    WHERE asset_id = ? AND kb_name = ? AND filename = ?
                    """,
                    (candidate_asset_id, kb_name, filename),
                )
                row = cursor.fetchone()
            return dict(row) if row else None

    def list_image_assets(
        self,
        kb_name: str,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出知识库或指定文档的图片资产清单。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if doc_id is not None:
                cursor.execute(
                    """
                    SELECT asset_id, kb_name, doc_id, filename, storage_uri,
                           mime_type, byte_size, created_at
                    FROM image_assets
                    WHERE kb_name = ? AND doc_id = ?
                    ORDER BY created_at ASC, filename ASC
                    """,
                    (kb_name, doc_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT asset_id, kb_name, doc_id, filename, storage_uri,
                           mime_type, byte_size, created_at
                    FROM image_assets
                    WHERE kb_name = ?
                    ORDER BY created_at ASC, filename ASC
                    """,
                    (kb_name,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def count_kbs(self) -> int:
        """统计知识库总数。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM knowledge_bases")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def count_documents(self, kb_name: str | None = None) -> int:
        """统计文档总数。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if kb_name:
                cursor.execute("SELECT COUNT(*) FROM documents WHERE kb_name = ?", (kb_name,))
            else:
                cursor.execute("SELECT COUNT(*) FROM documents")
            row = cursor.fetchone()
            return int(row[0]) if row else 0

    def ping(self) -> bool:
        """测试 SQLite 数据库健康连通性。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception as exc:
            logger.warning("SQLite 元数据目录健康检查失败: %s", exc)
            return False
