"""
企业级关系型元数据目录公共抽象基类。

沉淀 PostgreSQL 与 MySQL 适配器共用的事务连接生命周期管理、行映射标准化、
知识库/文档/切片链表/图片资产 CRUD 逻辑及校验规则，各具体后端仅需实现
方言 DDL、驱动加载与幂等冲突写入语句。

@author Ateng
@since 2026-11-11
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp_server_rag.metadata.base import (
    CHUNK_QUERY_BATCH_SIZE,
    CHUNK_SELECT_COLUMNS,
    deduplicate_chunk_ids,
    normalize_chunk_row,
)

logger = logging.getLogger(__name__)

VALID_VISIBILITIES: frozenset[str] = frozenset({"public", "private"})

_KB_SUMMARY_SELECT: str = """
    SELECT kb.name, kb.description, kb.embedding_model, kb.embedding_dimension,
           kb.owner, kb.visibility, kb.created_at, kb.updated_at,
           COUNT(DISTINCT d.id) AS document_count,
           COALESCE(SUM(d.chunk_count), 0) AS chunk_count
    FROM knowledge_bases kb
    LEFT JOIN documents d ON kb.name = d.kb_name
"""

_KB_GROUP_BY: str = (
    "GROUP BY kb.name, kb.description, kb.embedding_model, kb.embedding_dimension, "
    "kb.owner, kb.visibility, kb.created_at, kb.updated_at"
)


class _RelationalMetadataCatalog(ABC):
    """基于 PEP-249 DB-API 接口的企业级关系型元数据目录公共基类。"""

    _backend_name: str = "relational"
    _ddl_statements: tuple[str, ...] = ()

    def __init__(
        self,
        dsn: str,
        connection_factory: Callable[[], Any] | None = None,
        auto_init: bool = True,
    ) -> None:
        """初始化关系型元数据目录并按需校验驱动与初始化 Schema。

        Args:
            dsn: 标准数据库连接字符串。
            connection_factory: 可选外部连接工厂（主要用于测试接缝注入）；
                为 ``None`` 时使用原生驱动。
            auto_init: 是否在初始化阶段自动执行幂等 DDL 建表，默认 ``True``。

        Raises:
            ImportError: 当未提供 ``connection_factory`` 且对应数据库驱动未安装时抛出。
            ValueError: 当 ``dsn`` 为空或结构非法时抛出。
        """
        self._connection_factory = connection_factory
        if self._connection_factory is None:
            self._ensure_driver()

        clean_dsn = dsn.strip()
        if not clean_dsn:
            raise ValueError(
                f"使用 {self._backend_name} 元数据存储时，必须配置 "
                "MCP_RAG_DATABASE_URL 或 MCP_RAG_DB_HOST"
            )
        self.dsn = clean_dsn
        self._validate_dsn()
        if auto_init:
            self._init_schema()

    @abstractmethod
    def _ensure_driver(self) -> Any:
        """惰性导入并校验底层数据库驱动，缺失时抛出包含安装命令的 ImportError。"""

    @abstractmethod
    def _open_connection(self) -> Any:
        """建立并返回一个配置好超时的底层 DB-API 数据库连接对象。"""

    def _validate_dsn(self) -> None:
        """校验 DSN 连接串基础结构，子类可按需覆写加强校验。"""

    @contextmanager
    def _get_connection(self) -> Iterator[Any]:
        """获取数据库连接并在退出上下文时完成提交/回滚与连接关闭。"""
        conn = (
            self._connection_factory()
            if self._connection_factory is not None
            else self._open_connection()
        )
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception as rb_exc:
                logger.debug("%s 回滚事务时发生辅助异常: %s", self._backend_name, rb_exc)
            raise
        finally:
            try:
                conn.close()
            except Exception as close_exc:
                logger.debug("%s 关闭连接时发生辅助异常: %s", self._backend_name, close_exc)

    @contextmanager
    def _get_cursor(self) -> Iterator[Any]:
        """在事务连接内打开并在退出时安全关闭游标。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
            finally:
                if hasattr(cursor, "close"):
                    try:
                        cursor.close()
                    except Exception as close_exc:
                        logger.debug("%s 关闭游标时发生辅助异常: %s", self._backend_name, close_exc)

    @staticmethod
    def _row_to_dict(cursor: Any, row: Any) -> dict[str, Any]:
        """兼容 DictRow 与 PEP-249 元组行，统一转换为普通字典。"""
        if isinstance(row, Mapping):
            return dict(row)
        description = getattr(cursor, "description", None) or ()
        columns = [str(col[0]) for col in description]
        return dict(zip(columns, row))

    @staticmethod
    def _normalize_kb_row(data: dict[str, Any]) -> dict[str, Any]:
        """标准化知识库统计字段类型（如将 MySQL Decimal 聚合值转为 int）。"""
        data["embedding_dimension"] = int(data.get("embedding_dimension") or 0)
        data["document_count"] = int(data.get("document_count") or 0)
        data["chunk_count"] = int(data.get("chunk_count") or 0)
        return data

    @staticmethod
    def _normalize_doc_row(data: dict[str, Any]) -> dict[str, Any]:
        """标准化文档数值字段类型。"""
        data["chunk_count"] = int(data.get("chunk_count") or 0)
        data["file_size"] = int(data.get("file_size") or 0)
        return data

    @staticmethod
    def _normalize_asset_row(data: dict[str, Any]) -> dict[str, Any]:
        """标准化图片资产数值字段类型。"""
        data["byte_size"] = int(data.get("byte_size") or 0)
        return data

    def _init_schema(self) -> None:
        """幂等初始化目标关系型数据库全套元数据表结构。"""
        with self._get_cursor() as cursor:
            for ddl in self._ddl_statements:
                cursor.execute(ddl)

    @abstractmethod
    def _insert_kb_row(self, cursor: Any, params: tuple[Any, ...]) -> bool:
        """插入知识库记录；若同名主键冲突则返回 ``False``，成功返回 ``True``。"""

    @abstractmethod
    def _upsert_document(self, cursor: Any, params: tuple[Any, ...]) -> None:
        """按方言幂等写入或更新文档主记录。"""

    @abstractmethod
    def _upsert_image_asset(self, cursor: Any, params: tuple[Any, ...]) -> None:
        """按方言幂等写入或更新图片资产记录。"""

    def create_kb(
        self,
        name: str,
        description: str = "",
        embedding_model: str = "",
        embedding_dimension: int = 0,
        owner: str = "default",
        visibility: str = "public",
    ) -> bool:
        """创建新知识库元数据记录。"""
        normalized_visibility = visibility.strip().lower()
        if normalized_visibility not in VALID_VISIBILITIES:
            raise ValueError(
                f"非法的知识库可见性 visibility='{visibility}'，仅支持 'public' 或 'private'"
            )
        normalized_owner = owner.strip() or "default"
        now = datetime.now(timezone.utc).isoformat()
        params = (
            name,
            description,
            embedding_model,
            int(embedding_dimension or 0),
            normalized_owner,
            normalized_visibility,
            now,
            now,
        )
        with self._get_cursor() as cursor:
            return self._insert_kb_row(cursor, params)

    def get_kb(self, name: str) -> dict[str, Any] | None:
        """按名称查找知识库。未找到时返回 ``None``。"""
        with self._get_cursor() as cursor:
            cursor.execute(
                f"""
                {_KB_SUMMARY_SELECT}
                WHERE kb.name = %s
                {_KB_GROUP_BY}
                """,
                (name,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._normalize_kb_row(self._row_to_dict(cursor, row))

    def list_kbs(self) -> list[dict[str, Any]]:
        """获取所有知识库列表与切片统计。"""
        with self._get_cursor() as cursor:
            cursor.execute(
                f"""
                {_KB_SUMMARY_SELECT}
                {_KB_GROUP_BY}
                ORDER BY kb.created_at ASC
                """
            )
            return [
                self._normalize_kb_row(self._row_to_dict(cursor, row)) for row in cursor.fetchall()
            ]

    def clear_kb(self, name: str) -> bool:
        """清空指定知识库的所有文档、切片与关联图片资产记录。"""
        if not self.get_kb(name):
            return False
        now = datetime.now(timezone.utc).isoformat()
        with self._get_cursor() as cursor:
            cursor.execute("DELETE FROM image_assets WHERE kb_name = %s", (name,))
            cursor.execute("DELETE FROM chunks WHERE kb_name = %s", (name,))
            cursor.execute("DELETE FROM documents WHERE kb_name = %s", (name,))
            cursor.execute(
                "UPDATE knowledge_bases SET updated_at = %s WHERE name = %s",
                (now, name),
            )
            return True

    def update_kb_embedding_model(
        self,
        name: str,
        embedding_model: str,
        embedding_dimension: int = 0,
    ) -> None:
        """更新知识库绑定的嵌入模型标识与维度。"""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_cursor() as cursor:
            cursor.execute(
                """
                UPDATE knowledge_bases
                SET embedding_model = %s, embedding_dimension = %s, updated_at = %s
                WHERE name = %s
                """,
                (embedding_model, int(embedding_dimension or 0), now, name),
            )

    def delete_kb(self, name: str) -> bool:
        """物理删除指定知识库及其关联的文档、切片与图片资产记录。"""
        if not self.get_kb(name):
            return False
        with self._get_cursor() as cursor:
            cursor.execute("DELETE FROM image_assets WHERE kb_name = %s", (name,))
            cursor.execute("DELETE FROM chunks WHERE kb_name = %s", (name,))
            cursor.execute("DELETE FROM documents WHERE kb_name = %s", (name,))
            cursor.execute("DELETE FROM knowledge_bases WHERE name = %s", (name,))
            return True

    def get_document_by_title(self, kb_name: str, title: str) -> dict[str, Any] | None:
        """根据知识库名称与文档标题查找文档记录。"""
        with self._get_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, kb_name, title, content_hash, source_uri, chunk_count, file_size,
                       created_at, updated_at
                FROM documents
                WHERE kb_name = %s AND title = %s
                """,
                (kb_name, title),
            )
            row = cursor.fetchone()
            return self._normalize_doc_row(self._row_to_dict(cursor, row)) if row else None

    def get_document_by_id(self, kb_name: str, doc_id: str) -> dict[str, Any] | None:
        """根据文档唯一 ID 查找文档记录。"""
        with self._get_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, kb_name, title, content_hash, source_uri, chunk_count, file_size,
                       created_at, updated_at
                FROM documents
                WHERE kb_name = %s AND id = %s
                """,
                (kb_name, doc_id),
            )
            row = cursor.fetchone()
            return self._normalize_doc_row(self._row_to_dict(cursor, row)) if row else None

    def find_document(self, kb_name: str, doc_identifier: str) -> dict[str, Any] | None:
        """多策略查找文档：支持按文档 ID、标题、源路径智能匹配。"""
        clean_id = doc_identifier.strip()
        if not clean_id:
            return None

        doc = self.get_document_by_id(kb_name, clean_id)
        if doc:
            return doc

        try:
            resolved_str = str(Path(clean_id).resolve())
            doc = self.get_document_by_id(kb_name, resolved_str)
            if doc:
                return doc
        except Exception as exc:
            logger.debug("按路径解析文档标识 '%s' 时跳过路径匹配: %s", clean_id, exc)

        with self._get_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, kb_name, title, content_hash, source_uri, chunk_count, file_size,
                       created_at, updated_at
                FROM documents
                WHERE kb_name = %s AND (source_uri = %s OR title = %s)
                """,
                (kb_name, clean_id, clean_id),
            )
            row = cursor.fetchone()
            return self._normalize_doc_row(self._row_to_dict(cursor, row)) if row else None

    def delete_document(self, kb_name: str, doc_id: str) -> list[str]:
        """删除指定文档及其切片与图片资产记录，返回被删除的切片 ID 列表。"""
        with self._get_cursor() as cursor:
            cursor.execute(
                "SELECT id FROM chunks WHERE kb_name = %s AND doc_id = %s",
                (kb_name, doc_id),
            )
            chunk_ids = [str(self._row_to_dict(cursor, row)["id"]) for row in cursor.fetchall()]

            cursor.execute(
                "DELETE FROM image_assets WHERE kb_name = %s AND doc_id = %s",
                (kb_name, doc_id),
            )
            cursor.execute(
                "DELETE FROM chunks WHERE kb_name = %s AND doc_id = %s",
                (kb_name, doc_id),
            )
            cursor.execute(
                "DELETE FROM documents WHERE kb_name = %s AND id = %s",
                (kb_name, doc_id),
            )
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                "UPDATE knowledge_bases SET updated_at = %s WHERE name = %s",
                (now, kb_name),
            )
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
        """原子注册或更新文档与其有序切片链表记录。"""
        now = datetime.now(timezone.utc).isoformat()
        total_chunks = len(chunk_ids)
        with self._get_cursor() as cursor:
            self._upsert_document(
                cursor,
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
            cursor.execute(
                "DELETE FROM chunks WHERE kb_name = %s AND doc_id = %s",
                (kb_name, doc_id),
            )
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
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            cursor.execute(
                "UPDATE knowledge_bases SET updated_at = %s WHERE name = %s",
                (now, kb_name),
            )

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
        with self._get_cursor() as cursor:
            for start in range(0, len(unique_ids), CHUNK_QUERY_BATCH_SIZE):
                batch = unique_ids[start : start + CHUNK_QUERY_BATCH_SIZE]
                placeholders = ", ".join("%s" for _ in batch)
                cursor.execute(
                    f"""
                    SELECT {CHUNK_SELECT_COLUMNS}
                    FROM chunks
                    WHERE kb_name = %s AND id IN ({placeholders})
                    ORDER BY doc_id ASC, chunk_index ASC
                    """,
                    (kb_name, *batch),
                )
                results.extend(
                    normalize_chunk_row(self._row_to_dict(cursor, row)) for row in cursor.fetchall()
                )
        return results

    def get_chunks_by_doc(
        self,
        kb_name: str,
        doc_id: str,
    ) -> list[dict[str, Any]]:
        """按文档 ID 查询全部有序切片记录（按 ``chunk_index`` 升序）。"""
        with self._get_cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {CHUNK_SELECT_COLUMNS}
                FROM chunks
                WHERE kb_name = %s AND doc_id = %s
                ORDER BY chunk_index ASC
                """,
                (kb_name, doc_id),
            )
            return [
                normalize_chunk_row(self._row_to_dict(cursor, row)) for row in cursor.fetchall()
            ]

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
        with self._get_cursor() as cursor:
            for asset in assets:
                filename = str(asset.get("filename") or "").strip()
                storage_uri = str(asset.get("storage_uri") or "").strip()
                if not filename or not storage_uri:
                    raise ValueError("图片资产的 filename 与 storage_uri 不能为空")
                mime_type = str(asset.get("mime_type") or "application/octet-stream")
                byte_size = max(0, int(asset.get("byte_size") or 0))
                created_at = str(asset.get("created_at") or now)
                asset_id = str(asset.get("asset_id") or uuid.uuid4())

                cursor.execute(
                    """
                    DELETE FROM image_assets
                    WHERE kb_name = %s AND doc_id = %s AND filename = %s AND asset_id != %s
                    """,
                    (kb_name, doc_id, filename, asset_id),
                )
                self._upsert_image_asset(
                    cursor,
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

    def get_image_asset(
        self,
        kb_name: str,
        doc_id: str,
        filename: str,
    ) -> dict[str, Any] | None:
        """按文档与文件名查询单条图片资产记录。"""
        with self._get_cursor() as cursor:
            cursor.execute(
                """
                SELECT asset_id, kb_name, doc_id, filename, storage_uri,
                       mime_type, byte_size, created_at
                FROM image_assets
                WHERE kb_name = %s AND doc_id = %s AND filename = %s
                """,
                (kb_name, doc_id, filename),
            )
            row = cursor.fetchone()
            return self._normalize_asset_row(self._row_to_dict(cursor, row)) if row else None

    def list_image_assets(
        self,
        kb_name: str,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出知识库或指定文档的图片资产清单。"""
        with self._get_cursor() as cursor:
            if doc_id is not None:
                cursor.execute(
                    """
                    SELECT asset_id, kb_name, doc_id, filename, storage_uri,
                           mime_type, byte_size, created_at
                    FROM image_assets
                    WHERE kb_name = %s AND doc_id = %s
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
                    WHERE kb_name = %s
                    ORDER BY created_at ASC, filename ASC
                    """,
                    (kb_name,),
                )
            return [
                self._normalize_asset_row(self._row_to_dict(cursor, row))
                for row in cursor.fetchall()
            ]

    def count_kbs(self) -> int:
        """统计知识库总数。"""
        with self._get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS cnt FROM knowledge_bases")
            row = cursor.fetchone()
            if row is None:
                return 0
            if isinstance(row, Mapping):
                return int(row["cnt"])
            return int(row[0])

    def count_documents(self, kb_name: str | None = None) -> int:
        """统计文档总数。"""
        with self._get_cursor() as cursor:
            if kb_name:
                cursor.execute(
                    "SELECT COUNT(*) AS cnt FROM documents WHERE kb_name = %s",
                    (kb_name,),
                )
            else:
                cursor.execute("SELECT COUNT(*) AS cnt FROM documents")
            row = cursor.fetchone()
            if row is None:
                return 0
            if isinstance(row, Mapping):
                return int(row["cnt"])
            return int(row[0])

    def ping(self) -> bool:
        """测试外部关系型数据库健康连通性。"""
        try:
            with self._get_cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
        except Exception as exc:
            logger.warning("%s 元数据目录健康检查失败: %s", self._backend_name, exc)
            return False
