"""
SQLite 元数据目录管理器。

负责维护知识库注册表、文档内容指纹与切片统计元数据的事务性读写。

@author Ateng
@since 2026-11-11
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MetadataCatalog:
    """基于内置 SQLite 的元数据目录管理器。"""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "metadata.db"
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """获取 SQLite 数据库连接并配置 Row 工厂。"""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """初始化元数据数据表结构。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    name TEXT PRIMARY KEY,
                    description TEXT NOT NULL DEFAULT '',
                    embedding_model TEXT NOT NULL DEFAULT '',
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
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            conn.commit()

    def create_kb(
        self,
        name: str,
        description: str = "",
        embedding_model: str = "",
    ) -> bool:
        """创建新知识库记录。若已存在则返回 False。"""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO knowledge_bases
                    (name, description, embedding_model, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (name, description, embedding_model, now, now),
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False

    def get_kb(self, name: str) -> dict[str, Any] | None:
        """按名称查找知识库。未找到时返回 None。"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT kb.name, kb.description, kb.embedding_model, kb.created_at, kb.updated_at,
                       COUNT(DISTINCT d.id) AS document_count,
                       COALESCE(SUM(d.chunk_count), 0) AS chunk_count
                FROM knowledge_bases kb
                LEFT JOIN documents d ON kb.name = d.kb_name
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
                """
                SELECT kb.name, kb.description, kb.embedding_model, kb.created_at, kb.updated_at,
                       COUNT(DISTINCT d.id) AS document_count,
                       COALESCE(SUM(d.chunk_count), 0) AS chunk_count
                FROM knowledge_bases kb
                LEFT JOIN documents d ON kb.name = d.kb_name
                GROUP BY kb.name
                ORDER BY kb.created_at ASC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def clear_kb(self, name: str) -> bool:
        """清空指定知识库的所有文档与切片记录。"""
        if not self.get_kb(name):
            return False
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE kb_name = ?", (name,))
            cursor.execute("DELETE FROM documents WHERE kb_name = ?", (name,))
            cursor.execute(
                "UPDATE knowledge_bases SET updated_at = ? WHERE name = ?",
                (now, name),
            )
            conn.commit()
            return True

    def delete_kb(self, name: str) -> bool:
        """物理删除指定知识库及其关联数据。"""
        if not self.get_kb(name):
            return False
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE kb_name = ?", (name,))
            cursor.execute("DELETE FROM documents WHERE kb_name = ?", (name,))
            cursor.execute("DELETE FROM knowledge_bases WHERE name = ?", (name,))
            conn.commit()
            return True

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
        except Exception:
            return False
