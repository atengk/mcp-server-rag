"""
端到端与接缝测试：企业级 PostgreSQL 与 MySQL 元数据目录适配器接入 (Issue #14)。

在最外层数据库驱动连接边界与 FastMCP 工具边界验证：
1. pyproject.toml 可选依赖声明（[postgres] / [mysql]）及缺失驱动时清晰 ImportError 引导；
2. 服务端工厂根据 MCP_RAG_METADATA_STORE + MCP_RAG_DATABASE_URL / MCP_RAG_DB_* 实例化对应适配器；
3. PostgreSQL 与 MySQL 适配器幂等 DDL 建表方言、CRUD 契约等价性、事务回滚与级联清理；
4. FastMCP 工具（rag_doctor/rag_create_kb/rag_list_kbs/rag_index_text/rag_search）端到端切换。

@author Ateng
@since 2026-11-11
"""

import os
import re
import shutil
import sqlite3
import sys
import tempfile
from collections.abc import Generator
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mcp_server_rag.config import Settings, get_settings
from mcp_server_rag.metadata import (
    MetadataCatalogAdapter,
    MySQLMetadataCatalog,
    PostgresMetadataCatalog,
    get_metadata_catalog,
)
from mcp_server_rag.metadata.relational import _RelationalMetadataCatalog
from mcp_server_rag.server import (
    rag_clear_kb,
    rag_create_kb,
    rag_delete_document,
    rag_delete_kb,
    rag_doctor,
    rag_index_text,
    rag_list_kbs,
    rag_search,
)


@pytest.fixture(autouse=True)
def isolated_enterprise_env() -> Generator[str, None, None]:
    """为每个测试用例隔离临时数据目录并完整备份恢复环境变量。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_ent_meta_")
    env_keys = [
        "MCP_RAG_DATA_DIR",
        "MCP_RAG_METADATA_STORE",
        "MCP_RAG_DATABASE_URL",
        "MCP_RAG_DB_HOST",
        "MCP_RAG_DB_PORT",
        "MCP_RAG_DB_USER",
        "MCP_RAG_DB_PASSWORD",
        "MCP_RAG_DB_NAME",
        "MCP_RAG_EMBEDDING_PROVIDER",
        "MCP_RAG_EMBEDDING_MODEL",
    ]
    backup = {k: os.environ.get(k) for k in env_keys}
    os.environ["MCP_RAG_DATA_DIR"] = temp_dir
    for key in (
        "MCP_RAG_METADATA_STORE",
        "MCP_RAG_DATABASE_URL",
        "MCP_RAG_DB_HOST",
        "MCP_RAG_DB_PORT",
        "MCP_RAG_DB_USER",
        "MCP_RAG_DB_PASSWORD",
        "MCP_RAG_DB_NAME",
    ):
        os.environ.pop(key, None)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    yield temp_dir

    for k, v in backup.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_missing_driver_raises_informative_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证未安装可选驱动时，PostgreSQL 与 MySQL 适配器均主动抛出清晰 ImportError 安装指引。"""
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(ImportError) as pg_exc:
        PostgresMetadataCatalog(dsn="postgresql://rag:secret@localhost:5432/rag_db")
    pg_msg = str(pg_exc.value)
    assert "atengk-mcp-server-rag[postgres]" in pg_msg
    assert "psycopg" in pg_msg

    with pytest.raises(ImportError) as factory_pg_exc:
        os.environ["MCP_RAG_METADATA_STORE"] = "postgresql"
        os.environ["MCP_RAG_DB_HOST"] = "pg.local"
        os.environ["MCP_RAG_DB_NAME"] = "team_rag"
        get_metadata_catalog(get_settings())
    assert "atengk-mcp-server-rag[postgres]" in str(factory_pg_exc.value)

    monkeypatch.setitem(sys.modules, "pymysql", None)
    with pytest.raises(ImportError) as mysql_exc:
        MySQLMetadataCatalog(dsn="mysql://rag:secret@localhost:3306/rag_db")
    mysql_msg = str(mysql_exc.value)
    assert "atengk-mcp-server-rag[mysql]" in mysql_msg
    assert "pymysql" in mysql_msg

    with pytest.raises(ImportError) as factory_mysql_exc:
        get_metadata_catalog(
            Settings(
                metadata_store="mysql",
                database_url="mysql://rag:secret@mysql.local:3306/team_rag",
            )
        )
    assert "atengk-mcp-server-rag[mysql]" in str(factory_mysql_exc.value)

    # 额外校验：pyproject.toml 声明了轻量可选依赖组 postgres 与 mysql
    pyproject_text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text("utf-8")
    assert "postgres = [" in pyproject_text
    assert "mysql = [" in pyproject_text


class _FakeCursor:
    """记录原始 SQL/参数并桥接至内存 SQLite 执行真实验证的测试游标。"""

    def __init__(self, harness: "_FakeDBHarness") -> None:
        self._harness = harness
        self._inner = harness.sqlite_conn.cursor()
        self.closed = False

    @property
    def description(self) -> Any:
        return self._inner.description

    @property
    def rowcount(self) -> int:
        return self._inner.rowcount

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        bound = tuple(params or ())
        self._harness.executed_queries.append((sql.strip(), bound))
        if self._harness.fail_next_execute is not None:
            exc = self._harness.fail_next_execute
            self._harness.fail_next_execute = None
            raise exc
        translated = self._harness.translate_sql(sql)
        if translated:
            self._inner.execute(translated, bound)

    def fetchone(self) -> Any:
        return self._inner.fetchone()

    def fetchall(self) -> list[Any]:
        return self._inner.fetchall()

    def close(self) -> None:
        self.closed = True
        self._inner.close()


class _FakeConnection:
    """追踪 commit/rollback/close 状态的最外层 DB-API 连接桩。"""

    def __init__(self, harness: "_FakeDBHarness") -> None:
        self._harness = harness
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._harness)

    def commit(self) -> None:
        self.committed = True
        self._harness.sqlite_conn.commit()

    def rollback(self) -> None:
        self.rolled_back = True
        self._harness.sqlite_conn.rollback()

    def close(self) -> None:
        self.closed = True


class _FakeDBHarness:
    """支持 PostgreSQL 与 MySQL 方言 SQL 记录与真实 SQLite 执行校验的连接夹具。"""

    def __init__(self) -> None:
        self.sqlite_conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.sqlite_conn.execute("PRAGMA foreign_keys = ON")
        self.executed_queries: list[tuple[str, tuple[Any, ...]]] = []
        self.connections: list[_FakeConnection] = []
        self.fail_next_execute: Exception | None = None

    def connect(self, *_args: Any, **_kwargs: Any) -> _FakeConnection:
        conn = _FakeConnection(self)
        self.connections.append(conn)
        return conn

    @staticmethod
    def translate_sql(sql: str) -> str:
        """将 PG/MySQL DDL 与参数占位符转译为可执行等价 SQLite 语句。"""
        text = sql.strip()
        if text.upper().startswith("CREATE TABLE"):
            text = re.sub(
                r"\)\s*ENGINE=InnoDB.*$",
                ")",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            text = re.sub(r"\bLONGTEXT\b", "TEXT", text, flags=re.IGNORECASE)
            # 清理 MySQL 建表内联 INDEX 定义，交由底层列与外键约束执行校验
            lines = [
                line
                for line in text.splitlines()
                if not re.match(r"^\s*INDEX\s+\w+\s*\(", line, flags=re.IGNORECASE)
            ]
            text = "\n".join(lines)

        text = text.replace("%s", "?")
        if "ON DUPLICATE KEY UPDATE" in text:
            if "INTO documents" in text:
                text = text.replace("ON DUPLICATE KEY UPDATE", "ON CONFLICT(id) DO UPDATE SET")
            elif "INTO image_assets" in text:
                text = text.replace(
                    "ON DUPLICATE KEY UPDATE",
                    "ON CONFLICT(asset_id) DO UPDATE SET",
                )
        return text

    def close(self) -> None:
        self.sqlite_conn.close()


def _assert_crud_parity(catalog: MetadataCatalogAdapter, harness: _FakeDBHarness) -> None:
    """对任意遵循 MetadataCatalogAdapter 协议的后端执行全生命周期等价断言。"""
    assert isinstance(catalog, MetadataCatalogAdapter)
    assert catalog.ping() is True

    # 1. 创建 KB（含重复名称与非法 visibility 校验）
    assert catalog.create_kb(
        name="corp_kb",
        description="企业知识库",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=1024,
        owner="alice",
        visibility="private",
    )
    assert (
        catalog.create_kb(
            name="corp_kb",
            description="重复名称",
            owner="bob",
            visibility="public",
        )
        is False
    )
    with pytest.raises(ValueError, match="visibility"):
        catalog.create_kb(name="bad_kb", visibility="restricted")

    kb = catalog.get_kb("corp_kb")
    assert kb is not None
    assert kb["owner"] == "alice"
    assert kb["visibility"] == "private"
    assert kb["embedding_dimension"] == 1024
    assert isinstance(kb["document_count"], int) and kb["document_count"] == 0

    catalog.update_kb_embedding_model("corp_kb", "text-embedding-3-large", 3072)
    updated_kb = catalog.get_kb("corp_kb")
    assert updated_kb is not None
    assert updated_kb["embedding_model"] == "text-embedding-3-large"
    assert updated_kb["embedding_dimension"] == 3072

    # 2. 注册文档与有序切片链表 + 缩减再索引旧切片清理
    catalog.register_document_and_chunks(
        kb_name="corp_kb",
        doc_id="doc_1",
        title="架构白皮书",
        content_hash="sha_1",
        source_uri="docs/whitepaper.pdf",
        chunk_ids=["c1", "c2", "c3"],
        file_size=4096,
        chunk_contents=["第一章概述", "第二章设计", "第三章总结"],
        chunk_images=[["fig1.png"], [], ["fig2.png"]],
    )
    assert catalog.count_kbs() == 1
    assert catalog.count_documents("corp_kb") == 1
    doc_found = catalog.find_document("corp_kb", "架构白皮书")
    assert doc_found is not None
    assert doc_found["chunk_count"] == 3
    assert doc_found["file_size"] == 4096

    chunk_rows = harness.sqlite_conn.execute(
        "SELECT id, content, prev_chunk_id, next_chunk_id, images_json "
        "FROM chunks WHERE doc_id = 'doc_1' ORDER BY chunk_index ASC"
    ).fetchall()
    assert len(chunk_rows) == 3
    assert chunk_rows[0] == ("c1", "第一章概述", None, "c2", '["fig1.png"]')
    assert chunk_rows[1] == ("c2", "第二章设计", "c1", "c3", "[]")
    assert chunk_rows[2] == ("c3", "第三章总结", "c2", None, '["fig2.png"]')

    # 再索引缩减切片数应清理旧 c3
    catalog.register_document_and_chunks(
        kb_name="corp_kb",
        doc_id="doc_1",
        title="架构白皮书",
        content_hash="sha_2",
        source_uri="docs/whitepaper.pdf",
        chunk_ids=["c1", "c2"],
        file_size=2048,
        chunk_contents=["精简首章", "精简末章"],
        chunk_images=[[], []],
    )
    assert (
        harness.sqlite_conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = 'doc_1'"
        ).fetchone()[0]
        == 2
    )

    # 3. image_assets 登记、同名冲突/同 ID 幂等更新与级联删除
    catalog.register_image_assets(
        kb_name="corp_kb",
        doc_id="doc_1",
        assets=[
            {
                "asset_id": "img_1",
                "filename": "fig1.png",
                "storage_uri": "rag://assets/corp_kb/doc_1/fig1.png",
                "mime_type": "image/png",
                "byte_size": 8192,
            }
        ],
    )
    # 冲突更新验证：相同 asset_id 触发 UPSERT，同名更换 asset_id 覆盖旧记录
    catalog.register_image_assets(
        kb_name="corp_kb",
        doc_id="doc_1",
        assets=[
            {
                "asset_id": "img_1",
                "filename": "fig1.png",
                "storage_uri": "rag://assets/corp_kb/doc_1/fig1.png",
                "mime_type": "image/png",
                "byte_size": 9000,
            }
        ],
    )
    catalog.register_image_assets(
        kb_name="corp_kb",
        doc_id="doc_1",
        assets=[
            {
                "asset_id": "img_1_v2",
                "filename": "fig1.png",
                "storage_uri": "rag://assets/corp_kb/doc_1/fig1.png",
                "mime_type": "image/webp",
                "byte_size": 4096,
            }
        ],
    )
    asset = catalog.get_image_asset("corp_kb", "doc_1", "fig1.png")
    assert asset is not None
    assert asset["asset_id"] == "img_1_v2"
    assert asset["mime_type"] == "image/webp"
    assert asset["byte_size"] == 4096
    assert len(catalog.list_image_assets("corp_kb")) == 1

    # 验证 MySQL Decimal 聚合统计字段（如 SUM 返回 Decimal）被归一化为原生 int
    normalized = _RelationalMetadataCatalog._normalize_kb_row(
        {
            "name": "corp_kb",
            "embedding_dimension": Decimal("1024"),
            "document_count": 2,
            "chunk_count": Decimal("15"),
        }
    )
    assert (
        normalized["embedding_dimension"] == 1024 and type(normalized["embedding_dimension"]) is int
    )
    assert normalized["chunk_count"] == 15 and type(normalized["chunk_count"]) is int

    deleted_chunks = catalog.delete_document("corp_kb", "doc_1")
    assert deleted_chunks == ["c1", "c2"]
    assert catalog.list_image_assets("corp_kb") == []
    assert catalog.get_document_by_id("corp_kb", "doc_1") is None

    # 4. clear_kb 与 delete_kb 级联校验
    catalog.register_document_and_chunks(
        kb_name="corp_kb",
        doc_id="doc_2",
        title="运维手册",
        content_hash="sha_ops",
        source_uri="docs/ops.md",
        chunk_ids=["c_ops_1"],
        file_size=512,
        chunk_contents=["部署说明"],
    )
    catalog.register_image_assets(
        kb_name="corp_kb",
        doc_id="doc_2",
        assets=[
            {
                "asset_id": "img_ops",
                "filename": "topo.png",
                "storage_uri": "rag://assets/corp_kb/doc_2/topo.png",
            }
        ],
    )
    assert catalog.clear_kb("corp_kb") is True
    assert catalog.count_documents("corp_kb") == 0
    assert catalog.list_image_assets("corp_kb") == []
    assert catalog.delete_kb("corp_kb") is True
    assert catalog.list_kbs() == []

    # 5. 验证所有连接用后均已关闭且提交，异常触发回滚并关连接
    assert all(c.closed for c in harness.connections)
    assert any(c.committed for c in harness.connections)

    harness.fail_next_execute = RuntimeError("simulated sql boom")
    with pytest.raises(RuntimeError, match="simulated sql boom"):
        catalog.create_kb(name="rollback_kb")
    assert harness.connections[-1].rolled_back is True
    assert harness.connections[-1].closed is True

    harness.fail_next_execute = RuntimeError("db unavailable")
    assert catalog.ping() is False


def test_postgres_adapter_ddl_dialect_and_crud_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 PostgresMetadataCatalog 的 DDL 方言、超时配置、ON CONFLICT 契约及全生命周期。"""
    with closing(_FakeDBHarness()) as harness:
        pg_connect_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def fake_pg_connect(*args: Any, **kwargs: Any) -> _FakeConnection:
            pg_connect_calls.append((args, kwargs))
            return harness.connect()

        monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=fake_pg_connect))
        catalog = PostgresMetadataCatalog(
            dsn="postgresql://rag_admin:secret@pg.internal:5432/team_rag",
        )

        assert len(pg_connect_calls) >= 1
        assert pg_connect_calls[0][1]["connect_timeout"] == 10
        assert "statement_timeout=30000" in pg_connect_calls[0][1]["options"]

        # 验证幂等 DDL 建表语句与 PostgreSQL 索引/外键/冲突方言
        ddl_sql = "\n".join(sql for sql, _ in harness.executed_queries[:6])
        for table in ("knowledge_bases", "documents", "chunks", "image_assets"):
            assert f"CREATE TABLE IF NOT EXISTS {table}" in ddl_sql
        assert "FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE" in ddl_sql
        assert "CREATE INDEX IF NOT EXISTS idx_chunks_kb_doc_idx" in ddl_sql
        assert "CREATE INDEX IF NOT EXISTS idx_image_assets_kb_doc" in ddl_sql

        _assert_crud_parity(catalog, harness)

        all_sql = "\n".join(sql for sql, _ in harness.executed_queries)
        assert "ON CONFLICT (name) DO NOTHING" in all_sql
        assert "ON CONFLICT (id) DO UPDATE SET" in all_sql
        assert "ON CONFLICT (asset_id) DO UPDATE SET" in all_sql


def test_mysql_adapter_ddl_dialect_dsn_and_crud_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 MySQLMetadataCatalog 的 UTF8MB4 DDL、DSN 解码、ON DUPLICATE KEY UPDATE 与生命周期。"""
    # 1. 验证非法 DSN 密码脱敏与 scheme 校验
    fake_pymysql = SimpleNamespace(connect=lambda **_kwargs: None)
    monkeypatch.setitem(sys.modules, "pymysql", fake_pymysql)
    with pytest.raises(ValueError) as dsn_err:
        MySQLMetadataCatalog(dsn="mysql://rag_admin:TopSecretPass99@host:3306/", auto_init=False)
    assert "TopSecretPass99" not in str(dsn_err.value)
    assert "***" in str(dsn_err.value)

    with pytest.raises(ValueError, match="无效的 MySQL 连接字符串"):
        MySQLMetadataCatalog(dsn="postgresql://rag:pass@host:5432/db", auto_init=False)

    # 2. 验证特殊字符 URL 解码及传给 pymysql.connect 的关键字参数
    with closing(_FakeDBHarness()) as harness:
        captured_connect_kwargs: list[dict[str, Any]] = []

        def fake_pymysql_connect(**kwargs: Any) -> _FakeConnection:
            captured_connect_kwargs.append(kwargs)
            return harness.connect()

        mock_pymysql = SimpleNamespace(connect=fake_pymysql_connect)
        monkeypatch.setitem(sys.modules, "pymysql", mock_pymysql)
        catalog = MySQLMetadataCatalog(
            dsn="mysql://rag_admin:p%40ss%3Aword%23123%204@mysql.corp.local:3307/team%20rag"
        )

        assert len(captured_connect_kwargs) >= 1
        conn_kwargs = captured_connect_kwargs[0]
        assert conn_kwargs["host"] == "mysql.corp.local"
        assert conn_kwargs["port"] == 3307
        assert conn_kwargs["user"] == "rag_admin"
        assert conn_kwargs["password"] == "p@ss:word#123 4"
        assert conn_kwargs["database"] == "team rag"
        assert conn_kwargs["charset"] == "utf8mb4"
        assert conn_kwargs["connect_timeout"] == 10

        ddl_sql = "\n".join(sql for sql, _ in harness.executed_queries[:4])
        for table in ("knowledge_bases", "documents", "chunks", "image_assets"):
            assert f"CREATE TABLE IF NOT EXISTS {table}" in ddl_sql
        assert "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci" in ddl_sql
        assert "LONGTEXT" in ddl_sql
        assert "INDEX idx_chunks_kb_doc_idx" in ddl_sql
        assert "INDEX idx_image_assets_kb_doc" in ddl_sql

        _assert_crud_parity(catalog, harness)

        all_sql = "\n".join(sql for sql, _ in harness.executed_queries)
        assert "ON DUPLICATE KEY UPDATE" in all_sql
        assert "VALUES(content_hash)" not in all_sql

        # 3. 非 1062 的外键/非空异常不应被 _insert_kb_row 吞没
        class FakeMySQLIntegrityError(Exception):
            pass

        harness.fail_next_execute = FakeMySQLIntegrityError(
            1452, "Cannot add or update a child row: a foreign key constraint fails"
        )
        with pytest.raises(FakeMySQLIntegrityError):
            catalog.create_kb(name="fk_error_kb")


@pytest.mark.parametrize("backend", ["postgresql", "mysql"])
def test_server_e2e_postgres_and_mysql_backends(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 FastMCP 服务端通过环境变量平滑切换 PostgreSQL/MySQL 并驱动核心 MCP 工具。"""
    with closing(_FakeDBHarness()) as harness:
        os.environ["MCP_RAG_METADATA_STORE"] = backend
        os.environ["MCP_RAG_DB_HOST"] = f"{backend}.cluster.local"
        os.environ["MCP_RAG_DB_USER"] = "mcp_user"
        os.environ["MCP_RAG_DB_PASSWORD"] = "sec@ret"
        os.environ["MCP_RAG_DB_NAME"] = "shared_rag"

        fake_driver = SimpleNamespace(connect=harness.connect)
        driver_name = "psycopg" if backend == "postgresql" else "pymysql"
        monkeypatch.setitem(sys.modules, driver_name, fake_driver)

        doctor = rag_doctor()
        assert doctor["metadata_store"] == backend
        assert doctor["metadata_catalog_status"] == "ok"
        assert doctor["status"] == "healthy"

        kb_name = f"{backend}_team_kb"
        create_res = rag_create_kb(
            name=kb_name,
            description="多实例共享知识库",
            owner="team_lead",
            visibility="private",
        )
        assert create_res["success"] is True

        kbs = {item["name"]: item for item in rag_list_kbs()}
        kb_item = kbs[kb_name]
        assert kb_item["owner"] == "team_lead"
        assert kb_item["visibility"] == "private"
        assert kb_item["embedding_dimension"] > 0

        index_res = rag_index_text(
            kb_name=kb_name,
            text="PostgreSQL 与 MySQL 适配器支撑团队跨节点并发检索与元数据共享。",
            title="team_arch_doc",
        )
        assert index_res["success"] is True
        assert index_res["chunk_count"] >= 1

        search_results = rag_search(
            kb_name=kb_name,
            query="跨节点并发检索与元数据共享",
            top_k=3,
        )
        assert len(search_results) >= 1
        assert search_results[0]["metadata"]["title"] == "team_arch_doc"

        # 验证删文档、清库与删库全生命周期均在目标后端正常工作
        del_doc = rag_delete_document(kb_name=kb_name, doc_id="team_arch_doc")
        assert del_doc["success"] is True
        assert rag_list_kbs()[0]["document_count"] == 0

        clear_res = rag_clear_kb(name=kb_name)
        assert clear_res["success"] is True

        del_kb = rag_delete_kb(name=kb_name)
        assert del_kb["success"] is True
        assert rag_list_kbs() == []

        # 验证整串 MCP_RAG_DATABASE_URL 优先级切换
        explicit_dsn = f"{backend}://url_user:url_pass@override.db.local:15432/override_db"
        os.environ["MCP_RAG_DATABASE_URL"] = explicit_dsn
        catalog = get_metadata_catalog(get_settings())
        assert catalog.dsn == explicit_dsn  # type: ignore[attr-defined]
