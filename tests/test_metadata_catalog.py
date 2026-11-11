"""
端到端与接缝测试：可插拔元数据目录抽象、拆分式 DB 配置与协作契约扩展 (Issue #13)。

在最高接缝（FastMCP 工具边界、配置解析契约与 MetadataCatalogAdapter 协议边界）验证：
1. 拆分式 MCP_RAG_DB_* 变量拼装与 MCP_RAG_DATABASE_URL 优先级覆盖；
2. 默认 SQLite 适配器遵循 MetadataCatalogAdapter 协议与 rag_doctor 展示；
3. knowledge_bases 扩展 owner/visibility/embedding_dimension 字段与合法性校验；
4. 存量旧版 SQLite 数据库自动 ALTER TABLE 迁移与 chunks 双向链表落库；
5. image_assets 资产登记表及文档/知识库删除级联清理。

@author Ateng
@since 2026-11-11
"""

import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Generator
from contextlib import closing
from pathlib import Path

import pytest

from mcp_server_rag.config import Settings, get_settings
from mcp_server_rag.metadata import (
    MetadataCatalogAdapter,
    SQLiteMetadataCatalog,
    get_metadata_catalog,
)
from mcp_server_rag.metadata.factory import _resolve_sqlite_db_path
from mcp_server_rag.server import (
    rag_clear_kb,
    rag_create_kb,
    rag_delete_document,
    rag_delete_kb,
    rag_doctor,
    rag_index_text,
    rag_list_kbs,
)


@pytest.fixture(autouse=True)
def isolated_env() -> Generator[str, None, None]:
    """为每个测试用例提供隔离的临时数据目录并恢复全部 DB 相关环境变量。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_meta_")
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
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    yield temp_dir

    for k, v in backup.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_split_db_env_vars_and_url_precedence(isolated_env: str) -> None:
    """验证拆分式 DB 环境变量自动拼装与 MCP_RAG_DATABASE_URL 优先级覆盖。"""
    # 1. PostgreSQL 拆分变量与默认端口 5432、密码特殊字符转义（空格编码为 %20）
    os.environ["MCP_RAG_METADATA_STORE"] = "postgresql"
    os.environ["MCP_RAG_DB_HOST"] = "pg.internal.local"
    os.environ.pop("MCP_RAG_DB_PORT", None)
    os.environ["MCP_RAG_DB_USER"] = "rag_admin"
    os.environ["MCP_RAG_DB_PASSWORD"] = "p@ss:word#123 4"
    os.environ["MCP_RAG_DB_NAME"] = "team_rag"
    os.environ.pop("MCP_RAG_DATABASE_URL", None)

    pg_settings = get_settings()
    assert pg_settings.metadata_store == "postgresql"
    pg_dsn = pg_settings.resolved_database_url()
    assert (
        pg_dsn == "postgresql://rag_admin:p%40ss%3Aword%23123%204@pg.internal.local:5432/team_rag"
    )

    # 2. MySQL 拆分变量与指定端口 3307
    os.environ["MCP_RAG_METADATA_STORE"] = "mysql"
    os.environ["MCP_RAG_DB_HOST"] = "mysql.internal.local"
    os.environ["MCP_RAG_DB_PORT"] = "3307"
    os.environ["MCP_RAG_DB_PASSWORD"] = "p@ss:word#123"
    mysql_settings = get_settings()
    mysql_dsn = mysql_settings.resolved_database_url()
    assert mysql_dsn == "mysql://rag_admin:p%40ss%3Aword%23123@mysql.internal.local:3307/team_rag"

    # 3. 整串 MCP_RAG_DATABASE_URL 拥有最高优先级
    explicit_url = "postgresql://override:secret@db.prod.example.com:6432/prod_rag"
    os.environ["MCP_RAG_DATABASE_URL"] = explicit_url
    override_settings = get_settings()
    assert override_settings.resolved_database_url() == explicit_url

    # 4. 清理所有 MCP_RAG_DB_* 后回退默认 SQLite，并支持显式 sqlite:/// URL 覆盖
    for db_key in (
        "MCP_RAG_DB_HOST",
        "MCP_RAG_DB_PORT",
        "MCP_RAG_DB_USER",
        "MCP_RAG_DB_PASSWORD",
        "MCP_RAG_DB_NAME",
        "MCP_RAG_DATABASE_URL",
    ):
        os.environ.pop(db_key, None)
    os.environ["MCP_RAG_METADATA_STORE"] = ""

    sqlite_settings = get_settings()
    assert sqlite_settings.metadata_store == "sqlite"
    catalog = get_metadata_catalog(sqlite_settings)
    assert isinstance(catalog, MetadataCatalogAdapter)
    assert isinstance(catalog, SQLiteMetadataCatalog)
    assert catalog.db_path == Path(isolated_env) / "metadata.db"

    custom_sqlite = Path(isolated_env) / "custom_meta.db"
    os.environ["MCP_RAG_DATABASE_URL"] = f"sqlite:///{custom_sqlite.as_posix()}"
    custom_catalog = get_metadata_catalog(get_settings())
    assert isinstance(custom_catalog, SQLiteMetadataCatalog)
    assert custom_catalog.db_path == custom_sqlite
    assert custom_sqlite.exists()
    os.environ.pop("MCP_RAG_DATABASE_URL", None)

    posix_resolved = _resolve_sqlite_db_path(
        Settings(database_url="sqlite:////var/lib/rag/meta.db")
    )
    assert posix_resolved == Path("/var/lib/rag/meta.db")

    report = rag_doctor()
    assert report["metadata_store"] == "sqlite"
    assert report["metadata_catalog_status"] == "ok"

    # 5. 非法 metadata_store 应抛出明确异常
    with pytest.raises(ValueError, match="unsupported_db"):
        get_metadata_catalog(Settings(metadata_store="unsupported_db"))

    # 6. PostgreSQL/MySQL 在 #14 交付前应抛出清晰的 NotImplementedError
    with pytest.raises(NotImplementedError):
        get_metadata_catalog(Settings(metadata_store="postgresql"))
    with pytest.raises(NotImplementedError):
        get_metadata_catalog(Settings(metadata_store="mysql"))


def test_kb_owner_visibility_and_embedding_dimension(isolated_env: str) -> None:
    """验证知识库 owner、visibility 与 embedding_dimension 扩展及向后兼容默认行为。"""
    # 1. 不传入 owner/visibility 时保持 100% 向后兼容默认值
    res_default = rag_create_kb(name="legacy_compat_kb", description="默认公开知识库")
    assert res_default["success"] is True

    # 2. 显式传入 owner 与 visibility='private'
    res_private = rag_create_kb(
        name="private_team_kb",
        description="团队私有知识库",
        owner="alice",
        visibility="private",
    )
    assert res_private["success"] is True

    # 3. 非法 visibility 应被拒绝
    res_invalid = rag_create_kb(
        name="bad_vis_kb",
        description="非法可见性",
        owner="bob",
        visibility="internal_only",
    )
    assert res_invalid["success"] is False
    assert "visibility" in res_invalid["message"]

    # 4. rag_list_kbs 返回扩展字段且持久化了实时探测的 embedding_dimension
    kbs = {item["name"]: item for item in rag_list_kbs()}
    assert "bad_vis_kb" not in kbs

    default_kb = kbs["legacy_compat_kb"]
    assert default_kb["owner"] == "default"
    assert default_kb["visibility"] == "public"
    assert isinstance(default_kb["embedding_dimension"], int)
    assert default_kb["embedding_dimension"] > 0

    private_kb = kbs["private_team_kb"]
    assert private_kb["owner"] == "alice"
    assert private_kb["visibility"] == "private"
    assert private_kb["embedding_dimension"] == default_kb["embedding_dimension"]


def test_legacy_sqlite_migration_and_chunks_linked_list(isolated_env: str) -> None:
    """验证存量旧版 SQLite 自动 ALTER TABLE 迁移及 chunks 正文与前后向链表落库。"""
    db_path = Path(isolated_env) / "metadata.db"

    # 1. 手工构造旧版表结构并写入存量数据（缺少扩展字段与 image_assets 表）
    with closing(sqlite3.connect(str(db_path))) as conn, conn:
        conn.execute(
            """
            CREATE TABLE knowledge_bases (
                name TEXT PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                embedding_model TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                kb_name TEXT NOT NULL,
                title TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_uri TEXT NOT NULL DEFAULT '',
                chunk_count INTEGER NOT NULL DEFAULT 0,
                file_size INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                kb_name TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO knowledge_bases
            (name, description, embedding_model, created_at, updated_at)
            VALUES (
                'legacy_kb',
                '旧版知识库',
                '',
                '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00'
            )
            """
        )
        conn.commit()

    # 2. 通过高层工具与目录初始化触发自动迁移（且多次初始化保持幂等）
    kbs = {item["name"]: item for item in rag_list_kbs()}
    assert "legacy_kb" in kbs
    assert kbs["legacy_kb"]["owner"] == "default"
    assert kbs["legacy_kb"]["visibility"] == "public"
    assert kbs["legacy_kb"]["embedding_dimension"] == 0

    SQLiteMetadataCatalog(Path(isolated_env))  # 二次初始化幂等校验

    # 3. 构造超长文本通过 rag_index_text 触发多切片落库
    assert rag_create_kb(name="linked_kb", description="切片链表验证")["success"] is True
    long_paragraphs = [
        f"第{i}段内容：关于分布式检索增强生成架构的多级元数据目录设计说明，涵盖上下文扩展与引用溯源。"
        * 4
        for i in range(1, 10)
    ]
    raw_text = "\n\n".join(long_paragraphs)
    idx_res = rag_index_text(kb_name="linked_kb", text=raw_text, title="arch_notes")
    assert idx_res["success"] is True
    assert idx_res["chunk_count"] >= 2

    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, chunk_index, content, prev_chunk_id, next_chunk_id, images_json
            FROM chunks
            WHERE kb_name = 'linked_kb'
            ORDER BY chunk_index ASC
            """
        ).fetchall()

    assert len(rows) == idx_res["chunk_count"]
    assert rows[0]["prev_chunk_id"] is None
    assert rows[-1]["next_chunk_id"] is None
    for i, row in enumerate(rows):
        assert row["content"].strip() != ""
        assert json.loads(row["images_json"]) == []
        if i > 0:
            assert row["prev_chunk_id"] == rows[i - 1]["id"]
        if i + 1 < len(rows):
            assert row["next_chunk_id"] == rows[i + 1]["id"]

    # 4. 验证 chunk_images JSON 序列化存储
    catalog = SQLiteMetadataCatalog(Path(isolated_env))
    catalog.register_document_and_chunks(
        kb_name="linked_kb",
        doc_id="img_doc",
        title="img_doc",
        content_hash="hash_img_1",
        source_uri="manual://img_doc",
        chunk_ids=["c_a", "c_b"],
        file_size=42,
        chunk_contents=["图一说明", "图二说明"],
        chunk_images=[["fig1.png", "fig2.png"], []],
    )
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        img_rows = conn.execute(
            """
            SELECT id, content, prev_chunk_id, next_chunk_id, images_json
            FROM chunks
            WHERE doc_id = 'img_doc'
            ORDER BY chunk_index ASC
            """
        ).fetchall()
    assert len(img_rows) == 2
    assert img_rows[0]["content"] == "图一说明"
    assert img_rows[0]["prev_chunk_id"] is None
    assert img_rows[0]["next_chunk_id"] == "c_b"
    assert json.loads(img_rows[0]["images_json"]) == ["fig1.png", "fig2.png"]
    assert img_rows[1]["prev_chunk_id"] == "c_a"
    assert img_rows[1]["next_chunk_id"] is None
    assert json.loads(img_rows[1]["images_json"]) == []

    # 5. 直接再索引（缩减切片数量）时应自动清理多余旧切片
    catalog.register_document_and_chunks(
        kb_name="linked_kb",
        doc_id="img_doc",
        title="img_doc",
        content_hash="hash_img_2",
        source_uri="manual://img_doc",
        chunk_ids=["c_only"],
        file_size=18,
        chunk_contents=["唯一更新切片"],
        chunk_images=[[]],
    )
    with closing(sqlite3.connect(str(db_path))) as conn:
        remaining_ids = [
            row[0]
            for row in conn.execute(
                "SELECT id FROM chunks WHERE doc_id = 'img_doc' ORDER BY chunk_index ASC"
            ).fetchall()
        ]
    assert remaining_ids == ["c_only"]


def test_image_assets_lifecycle_and_cascading_delete(isolated_env: str) -> None:
    """验证 image_assets 幂等登记、查询及随文档删除/清空/删库的级联清理。"""
    assert rag_create_kb(name="assets_kb", description="图片资产级联测试")["success"] is True
    assert rag_index_text(kb_name="assets_kb", text="文档A包含架构图。", title="doc_a")["success"]
    assert rag_index_text(kb_name="assets_kb", text="文档B包含流程图。", title="doc_b")["success"]

    catalog = get_metadata_catalog(get_settings())
    catalog.register_image_assets(
        kb_name="assets_kb",
        doc_id="doc_a",
        assets=[
            {
                "asset_id": "asset_a_1",
                "filename": "arch.png",
                "storage_uri": "images/assets_kb/doc_a/arch.png",
                "mime_type": "image/png",
                "byte_size": 2048,
            },
            {
                "asset_id": "asset_a_2",
                "filename": "diagram.jpg",
                "storage_uri": "images/assets_kb/doc_a/diagram.jpg",
                "mime_type": "image/jpeg",
                "byte_size": 4096,
            },
        ],
    )
    # 重复写入同名文件（换新 asset_id）应幂等覆盖而非累积重复行
    catalog.register_image_assets(
        kb_name="assets_kb",
        doc_id="doc_a",
        assets=[
            {
                "asset_id": "asset_a_1_new",
                "filename": "arch.png",
                "storage_uri": "images/assets_kb/doc_a/arch.png",
                "mime_type": "image/png",
                "byte_size": 2050,
            },
        ],
    )
    catalog.register_image_assets(
        kb_name="assets_kb",
        doc_id="doc_b",
        assets=[
            {
                "asset_id": "asset_b_1",
                "filename": "flow.png",
                "storage_uri": "images/assets_kb/doc_b/flow.png",
                "mime_type": "image/png",
                "byte_size": 1024,
            },
        ],
    )

    asset_a1 = catalog.get_image_asset("assets_kb", "doc_a", "arch.png")
    assert asset_a1 is not None
    assert asset_a1["asset_id"] == "asset_a_1_new"
    assert asset_a1["byte_size"] == 2050
    assert len(catalog.list_image_assets("assets_kb", doc_id="doc_a")) == 2
    assert len(catalog.list_image_assets("assets_kb")) == 3

    # 1. 删除文档 doc_a 级联清理 doc_a 资产，保留 doc_b 资产
    del_doc_res = rag_delete_document(kb_name="assets_kb", doc_id="doc_a")
    assert del_doc_res["success"] is True
    assert catalog.get_image_asset("assets_kb", "doc_a", "arch.png") is None
    assert catalog.list_image_assets("assets_kb", doc_id="doc_a") == []
    assert len(catalog.list_image_assets("assets_kb", doc_id="doc_b")) == 1

    # 2. clear_kb 级联清空剩余资产但保留知识库
    clear_res = rag_clear_kb(name="assets_kb")
    assert clear_res["success"] is True
    assert catalog.list_image_assets("assets_kb") == []
    assert catalog.get_kb("assets_kb") is not None

    # 3. delete_kb 级联物理删除全部关联资产记录
    assert rag_create_kb(name="assets_drop_kb", description="删库清理验证")["success"] is True
    assert (
        rag_index_text(kb_name="assets_drop_kb", text="文档C附图。", title="doc_c")["success"]
        is True
    )
    catalog.register_image_assets(
        kb_name="assets_drop_kb",
        doc_id="doc_c",
        assets=[
            {
                "asset_id": "asset_c_1",
                "filename": "c.png",
                "storage_uri": "images/assets_drop_kb/doc_c/c.png",
                "mime_type": "image/png",
                "byte_size": 512,
            }
        ],
    )
    assert len(catalog.list_image_assets("assets_drop_kb")) == 1

    drop_res = rag_delete_kb(name="assets_drop_kb")
    assert drop_res["success"] is True
    db_path = Path(isolated_env) / "metadata.db"
    with closing(sqlite3.connect(str(db_path))) as conn:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM image_assets WHERE kb_name = 'assets_drop_kb'"
        ).fetchone()[0]
    assert remaining == 0

    # 4. image_assets 拒绝空 filename 或 storage_uri
    with pytest.raises(ValueError, match="不能为空"):
        catalog.register_image_assets(
            kb_name="assets_kb",
            doc_id="doc_b",
            assets=[{"filename": "  ", "storage_uri": "images/x.png"}],
        )
