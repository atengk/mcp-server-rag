"""
原生数据库 FTS 倒排索引与 RRF 向量混合检索全链路测试。

遵循 TDD 红绿重构契约，全面覆盖中英混合分词、SQLite FTS5 原生索引与事务性清理、
PostgreSQL tsvector/GIN 与 MySQL FULLTEXT/ngram 方言 SQL 契约，
以及 rag_search / rag_query_with_context 端到端 RRF 混合检索行为。

@author Ateng
@since 2026-11-11
"""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from mcp_server_rag import server
from mcp_server_rag.embedding import EmbeddingProvider
from mcp_server_rag.fts import (
    build_mysql_ft_query,
    build_postgres_tsquery,
    build_sqlite_fts5_query,
    extract_fts_query_terms,
    tokenize_fts_text,
)
from mcp_server_rag.hybrid_search import RRF_K, HybridSearchOptions, execute_hybrid_search
from mcp_server_rag.metadata import (
    MySQLMetadataCatalog,
    PostgresMetadataCatalog,
    SQLiteMetadataCatalog,
)


class _DeterministicContrastEmbedding:
    """可控对比嵌入桩，用于模拟稠密向量偏向语义干扰项而弱化精确错误码的场景。"""

    @property
    def dimension(self) -> int:
        return 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if "网络超时通用排障手册" in text:
                # 故意赋予干扰文档极高向量接近度（与查询同向）
                vectors.append([0.99, 0.05, 0.0, 0.0])
            elif "0x80070005" in text:
                vectors.append([0.55, 0.83, 0.0, 0.0])
            elif "ERR_GATEWAY_TIMEOUT_504" in text:
                vectors.append([0.35, 0.93, 0.0, 0.0])
            elif "CVE-2026-0001" in text:
                vectors.append([0.30, 0.95, 0.0, 0.0])
            elif "微服务熔断降级" in text:
                vectors.append([0.65, 0.65, 0.20, 0.0])
            else:
                vectors.append([0.0, 0.0, 0.1, 0.99])
        return vectors

    def embed_query(self, query: str) -> list[float]:
        if " cake " in query or "蛋糕" in query:
            return [0.0, 0.0, 0.1, 0.99]
        return [1.0, 0.0, 0.0, 0.0]


class _SpyCursor:
    """记录关系型 SQL 执行轨迹并在 driver 接缝层返回预置结果的轻量游标。"""

    def __init__(self, state: "_SpyDB") -> None:
        self._state = state
        self._rows: list[Any] = []
        self.rowcount: int = 0
        self.description: list[tuple[str, ...]] = []
        self.closed = False

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        clean_sql = " ".join(sql.strip().split())
        self._state.queries.append((clean_sql, params))
        if self._state.execute_hook is not None:
            result = self._state.execute_hook(clean_sql, params, self)
            if result is not None:
                return
        self._rows = []
        self.description = []
        self.rowcount = 1

    def executemany(self, sql: str, seq_of_params: list[tuple[Any, ...]]) -> None:
        clean_sql = " ".join(sql.strip().split())
        self._state.many_queries.append((clean_sql, list(seq_of_params)))
        self.rowcount = len(seq_of_params)

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Any]:
        return list(self._rows)

    def close(self) -> None:
        self.closed = True


class _SpyConnection:
    def __init__(self, state: "_SpyDB") -> None:
        self._state = state

    def cursor(self) -> _SpyCursor:
        return _SpyCursor(self._state)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


class _SpyDB:
    def __init__(self, execute_hook: Any = None) -> None:
        self.queries: list[tuple[str, tuple[Any, ...]]] = []
        self.many_queries: list[tuple[str, list[tuple[Any, ...]]]] = []
        self.execute_hook = execute_hook

    def connect(self) -> _SpyConnection:
        return _SpyConnection(self)


def _init_hybrid_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: EmbeddingProvider | None = None,
) -> Path:
    """初始化独立 data_dir 环境变量并注入可控嵌入模型测试桩。"""
    data_dir = tmp_path / "rag_data"
    monkeypatch.setenv("MCP_RAG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MCP_RAG_METADATA_STORE", "sqlite")
    monkeypatch.setenv("MCP_RAG_VECTOR_STORE", "chromadb")
    if provider is not None:
        monkeypatch.setattr(server, "get_embedding_provider", lambda _s: provider)
    return data_dir


# ==============================================================================
# 1. FTS 分词与方言查询构建测试
# ==============================================================================


def test_fts_tokenization_and_query_builders() -> None:
    """验证错误码/CVE/驼峰/中文混合分词及 FTS5/PG/MySQL 防注入查询构建。"""
    sample = "系统返回 ERR_GATEWAY_TIMEOUT_504，关联编号 CVE-2026-0001 及 错误码 0x80070005。"
    indexed = tokenize_fts_text(sample)
    tokens = set(indexed.split())

    assert "err_gateway_timeout_504" in tokens
    assert "cve_2026_0001" in tokens
    assert "0x80070005" in tokens
    assert "错误" in tokens
    assert "误码" in tokens

    # 复合标识符在查询提取中应保持完整规范词元，避免 '2026' 泛化污染无关文档
    cve_terms = extract_fts_query_terms("请排查 CVE-2026-0001 原因")
    assert "cve_2026_0001" in cve_terms
    assert "2026" not in cve_terms
    assert "0001" not in cve_terms
    assert "排查" in cve_terms
    assert "原因" in cve_terms

    # 单字非虚词保留、单字纯虚词在其他实词存在时被过滤
    assert extract_fts_query_terms("什么是 锁") == ["什么", "么是", "锁"]
    assert extract_fts_query_terms("的") == ["的"]

    # FTS5 / PG / MySQL 特殊符号防注入构造
    dangerous = 'ERR-504: "timeout" * (A OR B); -- 中文排障'
    fts5_q = build_sqlite_fts5_query(dangerous)
    assert '"err_504"' in fts5_q
    assert '"timeout"' in fts5_q
    assert " OR " in fts5_q
    assert build_sqlite_fts5_query("!!! ???") == ""

    pg_q = build_postgres_tsquery("CVE-2026-0001 网关超时")
    assert "(cve <-> 2026 <-> 0001)" in pg_q
    assert " | " in pg_q

    mysql_q = build_mysql_ft_query("CVE-2026-0001 ERR_GATEWAY_TIMEOUT_504 网关超时")
    assert "cve-2026-0001" in mysql_q
    assert "cve_2026_0001" not in mysql_q
    assert "err_gateway_timeout_504" in mysql_q
    assert "网关超时" in mysql_q


# ==============================================================================
# 2. SQLite FTS5 原生索引、存量迁移与事务级联清理测试
# ==============================================================================


def test_sqlite_fts5_lifecycle_overwrite_and_cleanup(tmp_path: Path) -> None:
    """验证 SQLite chunks_fts 建表、BM25 召回、覆盖防幽灵与文档/库删除级联清理。"""
    catalog = SQLiteMetadataCatalog(data_dir=tmp_path)
    assert catalog.create_kb("ops_kb")
    assert catalog.create_kb("other_kb")

    catalog.register_document_and_chunks(
        kb_name="ops_kb",
        doc_id="doc_err",
        title="Windows 权限故障指南",
        content_hash="h1",
        source_uri="manual://win",
        chunk_ids=["c_win_0", "c_win_1"],
        file_size=128,
        chunk_contents=[
            "当系统挂载共享卷发生 0x80070005 错误时，说明访问令牌权限不足。",
            "后续请检查 ACL 并重启服务，避免出现 ERR_GATEWAY_TIMEOUT_504。",
        ],
        chunk_images=[["/assets/ops_kb/doc_err/win.png"], []],
    )
    catalog.register_document_and_chunks(
        kb_name="other_kb",
        doc_id="doc_other",
        title="隔离库文档",
        content_hash="h2",
        source_uri="manual://other",
        chunk_ids=["c_other_0"],
        file_size=64,
        chunk_contents=["此处包含错误码 0x80070005，但属于隔离库 other_kb。"],
    )

    # 1. 精确错误码召回与 kb_names 作用域隔离
    hits = catalog.search_chunks_fts(["ops_kb"], "错误码 0x80070005", top_k=5)
    assert len(hits) >= 1
    assert hits[0]["chunk_id"] == "c_win_0"
    assert hits[0]["kb_name"] == "ops_kb"
    assert hits[0]["metadata"]["title"] == "Windows 权限故障指南"
    assert hits[0]["images"] == ["/assets/ops_kb/doc_err/win.png"]
    assert all(item["kb_name"] == "ops_kb" for item in hits)

    # 2. 幂等覆盖文档（减少切片并移除旧错误码），旧 FTS 幽灵记录必须彻底清除
    catalog.register_document_and_chunks(
        kb_name="ops_kb",
        doc_id="doc_err",
        title="Windows 权限故障指南",
        content_hash="h1_v2",
        source_uri="manual://win",
        chunk_ids=["c_win_v2_0"],
        file_size=64,
        chunk_contents=["最新版已迁移至 OAuth2 统一鉴权，相关安全漏洞编号 CVE-2026-0001 已修复。"],
    )
    assert catalog.search_chunks_fts(["ops_kb"], "0x80070005", top_k=5) == []
    assert catalog.search_chunks_fts(["ops_kb"], "ERR_GATEWAY_TIMEOUT_504", top_k=5) == []
    v2_hits = catalog.search_chunks_fts(["ops_kb"], "CVE-2026-0001", top_k=5)
    assert len(v2_hits) == 1
    assert v2_hits[0]["chunk_id"] == "c_win_v2_0"

    # 3. delete_document 清理对应 FTS 索引
    catalog.delete_document("ops_kb", "doc_err")
    assert catalog.search_chunks_fts(["ops_kb"], "CVE-2026-0001", top_k=5) == []

    # 4. clear_kb 与 delete_kb 彻底清理所属 KB 的全部 FTS 索引
    assert catalog.search_chunks_fts(["other_kb"], "0x80070005", top_k=5) != []
    catalog.clear_kb("other_kb")
    assert catalog.search_chunks_fts(["other_kb"], "0x80070005", top_k=5) == []

    catalog.register_document_and_chunks(
        kb_name="other_kb",
        doc_id="doc_readded",
        title="重录文档",
        content_hash="h3",
        source_uri="",
        chunk_ids=["c_readded"],
        file_size=32,
        chunk_contents=["重现 ERR_GATEWAY_TIMEOUT_504 告警日志。"],
    )
    assert len(catalog.search_chunks_fts(["other_kb"], "ERR_GATEWAY_TIMEOUT_504", top_k=5)) == 1
    catalog.delete_kb("other_kb")
    assert catalog.search_chunks_fts(["other_kb"], "ERR_GATEWAY_TIMEOUT_504", top_k=5) == []


def test_sqlite_fts5_legacy_migration_backfill(tmp_path: Path) -> None:
    """验证存量无 chunks_fts 表的旧 SQLite 库启动时会自动建表并回填历史切片倒排索引。"""
    db_file = tmp_path / "legacy_metadata.db"
    conn = sqlite3.connect(str(db_file))
    try:
        conn.executescript(
            """
            CREATE TABLE knowledge_bases (
                name TEXT PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                embedding_model TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
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
            );
            CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                kb_name TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL DEFAULT ''
            );
            INSERT INTO knowledge_bases VALUES (
                'legacy_kb', '', '', '2026-01-01', '2026-01-01'
            );
            INSERT INTO documents VALUES (
                'd1', 'legacy_kb', '历史故障单', 'h', '', 1, 50, '2026-01-01', '2026-01-01'
            );
            INSERT INTO chunks VALUES (
                'legacy_c0', 'legacy_kb', 'd1', 0,
                '旧系统核心错误码 ERR_GATEWAY_TIMEOUT_504 排查记录'
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    catalog = SQLiteMetadataCatalog(data_dir=tmp_path, db_path=db_file)
    hits = catalog.search_chunks_fts(["legacy_kb"], "ERR_GATEWAY_TIMEOUT_504", top_k=5)
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "legacy_c0"
    assert hits[0]["metadata"]["title"] == "历史故障单"

    # 再次打开应幂等跳过重复回填
    catalog_again = SQLiteMetadataCatalog(data_dir=tmp_path, db_path=db_file)
    hits_again = catalog_again.search_chunks_fts(["legacy_kb"], "ERR_GATEWAY_TIMEOUT_504", top_k=5)
    assert len(hits_again) == 1


# ==============================================================================
# 3. 端到端 RRF 混合检索 (SQLite FTS5 + ChromaDB)
# ==============================================================================


def test_e2e_hybrid_rrf_surfaces_exact_codes_over_dense_distractor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证当稠密向量被语义干扰项误导时，FTS+RRF 混合检索仍能把精确错误码/CVE 排在首位。"""
    _init_hybrid_env(monkeypatch, tmp_path, provider=_DeterministicContrastEmbedding())

    server.rag_create_kb("kb_ops", description="运维排障库")
    server.rag_create_kb("kb_sec", description="安全通告库")

    # 写入强稠密向量干扰文本，以及各自携带精确标识符的真实目标文档
    server.rag_index_text(
        kb_name="kb_ops",
        title="通用网络指南",
        text="网络超时通用排障手册：检查网关配置、代理与 DNS 缓存。",
    )
    server.rag_index_text(
        kb_name="kb_ops",
        title="系统故障 A",
        text="Windows CIFS 挂载失败触发 0x80070005 拒绝访问错误，请检查凭据。",
    )
    server.rag_index_text(
        kb_name="kb_ops",
        title="网关错误 B",
        text="Upstream 节点异常引起 ERR_GATEWAY_TIMEOUT_504 报错。",
    )
    server.rag_index_text(
        kb_name="kb_sec",
        title="漏洞公告",
        text="紧急修复高危安全漏洞 CVE-2026-0001，升级认证组件至最新版本。",
    )

    # 1. 精确错误码 0x80070005 必须借助 FTS+RRF 逆转稠密干扰并排在 Top-1
    hits_hex = server.rag_search("kb_ops", "如何解决 0x80070005", top_k=3)
    assert len(hits_hex) >= 2
    assert "0x80070005" in hits_hex[0]["text"]
    assert "fts" in hits_hex[0]["metadata"]["retrieval_channels"]
    assert hits_hex[0]["metadata"]["rrf_score"] > hits_hex[1]["metadata"]["rrf_score"]

    # 验证标准 RRF 公式 (k = 60)：双路命中分值严格等于 1/(60+vec_rank) + 1/(60+fts_rank)
    assert hits_hex[0]["metadata"]["retrieval_channels"] == ["vector", "fts"]
    expected_double_rrf = round((1.0 / (RRF_K + 2)) + (1.0 / (RRF_K + 1)), 6)
    assert hits_hex[0]["metadata"]["rrf_score"] == pytest.approx(expected_double_rrf, abs=1e-5)

    # 2. rag_query_with_context 通配符 '*' 跨库检索精确定位 CVE-2026-0001
    context_md = server.rag_query_with_context("*", "CVE-2026-0001 修复建议", top_k=2)
    assert "CVE-2026-0001" in context_md
    assert "漏洞公告" in context_md
    assert "kb_sec" in context_md

    # 3. 文档更新后，旧错误码不再被 FTS 召回，新标识可立即召回
    server.rag_index_text(
        kb_name="kb_ops",
        title="系统故障 A",
        text="Windows CIFS 已切换新错误码 ERR_NEW_ACL_403，旧代码已废弃。",
    )
    stale_hits = server.rag_search("kb_ops", "0x80070005", top_k=3)
    assert all("0x80070005" not in h["text"] for h in stale_hits)
    fresh_hits = server.rag_search("kb_ops", "ERR_NEW_ACL_403", top_k=2)
    assert len(fresh_hits) >= 1
    assert "ERR_NEW_ACL_403" in fresh_hits[0]["text"]

    # 4. 模拟纯 FTS 召回并验证高 RRF 项在后项为 1.0 时不发生分数倒挂
    catalog, vector_store = server._get_services()

    class _EmptyVectorStore:
        def search(self, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    fts_only_hits = execute_hybrid_search(
        catalog=catalog,
        vector_store=_EmptyVectorStore(),  # type: ignore[arg-type]
        target_kbs=["kb_ops"],
        query="ERR_NEW_ACL_403",
        query_vector=[1.0, 0.0, 0.0, 0.0],
        options=HybridSearchOptions(top_k=3),
    )
    assert len(fts_only_hits) == 1
    assert fts_only_hits[0]["metadata"]["retrieval_channels"] == ["fts"]
    assert fts_only_hits[0]["score"] == 0.85

    class _HighScoreVectorStore:
        def search(self, **_kwargs: Any) -> list[dict[str, Any]]:
            return [
                {
                    "chunk_id": "通用网络指南_0",
                    "text": "网络超时通用排障手册：检查网关配置、代理与 DNS 缓存。",
                    "score": 1.0,
                    "metadata": {"doc_id": "通用网络指南"},
                },
                {
                    "chunk_id": "系统故障 A_0",
                    "text": "Windows CIFS 已切换新错误码 ERR_NEW_ACL_403，旧代码已废弃。",
                    "score": 0.60,
                    "metadata": {"doc_id": "系统故障 A"},
                },
            ]

    monotonic_hits = execute_hybrid_search(
        catalog=catalog,
        vector_store=_HighScoreVectorStore(),  # type: ignore[arg-type]
        target_kbs=["kb_ops"],
        query="ERR_NEW_ACL_403",
        query_vector=[1.0, 0.0, 0.0, 0.0],
        options=HybridSearchOptions(top_k=2),
    )
    assert monotonic_hits[0]["chunk_id"] == "系统故障 A_0"
    assert monotonic_hits[0]["score"] >= monotonic_hits[1]["score"] == 1.0
    del vector_store


def test_e2e_hybrid_search_respects_filters_deletion_and_context_expansion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证混合检索严格遵守 metadata_filter / keyword / 删除联动与上下文窗口扩展。"""
    _init_hybrid_env(monkeypatch, tmp_path, provider=_DeterministicContrastEmbedding())
    monkeypatch.setenv("MCP_RAG_CHUNK_SIZE", "32")
    monkeypatch.setenv("MCP_RAG_CHUNK_OVERLAP", "0")

    server.rag_create_kb("kb_filter", description="过滤测试库")
    server.rag_index_text(
        kb_name="kb_filter",
        title="生产配置异常",
        text=(
            "第一段背景说明网关架构与负载均衡流程。\n\n"
            "第二段记录 ERR_GATEWAY_TIMEOUT_504 发生在 prod 环境。"
        ),
        metadata={"env": "prod"},
    )
    server.rag_index_text(
        kb_name="kb_filter",
        title="测试环境异常",
        text="dev 环境同样出现 ERR_GATEWAY_TIMEOUT_504 报错信息。",
        metadata={"env": "dev"},
    )

    # 1. metadata_filter 必须严格隔离未匹配切片，绝不允许 FTS 越权泄漏 dev 环境文档
    prod_hits = server.rag_search(
        kb_name="kb_filter",
        query="ERR_GATEWAY_TIMEOUT_504",
        top_k=5,
        metadata_filter={"env": "prod"},
        expand_context=1,
    )
    assert len(prod_hits) == 1
    assert prod_hits[0]["metadata"]["env"] == "prod"
    assert "第一段背景说明" in prod_hits[0]["text"]
    assert "ERR_GATEWAY_TIMEOUT_504" in prod_hits[0]["text"]
    assert "dev 环境" not in prod_hits[0]["text"]

    # 2. keyword 精准过滤同样作用于 FTS+RRF 召回通道
    kw_hits = server.rag_search(
        kb_name="kb_filter",
        query="ERR_GATEWAY_TIMEOUT_504",
        top_k=5,
        keyword="dev",
    )
    assert len(kw_hits) == 1
    assert kw_hits[0]["metadata"]["env"] == "dev"

    # 3. rag_delete_document 后，FTS 不应残留幽灵切片
    catalog, _ = server._get_services()
    dev_doc = catalog.get_document_by_title("kb_filter", "测试环境异常")
    assert dev_doc is not None
    server.rag_delete_document("kb_filter", dev_doc["id"])
    assert server.rag_search("kb_filter", "ERR_GATEWAY_TIMEOUT_504", top_k=5, keyword="dev") == []


# ==============================================================================
# 4. PostgreSQL / MySQL 原生 FTS 方言契约测试（驱动接缝层验证）
# ==============================================================================


def test_postgres_native_fts_ddl_upsert_and_query_contract() -> None:
    """验证 PostgreSQL 后端自动建立 tsvector/GIN 索引、同步写入及 ts_rank_cd 检索 SQL。"""

    def pg_hook(clean_sql: str, _params: tuple[Any, ...], cur: _SpyCursor) -> bool | None:
        if "SELECT id, content FROM chunks WHERE content_tsv IS NULL" in clean_sql:
            cur._rows = []
            cur.description = [("id",), ("content",)]
            return True
        if "FROM chunks c" in clean_sql and "ts_rank_cd" in clean_sql:
            cur.description = [
                ("id",),
                ("kb_name",),
                ("doc_id",),
                ("chunk_index",),
                ("content",),
                ("prev_chunk_id",),
                ("next_chunk_id",),
                ("images_json",),
                ("created_at",),
                ("doc_title",),
                ("doc_source_uri",),
                ("fts_score",),
            ]
            cur._rows = [
                (
                    "pg_c0",
                    "kb_pg",
                    "pg_d1",
                    0,
                    "错误码 0x80070005 故障记录",
                    None,
                    None,
                    "[]",
                    "2026-11-11T00:00:00Z",
                    "PG 故障文档",
                    "doc://pg",
                    0.75,
                )
            ]
            return True
        return None

    spy = _SpyDB(execute_hook=pg_hook)
    catalog = PostgresMetadataCatalog(
        dsn="postgresql://rag:pass@pg.internal:5432/rag_db",
        connection_factory=spy.connect,
    )

    all_ddl = "\n".join(sql for sql, _ in spy.queries)
    assert "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR" in all_ddl
    gin_ddl = "CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING GIN (content_tsv)"
    assert gin_ddl in all_ddl

    catalog.register_document_and_chunks(
        kb_name="kb_pg",
        doc_id="pg_d1",
        title="PG 故障文档",
        content_hash="h_pg",
        source_uri="doc://pg",
        chunk_ids=["pg_c0"],
        file_size=40,
        chunk_contents=["错误码 0x80070005 故障记录"],
    )
    insert_calls = [(sql, p) for sql, p in spy.queries if sql.startswith("INSERT INTO chunks ")]
    assert len(insert_calls) == 1
    assert "to_tsvector('simple', %s)" in insert_calls[0][0]
    assert "0x80070005" in str(insert_calls[0][1][-1])

    fts_hits = catalog.search_chunks_fts(["kb_pg"], "错误码 0x80070005", top_k=5)
    assert len(fts_hits) == 1
    assert fts_hits[0]["chunk_id"] == "pg_c0"
    assert fts_hits[0]["fts_score"] == 0.75
    assert fts_hits[0]["metadata"]["title"] == "PG 故障文档"

    fts_selects = [sql for sql, _ in spy.queries if "ts_rank_cd(c.content_tsv" in sql]
    assert len(fts_selects) == 1
    assert "c.content_tsv @@ to_tsquery('simple', %s)" in fts_selects[0]

    assert catalog.search_chunks_fts([], "0x80070005", top_k=5) == []
    assert catalog.search_chunks_fts(["kb_pg"], "!!!", top_k=5) == []

    # 校验存量空 content_tsv 记录通过 executemany 批量回填，拒绝 N+1 逐条 UPDATE
    backfill_round = 0

    def pg_backfill_hook(clean_sql: str, _params: tuple[Any, ...], cur: _SpyCursor) -> bool | None:
        nonlocal backfill_round
        if "SELECT id, content FROM chunks WHERE content_tsv IS NULL" in clean_sql:
            cur.description = [("id",), ("content",)]
            if backfill_round == 0:
                backfill_round += 1
                cur._rows = [("legacy_1", "报错 0x80070005"), ("legacy_2", "漏洞 CVE-2026-0001")]
            else:
                cur._rows = []
            return True
        return None

    backfill_spy = _SpyDB(execute_hook=pg_backfill_hook)
    PostgresMetadataCatalog(
        dsn="postgresql://rag:pass@pg.internal:5432/rag_db",
        connection_factory=backfill_spy.connect,
    )
    assert len(backfill_spy.many_queries) == 1
    assert (
        "UPDATE chunks SET content_tsv = to_tsvector('simple', %s) WHERE id = %s"
        in (backfill_spy.many_queries[0][0])
    )
    assert len(backfill_spy.many_queries[0][1]) == 2
    assert not any(sql.startswith("UPDATE chunks ") for sql, _ in backfill_spy.queries)


def test_mysql_native_fts_ngram_fallback_and_query_contract() -> None:
    """验证 MySQL FULLTEXT ngram 建表、插件缺失降级重连建索引与 MATCH AGAINST 检索 SQL。"""

    def mysql_hook_fallback(
        clean_sql: str,
        _params: tuple[Any, ...],
        cur: _SpyCursor,
    ) -> bool | None:
        if clean_sql.startswith("SHOW INDEX FROM chunks"):
            cur._rows = []
            return True
        if "WITH PARSER ngram" in clean_sql:
            raise RuntimeError(1128, "Function 'ngram' is not defined")
        if "MATCH(c.content) AGAINST" in clean_sql:
            cur.description = [
                ("id",),
                ("kb_name",),
                ("doc_id",),
                ("chunk_index",),
                ("content",),
                ("prev_chunk_id",),
                ("next_chunk_id",),
                ("images_json",),
                ("created_at",),
                ("doc_title",),
                ("doc_source_uri",),
                ("fts_score",),
            ]
            cur._rows = [
                (
                    "my_c0",
                    "kb_my",
                    "my_d1",
                    0,
                    "Nginx 报错 ERR_GATEWAY_TIMEOUT_504",
                    None,
                    None,
                    "[]",
                    "2026-11-11T00:00:00Z",
                    "MySQL 故障文档",
                    "doc://mysql",
                    1.25,
                )
            ]
            return True
        return None

    spy = _SpyDB(execute_hook=mysql_hook_fallback)
    catalog = MySQLMetadataCatalog(
        dsn="mysql://rag:pass@mysql.internal:3306/rag_db",
        connection_factory=spy.connect,
    )

    sql_history = [sql for sql, _ in spy.queries]
    ngram_stmt = "ADD FULLTEXT INDEX idx_chunks_content_ft (content) WITH PARSER ngram"
    std_stmt = "ADD FULLTEXT INDEX idx_chunks_content_ft (content)"
    assert any(ngram_stmt in s for s in sql_history)
    assert any(s.endswith(std_stmt) for s in sql_history)

    hits = catalog.search_chunks_fts(["kb_my"], "ERR_GATEWAY_TIMEOUT_504", top_k=5)
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "my_c0"
    assert hits[0]["metadata"]["title"] == "MySQL 故障文档"
    assert any("IN NATURAL LANGUAGE MODE" in s for s, _ in spy.queries)
