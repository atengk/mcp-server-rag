"""
端到端与接缝测试：可插拔企业级 Milvus 向量存储适配器 (Issue #10)。

在单一最高接缝（FastMCP Tool 协议执行层边界与 VectorStoreAdapter 抽象契约）测试：
1. 未安装 pymilvus 时的友好异常提示与安装引导契约；
2. 统一集合 mcp_rag_chunks 与标量 kb_name 分区过滤；
3. Milvus 适配器向量增删查生命周期契约与 ChromaDB 返回格式 100% 对齐；
4. MCP 服务端在 MCP_RAG_VECTOR_STORE=milvus 模式下的端到端驱动检索。

@author Ateng
@since 2026-11-11
"""

import os
import shutil
import sys
import tempfile
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mcp_server_rag.config import Settings
from mcp_server_rag.server import (
    rag_create_kb,
    rag_doctor,
    rag_index_text,
    rag_search,
)
from mcp_server_rag.vector_store import (
    MilvusVectorStoreAdapter,
    VectorStoreAdapter,
    get_vector_store,
)


@pytest.fixture(autouse=True)
def test_data_dir() -> Generator[str, None, None]:
    """为每个测试用例隔离独立的临时数据目录，并在测试后清理。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_milvus_")
    old_data_dir = os.environ.get("MCP_RAG_DATA_DIR")
    old_store = os.environ.get("MCP_RAG_VECTOR_STORE")
    old_milvus_uri = os.environ.get("MCP_RAG_MILVUS_URI")
    old_milvus_token = os.environ.get("MCP_RAG_MILVUS_TOKEN")
    old_milvus_coll = os.environ.get("MCP_RAG_MILVUS_COLLECTION")
    old_milvus_prefix = os.environ.get("MCP_RAG_MILVUS_COLLECTION_PREFIX")
    old_emb_dim = os.environ.get("MCP_RAG_EMBEDDING_DIMENSION")
    os.environ["MCP_RAG_DATA_DIR"] = temp_dir
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    yield temp_dir
    if old_data_dir is not None:
        os.environ["MCP_RAG_DATA_DIR"] = old_data_dir
    else:
        os.environ.pop("MCP_RAG_DATA_DIR", None)

    if old_store is not None:
        os.environ["MCP_RAG_VECTOR_STORE"] = old_store
    else:
        os.environ.pop("MCP_RAG_VECTOR_STORE", None)

    if old_milvus_uri is not None:
        os.environ["MCP_RAG_MILVUS_URI"] = old_milvus_uri
    else:
        os.environ.pop("MCP_RAG_MILVUS_URI", None)

    if old_milvus_token is not None:
        os.environ["MCP_RAG_MILVUS_TOKEN"] = old_milvus_token
    else:
        os.environ.pop("MCP_RAG_MILVUS_TOKEN", None)

    if old_milvus_coll is not None:
        os.environ["MCP_RAG_MILVUS_COLLECTION"] = old_milvus_coll
    else:
        os.environ.pop("MCP_RAG_MILVUS_COLLECTION", None)

    if old_milvus_prefix is not None:
        os.environ["MCP_RAG_MILVUS_COLLECTION_PREFIX"] = old_milvus_prefix
    else:
        os.environ.pop("MCP_RAG_MILVUS_COLLECTION_PREFIX", None)

    if old_emb_dim is not None:
        os.environ["MCP_RAG_EMBEDDING_DIMENSION"] = old_emb_dim
    else:
        os.environ.pop("MCP_RAG_EMBEDDING_DIMENSION", None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_milvus_missing_dependency_informative_error() -> None:
    """验证未安装 pymilvus 时提供清晰的语义化报错并引导安装可选扩展。"""
    with patch.dict(sys.modules, {"pymilvus": None}):
        with pytest.raises(ImportError) as exc_info:
            adapter = MilvusVectorStoreAdapter(uri="http://localhost:19530")
            adapter._get_client()
        error_msg = str(exc_info.value)
        assert "ateng-mcp-server-rag[milvus]" in error_msg
        assert "pymilvus" in error_msg


def test_milvus_adapter_lifecycle_and_contract() -> None:
    """验证 MilvusVectorStoreAdapter 遵循 VectorStoreAdapter 协议，数据格式与 ChromaDB 对齐。"""
    mock_client = MagicMock()
    mock_client.has_collection.return_value = False
    mock_client.query.return_value = [{"count(*)": 2}]
    mock_client.list_collections.return_value = ["mcp_rag_chunks_d512"]

    # 模拟 search 检索返回
    mock_client.search.return_value = [
        [
            {
                "id": "doc1_0",
                "distance": 0.8876,
                "entity": {
                    "text": "Milvus 分布式向量检索架构",
                    "metadata": {"category": "database"},
                    "kb_name": "kb_milvus_test",
                },
            }
        ]
    ]

    adapter = MilvusVectorStoreAdapter(
        uri="http://mock-milvus:19530",
        token="test-token",
        client=mock_client,
    )

    # 1. 验证 Protocol 兼容性
    assert isinstance(adapter, VectorStoreAdapter)

    # 2. 知识库创建（按默认 512 维度路由至 mcp_rag_chunks_d512）
    adapter.create_kb("kb_milvus_test")
    mock_client.has_collection.assert_called_with(collection_name="mcp_rag_chunks_d512")
    mock_client.create_collection.assert_called_once()

    # 3. 写入切片
    mock_client.has_collection.return_value = True
    adapter.add_chunks(
        kb_name="kb_milvus_test",
        chunk_ids=["doc1_0"],
        texts=["Milvus 分布式向量检索架构"],
        embeddings=[[0.1] * 512],
        metadatas=[{"category": "database"}],
    )
    assert mock_client.insert.called
    insert_args = mock_client.insert.call_args[1]
    assert insert_args["collection_name"] == "mcp_rag_chunks_d512"
    assert insert_args["data"][0]["kb_name"] == "kb_milvus_test"
    assert insert_args["data"][0]["id"] == "doc1_0"

    # 4. 执行检索并验证契约字段
    results = adapter.search(
        kb_name="kb_milvus_test",
        query_embedding=[0.1] * 512,
        top_k=3,
    )
    assert len(results) == 1
    hit = results[0]
    assert hit["chunk_id"] == "doc1_0"
    assert hit["text"] == "Milvus 分布式向量检索架构"
    assert hit["score"] == 0.8876
    assert hit["metadata"]["category"] == "database"
    assert hit["kb_name"] == "kb_milvus_test"

    # 5. 统计与诊断状态
    assert adapter.count("kb_milvus_test") == 2
    assert adapter.ping() is True
    info = adapter.get_info()
    assert info["type"] == "milvus"
    assert info["status"] == "ok"

    # 6. 删除切片与清空知识库
    adapter.delete_chunks("kb_milvus_test", ["doc1_0"])
    mock_client.delete.assert_called_with(collection_name="mcp_rag_chunks_d512", ids=["doc1_0"])

    adapter.clear_kb("kb_milvus_test")
    mock_client.delete.assert_called_with(
        collection_name="mcp_rag_chunks_d512",
        filter='kb_name == "kb_milvus_test"',
    )


def test_mcp_server_end_to_end_with_milvus_store() -> None:
    """验证在最高接缝层配置 MCP_RAG_VECTOR_STORE=milvus 时，MCP 工具链路正常协同。"""
    os.environ["MCP_RAG_VECTOR_STORE"] = "milvus"
    os.environ["MCP_RAG_MILVUS_URI"] = "http://mock-milvus:19530"

    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    mock_client.list_collections.return_value = ["mcp_rag_chunks"]
    mock_client.query.return_value = [{"count(*)": 1}]
    mock_client.search.return_value = [
        [
            {
                "id": "milvus_guide_0",
                "distance": 0.9521,
                "entity": {
                    "text": "Milvus 是为海量向量相似度搜索打造的企业级开源向量数据库。",
                    "metadata": {"doc_id": "milvus_guide", "title": "milvus_guide"},
                    "kb_name": "kb_enterprise",
                },
            }
        ]
    ]

    mock_adapter = MilvusVectorStoreAdapter(
        uri="http://mock-milvus:19530",
        client=mock_client,
    )

    with patch("mcp_server_rag.server.get_vector_store", return_value=mock_adapter):
        # 1. 诊断探针验证
        doctor_res = rag_doctor()
        assert doctor_res["vector_store"] == "milvus"
        assert doctor_res["vector_store_status"] == "ok"

        # 2. 建库与文本索引
        res_create = rag_create_kb("kb_enterprise", "企业级知识库")
        assert res_create["success"] is True

        res_index = rag_index_text(
            kb_name="kb_enterprise",
            title="milvus_guide",
            text="Milvus 是为海量向量相似度搜索打造的企业级开源向量数据库。",
        )
        assert res_index["success"] is True

        # 3. 语义搜索端到端返回
        search_res = rag_search(
            kb_name="kb_enterprise",
            query="企业级向量数据库",
            top_k=1,
        )
        assert len(search_res) == 1
        assert search_res[0]["chunk_id"] == "milvus_guide_0"
        assert search_res[0]["kb_name"] == "kb_enterprise"
        assert search_res[0]["score"] == 0.9521


def test_factory_vector_store_switching() -> None:
    """验证工厂方法依据配置项平滑切换 ChromaDB 与 Milvus 适配器。"""
    # 默认返回 ChromaDB 适配器
    chroma_store = get_vector_store(Settings(vector_store="chromadb"))
    assert chroma_store.__class__.__name__ == "ChromaVectorStoreAdapter"

    # 指定 milvus 时返回 Milvus 适配器
    milvus_store = get_vector_store(
        Settings(
            vector_store="milvus",
            milvus_uri="http://custom-milvus:19530",
            milvus_token="secret",
        )
    )
    assert isinstance(milvus_store, MilvusVectorStoreAdapter)
    assert milvus_store.uri == "http://custom-milvus:19530"
    assert milvus_store.token == "secret"


def test_milvus_search_with_where_filter_and_empty() -> None:
    """验证 Milvus 搜索标量过滤与异常空安全。"""
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    mock_client.search.return_value = []

    adapter = MilvusVectorStoreAdapter(client=mock_client)

    # 1. 过滤条件拼接验证
    adapter.search(
        kb_name="kb_filter",
        query_embedding=[0.0] * 512,
        where={"category": "ai", "level": {"$eq": "expert"}},
    )
    assert mock_client.search.called
    filter_expr = mock_client.search.call_args[1]["filter"]
    assert 'kb_name == "kb_filter"' in filter_expr
    assert "category == 'ai'" in filter_expr
    assert "level == 'expert'" in filter_expr

    # 2. 集合不存在时安全返回空列表
    mock_client.has_collection.return_value = False
    empty_res = adapter.search(
        kb_name="kb_filter",
        query_embedding=[0.0] * 512,
    )
    assert empty_res == []

    # 3. 统计在集合不存在时安全返回 0
    assert adapter.count("kb_filter") == 0


def test_milvus_collection_settings_and_factory_wiring() -> None:
    """验证 MCP_RAG_MILVUS_COLLECTION 与 MCP_RAG_MILVUS_COLLECTION_PREFIX 解析及工厂接线。"""
    default_settings = Settings(vector_store="milvus")
    assert default_settings.milvus_collection == ""
    assert default_settings.milvus_collection_prefix == "mcp_rag_chunks"

    os.environ["MCP_RAG_MILVUS_COLLECTION_PREFIX"] = "team_rag"
    os.environ["MCP_RAG_EMBEDDING_DIMENSION"] = "1024"
    prefixed_settings = Settings(vector_store="milvus")
    assert prefixed_settings.milvus_collection_prefix == "team_rag"
    prefixed_store = get_vector_store(prefixed_settings)
    assert isinstance(prefixed_store, MilvusVectorStoreAdapter)
    assert prefixed_store.collection_prefix == "team_rag"
    assert prefixed_store.collection_name == "team_rag_d1024"

    os.environ["MCP_RAG_MILVUS_COLLECTION"] = "legacy_single_coll"
    fixed_settings = Settings(vector_store="milvus")
    assert fixed_settings.milvus_collection == "legacy_single_coll"
    fixed_store = get_vector_store(fixed_settings)
    assert isinstance(fixed_store, MilvusVectorStoreAdapter)
    assert fixed_store.collection_name == "legacy_single_coll"


class _FakeMilvusClient:
    """模拟真实 pymilvus.MilvusClient 集合维度约束与读写行为的测试桩。"""

    def __init__(self) -> None:
        self.collections: dict[str, int] = {}
        self.rows: dict[str, list[dict[str, Any]]] = {}

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def list_collections(self) -> list[str]:
        return list(self.collections.keys())

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        dim = self.collections[collection_name]
        return {
            "collection_name": collection_name,
            "fields": [
                {"name": "id", "type": "VARCHAR"},
                {"name": "vector", "type": "FLOAT_VECTOR", "params": {"dim": dim}},
            ],
        }

    def create_collection(self, collection_name: str, dimension: int, **_: Any) -> None:
        if collection_name in self.collections:
            raise RuntimeError(f"Collection {collection_name} already exists")
        self.collections[collection_name] = int(dimension)
        self.rows[collection_name] = []

    def insert(self, collection_name: str, data: list[dict[str, Any]]) -> None:
        expected_dim = self.collections[collection_name]
        for row in data:
            if len(row["vector"]) != expected_dim:
                raise ValueError(
                    f"Vector dimension mismatch for {collection_name}: "
                    f"expected {expected_dim}, got {len(row['vector'])}"
                )
        self.rows[collection_name].extend(data)

    def search(
        self,
        collection_name: str,
        data: list[list[float]],
        filter: str = "",
        limit: int = 5,
        **_: Any,
    ) -> list[list[dict[str, Any]]]:
        expected_dim = self.collections[collection_name]
        query_vec = data[0]
        if len(query_vec) != expected_dim:
            raise ValueError(
                f"Query dimension mismatch for {collection_name}: "
                f"expected {expected_dim}, got {len(query_vec)}"
            )
        target_kb = ""
        if 'kb_name == "' in filter:
            target_kb = filter.split('kb_name == "', 1)[1].split('"', 1)[0]
        matched = [
            {
                "id": row["id"],
                "distance": 0.91,
                "entity": {
                    "id": row["id"],
                    "text": row["text"],
                    "metadata": row["metadata"],
                    "kb_name": row["kb_name"],
                },
            }
            for row in self.rows[collection_name]
            if not target_kb or row["kb_name"] == target_kb
        ]
        return [matched[:limit]]

    def query(
        self,
        collection_name: str,
        filter: str = "",
        **_: Any,
    ) -> list[dict[str, Any]]:
        target_kb = ""
        if 'kb_name == "' in filter:
            target_kb = filter.split('kb_name == "', 1)[1].split('"', 1)[0]
        cnt = sum(
            1
            for row in self.rows.get(collection_name, [])
            if not target_kb or row["kb_name"] == target_kb
        )
        return [{"count(*)": cnt}]

    def delete(
        self,
        collection_name: str,
        ids: list[str] | None = None,
        filter: str = "",
    ) -> None:
        current = self.rows.get(collection_name, [])
        if ids is not None:
            id_set = set(ids)
            self.rows[collection_name] = [r for r in current if r["id"] not in id_set]
        elif 'kb_name == "' in filter:
            target_kb = filter.split('kb_name == "', 1)[1].split('"', 1)[0]
            self.rows[collection_name] = [r for r in current if r["kb_name"] != target_kb]


def test_milvus_dimension_isolated_collections_multi_model() -> None:
    """验证同一 Milvus 实例下不同维度模型自动按维度隔离集合，互不冲突。"""
    fake_client = _FakeMilvusClient()
    adapter = MilvusVectorStoreAdapter(client=fake_client)

    # 1. 创建并写入 1024 维知识库
    adapter.create_kb("kb_qwen_1024", dimension=1024)
    adapter.add_chunks(
        kb_name="kb_qwen_1024",
        chunk_ids=["q1"],
        texts=["1024维向量内容"],
        embeddings=[[0.1] * 1024],
        metadatas=[{"source": "qwen"}],
    )

    # 2. 同一实例切换创建并写入 512 维知识库（不冲突）
    adapter.create_kb("kb_fastembed_512", dimension=512)
    adapter.add_chunks(
        kb_name="kb_fastembed_512",
        chunk_ids=["f1"],
        texts=["512维向量内容"],
        embeddings=[[0.2] * 512],
        metadatas=[{"source": "fastembed"}],
    )

    assert sorted(fake_client.list_collections()) == [
        "mcp_rag_chunks_d1024",
        "mcp_rag_chunks_d512",
    ]

    # 3. 各维度独立检索、统计与跨维度清理
    hits_1024 = adapter.search("kb_qwen_1024", query_embedding=[0.1] * 1024, top_k=3)
    hits_512 = adapter.search("kb_fastembed_512", query_embedding=[0.2] * 512, top_k=3)
    assert len(hits_1024) == 1 and hits_1024[0]["chunk_id"] == "q1"
    assert len(hits_512) == 1 and hits_512[0]["chunk_id"] == "f1"

    assert adapter.count("kb_qwen_1024") == 1
    assert adapter.count("kb_fastembed_512") == 1

    # 即使 adapter 默认维度是 512，删除/清空 1024 维知识库也应准确命中 mcp_rag_chunks_d1024
    adapter.delete_chunks("kb_qwen_1024", ["q1"])
    assert adapter.count("kb_qwen_1024") == 0

    adapter.clear_kb("kb_fastembed_512")
    assert adapter.count("kb_fastembed_512") == 0


def test_milvus_legacy_single_collection_backward_compatibility() -> None:
    """验证历史单集合 mcp_rag_chunks 平滑兼容与显式固定集合约束。"""
    # 场景 A：历史单集合 mcp_rag_chunks (1024 维) 已存在时，1024 维自动复用，512 维自动分表隔离
    fake_client = _FakeMilvusClient()
    fake_client.create_collection("mcp_rag_chunks", dimension=1024)
    fake_client.insert(
        "mcp_rag_chunks",
        [
            {
                "id": "legacy_0",
                "vector": [0.5] * 1024,
                "kb_name": "kb_legacy",
                "text": "历史 1024 维切片",
                "metadata": {},
            }
        ],
    )

    adapter = MilvusVectorStoreAdapter(client=fake_client)
    assert adapter.resolve_collection_name(1024) == "mcp_rag_chunks"
    assert adapter.resolve_collection_name(512) == "mcp_rag_chunks_d512"

    # 1024 维可检索历史单集合数据，也可继续写入同一集合
    legacy_hits = adapter.search("kb_legacy", query_embedding=[0.5] * 1024, top_k=2)
    assert len(legacy_hits) == 1 and legacy_hits[0]["chunk_id"] == "legacy_0"

    # 512 维新库不会与 1024 维历史单集合冲突
    adapter.create_kb("kb_new_512", dimension=512)
    adapter.add_chunks(
        kb_name="kb_new_512",
        chunk_ids=["n1"],
        texts=["新建 512 维切片"],
        embeddings=[[0.2] * 512],
        metadatas=[{}],
    )
    assert "mcp_rag_chunks_d512" in fake_client.list_collections()
    assert adapter.count("kb_new_512") == 1
    assert adapter.count("kb_legacy") == 1

    # 场景 B：显式固定集合名时，始终路由至固定集合并在维度冲突时报错
    fixed_adapter = MilvusVectorStoreAdapter(
        collection_name="mcp_rag_chunks",
        client=fake_client,
    )
    assert fixed_adapter.resolve_collection_name(512) == "mcp_rag_chunks"
    with pytest.raises(ValueError, match="1024"):
        fixed_adapter.create_kb("kb_bad_dim", dimension=512)


def test_rag_create_kb_and_doctor_milvus_dimension_consistency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 rag_create_kb 与 rag_doctor 校验目标维度与 Milvus 集合维度一致性。"""
    from types import ModuleType

    from mcp_server_rag.server import rag_list_kbs

    fake_client = _FakeMilvusClient()
    fake_client.create_collection("legacy_fixed", dimension=1024)

    fake_pymilvus = ModuleType("pymilvus")
    fake_pymilvus.MilvusClient = MagicMock(return_value=fake_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)

    os.environ["MCP_RAG_VECTOR_STORE"] = "milvus"
    os.environ["MCP_RAG_MILVUS_COLLECTION"] = "legacy_fixed"

    # 1. 固定集合为 1024 维而当前模型为 512 维：
    #    rag_doctor 降级告警，rag_create_kb 拦截且不残留元数据
    report_mismatch = rag_doctor()
    assert report_mismatch["status"] == "degraded"
    assert report_mismatch["vector_store_status"] == "error"
    assert report_mismatch["milvus_collection"] == "legacy_fixed"
    assert report_mismatch["milvus_collection_strategy"] == "fixed"
    assert report_mismatch["milvus_collection_dimension"] == 1024
    assert report_mismatch["milvus_dimension_status"] == "mismatch"
    assert "1024" in report_mismatch["vector_store_error"]

    create_bad = rag_create_kb("kb_conflict", "应被维度门禁拦截")
    assert create_bad["success"] is False
    assert "1024" in create_bad["message"]
    assert all(kb["name"] != "kb_conflict" for kb in rag_list_kbs())

    # 2. 取消固定集合，启用按维度隔离模式：rag_doctor 与 rag_create_kb 正常通过
    os.environ.pop("MCP_RAG_MILVUS_COLLECTION", None)
    report_uninit = rag_doctor()
    assert report_uninit["status"] == "healthy"
    assert report_uninit["vector_store_status"] == "ok"
    assert report_uninit["milvus_collection"] == "mcp_rag_chunks_d512"
    assert report_uninit["milvus_collection_strategy"] == "by_dimension"
    assert report_uninit["milvus_dimension_status"] == "uninitialized"

    create_ok = rag_create_kb("kb_512_ok", "按维度隔离创建成功")
    assert create_ok["success"] is True

    report_ok = rag_doctor()
    assert report_ok["status"] == "healthy"
    assert report_ok["milvus_collection_dimension"] == 512
    assert report_ok["milvus_dimension_status"] == "ok"


def test_mcp_server_multi_dimension_milvus_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证通过真实 get_vector_store 接线在同实例多模型切换下建库、索引、跨库检索与删库。"""
    from types import ModuleType

    from mcp_server_rag.server import rag_delete_kb

    fake_client = _FakeMilvusClient()
    fake_pymilvus = ModuleType("pymilvus")
    fake_pymilvus.MilvusClient = MagicMock(return_value=fake_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)

    class _StubEmbeddingProvider:
        def __init__(self, dim: int) -> None:
            self._dim = dim

        @property
        def dimension(self) -> int:
            return self._dim

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[0.25] * self._dim for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            _ = text
            return [0.25] * self._dim

    current_dim = {"val": 1024}
    monkeypatch.setattr(
        "mcp_server_rag.server.get_embedding_provider",
        lambda _settings: _StubEmbeddingProvider(current_dim["val"]),
    )

    os.environ["MCP_RAG_VECTOR_STORE"] = "milvus"
    os.environ.pop("MCP_RAG_MILVUS_COLLECTION", None)

    # 1. 1024 维模型建库与索引
    assert rag_create_kb("kb_1024", "1024维库")["success"] is True
    assert rag_index_text("kb_1024", "Milvus 1024 维检索", title="doc_1024")["success"] is True
    assert "mcp_rag_chunks_d1024" in fake_client.list_collections()

    # 2. 切换为 512 维模型建库与索引（无冲突）
    current_dim["val"] = 512
    assert rag_create_kb("kb_512", "512维库")["success"] is True
    assert rag_index_text("kb_512", "Milvus 512 维检索", title="doc_512")["success"] is True
    assert "mcp_rag_chunks_d512" in fake_client.list_collections()

    # 3. 512 维模型跨全库检索不报错且能命中目标库与 FTS 结果
    hits = rag_search(kb_name="*", query="Milvus", top_k=5)
    assert any(h["kb_name"] == "kb_512" for h in hits)

    # 4. 在 512 维运行时下删除 1024 维知识库，应清理 mcp_rag_chunks_d1024 中的数据
    assert len(fake_client.rows["mcp_rag_chunks_d1024"]) == 1
    assert rag_delete_kb("kb_1024")["success"] is True
    assert len(fake_client.rows["mcp_rag_chunks_d1024"]) == 0
