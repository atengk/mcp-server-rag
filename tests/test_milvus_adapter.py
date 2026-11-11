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
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_milvus_missing_dependency_informative_error() -> None:
    """验证未安装 pymilvus 时提供清晰的语义化报错并引导安装可选扩展。"""
    with patch.dict(sys.modules, {"pymilvus": None}):
        with pytest.raises(ImportError) as exc_info:
            adapter = MilvusVectorStoreAdapter(uri="http://localhost:19530")
            adapter._get_client()
        error_msg = str(exc_info.value)
        assert "atengk-mcp-server-rag[milvus]" in error_msg
        assert "pymilvus" in error_msg


def test_milvus_adapter_lifecycle_and_contract() -> None:
    """验证 MilvusVectorStoreAdapter 遵循 VectorStoreAdapter 协议，数据格式与 ChromaDB 对齐。"""
    mock_client = MagicMock()
    mock_client.has_collection.return_value = False
    mock_client.query.return_value = [{"count(*)": 2}]
    mock_client.list_collections.return_value = ["mcp_rag_chunks"]

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

    # 2. 知识库创建（统一集合单表，延迟初始化或检查）
    adapter.create_kb("kb_milvus_test")
    mock_client.has_collection.assert_called_with(collection_name="mcp_rag_chunks")
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
    assert insert_args["collection_name"] == "mcp_rag_chunks"
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
    mock_client.delete.assert_called_with(collection_name="mcp_rag_chunks", ids=["doc1_0"])

    adapter.clear_kb("kb_milvus_test")
    mock_client.delete.assert_called_with(
        collection_name="mcp_rag_chunks",
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
