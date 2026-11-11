"""
向量存储适配器工厂。

根据运行时配置动态初始化并返回 ChromaDB 或 Milvus 向量存储适配器。

@author Ateng
@since 2026-11-11
"""

from mcp_server_rag.config import Settings
from mcp_server_rag.vector_store.base import VectorStoreAdapter
from mcp_server_rag.vector_store.chroma import ChromaVectorStoreAdapter


def get_vector_store(settings: Settings) -> VectorStoreAdapter:
    """根据运行时配置获取向量存储适配器实例。

    支持通过 MCP_RAG_VECTOR_STORE=chromadb|milvus 动态切换底层向量引擎。

    Args:
        settings: 全局运行时配置实例

    Returns:
        VectorStoreAdapter: 符合适配器协议的向量存储驱动实例
    """
    store_type = settings.vector_store.lower().strip()
    if store_type == "milvus":
        from mcp_server_rag.vector_store.milvus import MilvusVectorStoreAdapter

        return MilvusVectorStoreAdapter(
            uri=settings.milvus_uri,
            token=settings.milvus_token,
        )
    return ChromaVectorStoreAdapter(data_dir=settings.data_dir)
