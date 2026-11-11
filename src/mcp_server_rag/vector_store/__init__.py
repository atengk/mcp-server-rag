"""
向量存储适配器模块。

提供分层可插拔向量存储抽象与 ChromaDB、Milvus 驱动实现及工厂。

@author Ateng
@since 2026-11-11
"""

from .base import VectorStoreAdapter
from .chroma import ChromaVectorStoreAdapter
from .factory import get_vector_store
from .milvus import MilvusVectorStoreAdapter

__all__ = [
    "VectorStoreAdapter",
    "ChromaVectorStoreAdapter",
    "MilvusVectorStoreAdapter",
    "get_vector_store",
]
