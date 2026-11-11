"""
向量存储适配器模块。

@author Ateng
@since 2026-11-11
"""

from .base import VectorStoreAdapter
from .chroma import ChromaVectorStoreAdapter

__all__ = ["VectorStoreAdapter", "ChromaVectorStoreAdapter"]
