"""
向量嵌入模型模块。

@author Ateng
@since 2026-11-11
"""

from .base import EmbeddingProvider
from .factory import get_embedding_provider
from .fastembed_provider import FastEmbedProvider
from .openai_provider import OpenAICompatibleEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "FastEmbedProvider",
    "OpenAICompatibleEmbeddingProvider",
    "get_embedding_provider",
]
