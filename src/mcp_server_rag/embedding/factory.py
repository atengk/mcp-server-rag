"""
向量嵌入模型提供方工厂。

@author Ateng
@since 2026-11-11
"""

from mcp_server_rag.config import Settings
from mcp_server_rag.embedding.base import EmbeddingProvider
from mcp_server_rag.embedding.fastembed_provider import FastEmbedProvider

# 缓存单例以避免重复加载 ONNX 模型
_CACHED_PROVIDER: EmbeddingProvider | None = None
_CACHED_MODEL_NAME: str | None = None


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """根据运行时配置获取嵌入模型实例。"""
    global _CACHED_PROVIDER, _CACHED_MODEL_NAME

    if _CACHED_PROVIDER is not None and _CACHED_MODEL_NAME == settings.embedding_model:
        return _CACHED_PROVIDER

    provider = FastEmbedProvider(model_name=settings.embedding_model)
    _CACHED_PROVIDER = provider
    _CACHED_MODEL_NAME = settings.embedding_model
    return provider
