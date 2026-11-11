"""
向量嵌入模型提供方工厂。

根据运行时配置动态初始化并缓存 FastEmbed 或 OpenAI 兼容协议嵌入提供方。

@author Ateng
@since 2026-11-11
"""

import threading
from dataclasses import dataclass

from mcp_server_rag.config import Settings
from mcp_server_rag.embedding.base import EmbeddingProvider
from mcp_server_rag.embedding.fastembed_provider import FastEmbedProvider
from mcp_server_rag.embedding.openai_provider import (
    DEFAULT_PROVIDER_BASE_URLS,
    OpenAICompatibleEmbeddingProvider,
)

_MAX_CACHE_SIZE = 32
_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class _ProviderCacheKey:
    """Embedding 提供方单例缓存键。"""

    provider_type: str
    base_url: str
    api_key: str
    model_name: str
    dimension: int | None


# 按配置签名缓存单例，避免重复加载 ONNX 模型或反复触发网络维度探针
_PROVIDER_CACHE: dict[_ProviderCacheKey, EmbeddingProvider] = {}


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """根据运行时配置获取并缓存对应的嵌入模型提供方实例。

    Args:
        settings: 全局运行时配置实例

    Returns:
        EmbeddingProvider: 满足统一向量嵌入协议的提供方实例（FastEmbed 或 OpenAI 兼容协议实现）

    Raises:
        ValueError: 当 `settings.embedding_provider` 不在支持列表或配置非法时抛出
        RuntimeError: 远端兼容接口自动维度探针探测失败时抛出
    """
    provider_type = settings.embedding_provider.strip().lower() or "fastembed"
    cache_key = _ProviderCacheKey(
        provider_type=provider_type,
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
    )

    with _CACHE_LOCK:
        cached = _PROVIDER_CACHE.get(cache_key)
        if cached is not None:
            return cached

    provider: EmbeddingProvider
    if provider_type == "fastembed":
        provider = FastEmbedProvider(model_name=settings.embedding_model)
    elif provider_type in DEFAULT_PROVIDER_BASE_URLS:
        provider = OpenAICompatibleEmbeddingProvider(
            model_name=settings.embedding_model,
            provider_type=provider_type,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            dimension=settings.embedding_dimension,
        )
    else:
        raise ValueError(
            f"不支持的 Embedding 提供方: '{settings.embedding_provider}'，"
            "可选值为: fastembed, openai, qwen, ollama, deepseek, siliconflow, openai-compatible"
        )

    with _CACHE_LOCK:
        if len(_PROVIDER_CACHE) >= _MAX_CACHE_SIZE:
            # 淘汰最早插入的缓存条目，控制内存占用上限
            oldest_key = next(iter(_PROVIDER_CACHE))
            _PROVIDER_CACHE.pop(oldest_key, None)
        _PROVIDER_CACHE[cache_key] = provider
    return provider
