"""
Reranker 驱动工厂与生命周期管理模块。

依据当前运行时 Settings 中的 MCP_RAG_RERANK_PROVIDER 配置，
创建并返回 FastEmbed 本地交叉编码器或云端 HTTP API 重排器实例。

@author Ateng
@since 2026-11-11
"""

import logging
import threading

from mcp_server_rag.config import Settings
from mcp_server_rag.reranker.api_reranker import ApiReranker
from mcp_server_rag.reranker.base import RerankerProvider
from mcp_server_rag.reranker.fastembed_reranker import FastEmbedReranker

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()
_RERANKER_CACHE: dict[tuple[str, str, str, str, int], RerankerProvider] = {}


def clear_reranker_cache() -> None:
    """清空全局 Reranker 提供方实例缓存（测试与热重载使用）。"""
    with _CACHE_LOCK:
        _RERANKER_CACHE.clear()


def get_reranker_provider(settings: Settings) -> RerankerProvider:
    """根据运行时配置获取并单例缓存对应的二次重排提供者实例。

    Args:
        settings: 当前系统配置对象。

    Returns:
        RerankerProvider: 符合协议规范的重排器实例。

    Raises:
        ValueError: 当 Provider 不受支持或必要配置项缺失时抛出。
    """
    provider_name = settings.rerank_provider.strip().lower()
    cache_key = (
        provider_name,
        settings.rerank_model.strip(),
        settings.rerank_base_url.strip(),
        settings.rerank_api_key.strip(),
        settings.rerank_timeout_seconds,
    )

    with _CACHE_LOCK:
        if cache_key in _RERANKER_CACHE:
            return _RERANKER_CACHE[cache_key]

        if provider_name == "fastembed":
            instance: RerankerProvider = FastEmbedReranker(model_name=settings.rerank_model)
        elif provider_name == "api":
            if not settings.rerank_base_url:
                raise ValueError(
                    "启用 API 模式 Reranker 时必须提供 MCP_RAG_RERANK_BASE_URL 环境变量"
                )
            instance = ApiReranker(
                base_url=settings.rerank_base_url,
                model_name=settings.rerank_model,
                api_key=settings.rerank_api_key,
                timeout_seconds=settings.rerank_timeout_seconds,
            )
        else:
            raise ValueError(
                f"不支持的 Reranker Provider '{settings.rerank_provider}' "
                "(合法值: 'fastembed', 'api')"
            )

        _RERANKER_CACHE[cache_key] = instance
        return instance
