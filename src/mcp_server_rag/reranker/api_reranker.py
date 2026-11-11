"""
基于标准 HTTP /v1/rerank 接口的云端重排客户端驱动实现。

兼容 SiliconFlow、Cohere、阿里百炼等行业标准 /v1/rerank 协议，
通过轻量 httpx 携带 Bearer 鉴权与严谨超时发起调用。

@author Ateng
@since 2026-11-11
"""

import logging
import re
from typing import Any

import httpx

from mcp_server_rag.reranker.base import RerankResult

logger = logging.getLogger(__name__)


def resolve_rerank_endpoint(base_url: str) -> str:
    """根据输入的基准地址解析标准化 /v1/rerank 端点 URL。"""
    clean = base_url.strip().rstrip("/")
    if not clean:
        raise ValueError("MCP_RAG_RERANK_BASE_URL 不能为空")
    if clean.endswith("/rerank"):
        return clean
    if re.search(r"/v\d+$", clean):
        return f"{clean}/rerank"
    return f"{clean}/v1/rerank"


class ApiReranker:
    """标准云端 HTTP /v1/rerank 重排器驱动实现。"""

    def __init__(
        self,
        base_url: str,
        model_name: str = "BAAI/bge-reranker-base",
        api_key: str = "",
        timeout_seconds: int = 30,
        client: httpx.Client | None = None,
    ) -> None:
        """初始化云端重排客户端。

        Args:
            base_url: 远端服务基准 URL。
            model_name: 远端部署的重排模型名称标识。
            api_key: 可选 Bearer 鉴权凭证。
            timeout_seconds: HTTP 请求超时秒数，默认 30 秒。
            client: 可选外部注入的 httpx 客户端实例（测试或连接池复用）。
        """
        self._endpoint = resolve_rerank_endpoint(base_url)
        self._model_name = model_name
        self._api_key = api_key.strip()
        self._timeout_seconds = max(1, timeout_seconds)
        self._injected_client = client

    @property
    def model_name(self) -> str:
        """返回模型名称。"""
        return self._model_name

    def _build_headers(self) -> dict[str, str]:
        """构造鉴权与内容格式请求头。"""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """通过远程 /v1/rerank 接口计算相关度打分并降序返回。

        Args:
            query: 查询语句。
            documents: 待重排的文档切片正文列表。
            top_n: 可选保留的最大重排结果数。

        Returns:
            list[RerankResult]: 按相关度降序排列的重排结果列表。

        Raises:
            RuntimeError: 当网络超时、HTTP 非 2xx 或响应结构异常时抛出。
        """
        if not documents or not query.strip():
            return []

        payload: dict[str, Any] = {
            "model": self._model_name,
            "query": query,
            "documents": documents,
        }
        if top_n is not None and top_n > 0:
            payload["top_n"] = top_n

        timeout = httpx.Timeout(
            connect=float(self._timeout_seconds),
            read=float(self._timeout_seconds),
            write=float(self._timeout_seconds),
            pool=float(self._timeout_seconds),
        )

        try:
            if self._injected_client is not None:
                response = self._injected_client.post(
                    self._endpoint,
                    headers=self._build_headers(),
                    json=payload,
                    timeout=timeout,
                )
            else:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(
                        self._endpoint,
                        headers=self._build_headers(),
                        json=payload,
                    )
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"请求云端 Rerank 服务超时 ({self._endpoint}): {exc}") from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300] if exc.response is not None else str(exc)
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            raise RuntimeError(
                f"云端 Rerank 服务返回错误状态码 ({status_code}, "
                f"endpoint={self._endpoint}): {detail}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"调用云端 Rerank 服务失败 ({self._endpoint}): {exc}") from exc

        raw_results = data.get("results") or data.get("data") or []
        results: list[RerankResult] = []
        for i, item in enumerate(raw_results):
            idx = item.get("index") if item.get("index") is not None else i
            raw_score = item.get("relevance_score")
            if raw_score is None:
                raw_score = item.get("score", 0.0)
            score = round(float(raw_score), 4)
            text = documents[idx] if 0 <= idx < len(documents) else ""
            results.append(RerankResult(index=int(idx), score=score, text=text))

        results.sort(key=lambda r: r.score, reverse=True)
        if top_n is not None and top_n > 0:
            return results[:top_n]
        return results
