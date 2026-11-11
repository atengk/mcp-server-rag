"""
通用 OpenAI 兼容协议向量嵌入提供方。

通过轻量 httpx 客户端调用标准 `/v1/embeddings` 接口，统一适配 OpenAI、
通义千问 (DashScope)、DeepSeek、SiliconFlow 及本地 Ollama 等服务，
支持轻量探针全自动识别输出维度与可选 MRL 降维覆写。

@author Ateng
@since 2026-11-11
"""

from typing import Any

import httpx

DEFAULT_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openai-compatible": "https://api.openai.com/v1",
    "openai_compatible": "https://api.openai.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "ollama": "http://localhost:11434/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
}


class OpenAICompatibleEmbeddingProvider:
    """OpenAI 兼容协议向量嵌入模型实现。"""

    def __init__(
        self,
        model_name: str,
        provider_type: str = "openai",
        base_url: str = "",
        api_key: str = "",
        dimension: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        """初始化 OpenAI 兼容协议嵌入提供方并完成维度确定。

        若未显式传入 `dimension`，将在构造阶段立即发起单条轻量探针请求自动探测模型维度。

        Args:
            model_name: 远端或本地服务中的嵌入模型标识（不能为空）
            provider_type: 提供方别名，用于解析缺省 Base URL（如 openai/qwen/ollama/deepseek）
            base_url: 自定义服务端点基地址；为空时按 `provider_type` 自动映射默认端点
            api_key: Bearer 鉴权密钥（Ollama 等本地免密服务可为空）
            dimension: 可选的显式 MRL 降维维度（正整数）；为空时自动探测
            timeout: 单次 HTTP 请求与读取超时秒数，默认 60.0 秒

        Raises:
            ValueError: 模型名称为空或显式 `dimension` 非正整数时抛出
            RuntimeError: 自动维度探测请求失败或返回空向量时抛出
        """
        if not model_name or not model_name.strip():
            raise ValueError("Embedding 模型名称 (MCP_RAG_EMBEDDING_MODEL) 不能为空")
        if dimension is not None and dimension < 1:
            raise ValueError("显式 Embedding 维度 (MCP_RAG_EMBEDDING_DIMENSION) 必须为正整数")

        self.provider_type = provider_type.strip().lower()
        self.model_name = model_name.strip()
        self.api_key = api_key.strip()
        self.timeout = timeout
        self._explicit_dimension = dimension

        resolved_base = (
            base_url.strip()
            or DEFAULT_PROVIDER_BASE_URLS.get(self.provider_type, "https://api.openai.com/v1")
        ).rstrip("/")
        if resolved_base.endswith("/embeddings"):
            self.endpoint = resolved_base
        else:
            self.endpoint = f"{resolved_base}/embeddings"

        if dimension is not None:
            self._dimension: int = dimension
        else:
            # 1. 初始化时通过单条轻量探针自动探测模型输出维度
            probe_vec = self.embed_query("dimension_probe")
            if not probe_vec:
                raise RuntimeError(
                    f"无法自动探测模型 '{self.model_name}' 的向量维度：远端返回空向量"
                )
            self._dimension = len(probe_vec)

    @property
    def dimension(self) -> int:
        """获取当前嵌入模型的输出向量维度。"""
        return self._dimension

    def _build_headers(self) -> dict[str, str]:
        """构造标准 HTTP 请求头。"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量计算多段文本的高维稠密向量。

        Args:
            texts: 待向量化的文本列表；为空列表时直接返回 `[]` 而不发起网络请求

        Returns:
            list[list[float]]: 与输入顺序严格一致的稠密向量列表

        Raises:
            ValueError: 远端返回向量的实际维度与显式配置 `dimension` 不一致时抛出
            RuntimeError: 网络超时、HTTP 非 2xx 状态码或响应 JSON 格式异常时抛出
        """
        if not texts:
            return []

        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
        }
        if self._explicit_dimension is not None:
            payload["dimensions"] = self._explicit_dimension

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint,
                    headers=self._build_headers(),
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"请求 Embedding 服务超时 ({self.endpoint}): {exc}") from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300] if exc.response is not None else str(exc)
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            raise RuntimeError(
                f"Embedding 服务返回异常状态码 ({status_code}, endpoint={self.endpoint}): {detail}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"调用 Embedding 服务失败 ({self.endpoint}): {exc}") from exc

        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            raise RuntimeError(
                f"Embedding 响应格式异常或条数不匹配：期望 {len(texts)} 条，实际返回 "
                f"{len(items) if isinstance(items, list) else '非列表'}"
            )

        sorted_items = sorted(items, key=lambda x: int(x.get("index", 0)))
        vectors: list[list[float]] = []
        for item in sorted_items:
            raw_vec = item.get("embedding")
            if not isinstance(raw_vec, list) or not raw_vec:
                raise RuntimeError("Embedding 响应条目的 embedding 字段缺失或为空")
            vec = [float(v) for v in raw_vec]
            if self._explicit_dimension is not None and len(vec) != self._explicit_dimension:
                raise ValueError(
                    f"Embedding 返回维度 ({len(vec)}) 与显式配置的 "
                    f"MCP_RAG_EMBEDDING_DIMENSION ({self._explicit_dimension}) 不一致"
                )
            vectors.append(vec)

        return vectors

    def embed_query(self, query: str) -> list[float]:
        """计算单条查询语句的高维稠密向量。

        Args:
            query: 单条检索查询文本

        Returns:
            list[float]: 单条稠密向量（空输入时返回空列表）
        """
        results = self.embed_texts([query])
        return results[0] if results else []
