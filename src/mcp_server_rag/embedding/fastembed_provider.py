"""
基于 FastEmbed 的轻量本地 CPU 向量嵌入模型实现。

基于 ONNX Runtime 进行高效向量推理，默认采用 BAAI/bge-small-zh-v1.5。

@author Ateng
@since 2026-11-11
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class FastEmbedProvider:
    """FastEmbed 本地轻量向量嵌入模型实现。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        """初始化 FastEmbed 本地推理模型。

        Args:
            model_name: FastEmbed 支持的本地嵌入模型名称。
        """
        # 必须在首次导入 fastembed / huggingface_hub 之前设置镜像源，防止锁定默认官网端点
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from fastembed import TextEmbedding

        self.model_name = model_name
        try:
            self._model: Any = TextEmbedding(model_name=self.model_name)
        except Exception as exc:
            # 离线或远端校验超时时记录日志并回退读取本地已缓存模型
            logger.warning(
                "拉取 FastEmbed 模型 '%s' 失败，尝试回退读取本地离线缓存: %s",
                self.model_name,
                exc,
            )
            self._model = TextEmbedding(model_name=self.model_name, local_files_only=True)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """获取当前本地模型的输出向量维度（按需触发首次探针并缓存）。"""
        if self._dimension is None:
            probe_vec = self.embed_query("dimension_probe")
            self._dimension = len(probe_vec)
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量计算多段文本的高维稠密向量。"""
        if not texts:
            return []
        raw_embeddings = self._model.embed(texts)
        vectors = [list(map(float, emb.tolist())) for emb in raw_embeddings]
        if self._dimension is None and vectors:
            self._dimension = len(vectors[0])
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """计算单条查询语句的高维稠密向量。"""
        results = self.embed_texts([query])
        return results[0] if results else []
