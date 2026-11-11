"""
基于 FastEmbed 的轻量本地 CPU 向量嵌入模型实现。

基于 ONNX Runtime 进行高效向量推理，默认采用 BAAI/bge-small-zh-v1.5。

@author Ateng
@since 2026-11-11
"""

import os

from fastembed import TextEmbedding


class FastEmbedProvider:
    """FastEmbed 本地轻量向量嵌入模型实现。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        self.model_name = model_name
        self._model = TextEmbedding(model_name=self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量计算多段文本的高维稠密向量。"""
        if not texts:
            return []
        raw_embeddings = self._model.embed(texts)
        return [list(map(float, emb.tolist())) for emb in raw_embeddings]

    def embed_query(self, query: str) -> list[float]:
        """计算单条查询语句的高维稠密向量。"""
        results = self.embed_texts([query])
        return results[0] if results else []
