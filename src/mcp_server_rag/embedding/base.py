"""
向量嵌入模型提供方抽象协议。

@author Ateng
@since 2026-11-11
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """向量嵌入模型提供方协议。"""

    @property
    def dimension(self) -> int:
        """获取当前嵌入模型输出的向量维度。"""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量计算多段文本的高维稠密向量。"""
        ...

    def embed_query(self, query: str) -> list[float]:
        """计算单条查询语句的高维稠密向量。"""
        ...
