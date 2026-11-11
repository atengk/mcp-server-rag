"""
二次交叉重排 (Reranker) 抽象协议与结果模型定义。

为本地纯 CPU ONNX (FastEmbed) 与云端 HTTP API (/v1/rerank) 提供统一契约规范，
对候选检索切片进行细粒度交叉注意力重排评分。

@author Ateng
@since 2026-11-11
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class RerankResult(BaseModel):
    """单条文档重排评分结果。"""

    index: int = Field(description="对应候选列表中原始切片的索引下标")
    score: float = Field(description="交叉注意力或云端模型计算出的相关性评分")
    text: str = Field(default="", description="被重排的文档切片正文片段")


@runtime_checkable
class RerankerProvider(Protocol):
    """Reranker 二次重排提供者协议接口。"""

    @property
    def model_name(self) -> str:
        """返回当前重排器使用的模型标识。"""
        ...

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """对候选文本切片计算与查询语句的交叉注意力相关性得分。

        Args:
            query: 用户检索查询语句。
            documents: 候选文档文本列表。
            top_n: 可选保留的最大重排结果数，缺省保留全部打分切片。

        Returns:
            list[RerankResult]: 按相关性评分降序排列的重排结果列表。
        """
        ...
