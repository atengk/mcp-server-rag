"""
基于 FastEmbed ONNX TextCrossEncoder 的本地纯 CPU 交叉编码重排器实现。

利用交叉注意力模型对 (query, document) 文本对进行细粒度双向编码与打分，
并采用 Sigmoid 将原始 logits 映射到 [0.0, 1.0] 单调相关性得分区间。

@author Ateng
@since 2026-11-11
"""

import logging
import math
from typing import Any

from mcp_server_rag.reranker.base import RerankResult

logger = logging.getLogger(__name__)


def _load_cross_encoder_class() -> Any:
    """动态加载 FastEmbed TextCrossEncoder 类，推迟重型依赖导入。"""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder


def _sigmoid(val: float) -> float:
    """对原始 logit 评分应用 Sigmoid 激活，单调映射至 (0.0, 1.0) 相关度区间。"""
    if val > 50.0:
        return 1.0
    if val < -50.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-val))


class FastEmbedReranker:
    """基于 FastEmbed TextCrossEncoder 的本地二次重排驱动实现。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        """初始化 FastEmbed 重排器。

        Args:
            model_name: 支持的 FastEmbed 交叉编码模型名称，默认 ``BAAI/bge-reranker-base``。
        """
        self._model_name = model_name
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        """返回模型名称。"""
        return self._model_name

    def _get_model(self) -> Any:
        """延迟加载并单例缓存 ONNX 交叉编码器模型。"""
        if self._model is None:
            encoder_cls = _load_cross_encoder_class()
            logger.info("正在加载 FastEmbed 交叉重排模型: %s", self._model_name)
            self._model = encoder_cls(model_name=self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """对候选切片计算交叉注意力相关性得分并降序重排。

        Args:
            query: 查询语句。
            documents: 候选文档文本列表。
            top_n: 可选保留的最大重排结果数。

        Returns:
            list[RerankResult]: 按 Sigmoid 归一化得分降序排列的结果列表。
        """
        if not documents or not query.strip():
            return []

        model = self._get_model()
        raw_scores = list(model.rerank(query, documents))

        results: list[RerankResult] = []
        for idx, (doc_text, raw_score) in enumerate(zip(documents, raw_scores)):
            norm_score = round(_sigmoid(float(raw_score)), 4)
            results.append(
                RerankResult(
                    index=idx,
                    score=norm_score,
                    text=doc_text,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        if top_n is not None and top_n > 0:
            return results[:top_n]
        return results
