"""
向量存储适配器抽象协议。

定义知识库创建、清空、销毁与向量增删改查的统一抽象契约。

@author Ateng
@since 2026-11-11
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorStoreAdapter(Protocol):
    """向量存储适配器协议接口。"""

    @property
    def requires_dimension(self) -> bool:
        """指示初始化建库集合时是否强依赖已知的向量维度。"""
        ...

    def create_kb(self, name: str, dimension: int | None = None) -> None:
        """为知识库创建底层向量集合或分区。

        Args:
            name: 知识库名称
            dimension: 可选的初始向量维度，由 Embedding 提供方自动探测注入
        """
        ...

    def clear_kb(self, name: str) -> None:
        """清空指定知识库的所有向量数据。"""
        ...

    def delete_kb(self, name: str) -> None:
        """物理删除指定知识库的向量集合或分区。"""
        ...

    def add_chunks(
        self,
        kb_name: str,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """向知识库批量写入向量切片。"""
        ...

    def delete_chunks(self, kb_name: str, chunk_ids: list[str]) -> None:
        """从知识库中删除指定切片 ID 列表。"""
        ...

    def search(
        self,
        kb_name: str,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """检索指定知识库中最相似的切片。

        Args:
            kb_name: 目标知识库名称
            query_embedding: 查询的高维嵌入向量
            top_k: 返回候选切片数上限
            where: 可选的元数据过滤条件
            where_document: 可选的文档正文过滤条件

        Returns:
            list[dict[str, Any]]: 包含 chunk_id, text, score, metadata, kb_name 的切片字典列表
        """
        ...

    def count(self, name: str) -> int:
        """获取指定知识库的向量总数。"""
        ...

    def ping(self) -> bool:
        """自检底层向量存储引擎连通性与健康状态。"""
        ...

    def get_info(self) -> dict[str, Any]:
        """获取向量存储引擎信息与诊断状态。"""
        ...
