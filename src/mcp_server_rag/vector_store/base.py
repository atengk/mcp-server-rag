"""
向量存储适配器抽象协议。

定义知识库创建、清空、销毁与向量增删改查的统一抽象契约。

@author Ateng
@since 2026-11-11
"""

from typing import Any, Protocol


class VectorStoreAdapter(Protocol):
    """向量存储适配器协议接口。"""

    def create_kb(self, name: str) -> None:
        """为知识库创建底层向量集合或分区。"""
        ...

    def clear_kb(self, name: str) -> None:
        """清空指定知识库的所有向量数据。"""
        ...

    def delete_kb(self, name: str) -> None:
        """物理删除指定知识库的向量集合或分区。"""
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
