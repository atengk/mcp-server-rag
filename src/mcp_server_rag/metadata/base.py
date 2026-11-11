"""
元数据目录抽象协议接口定义。

定义可插拔元数据存储引擎（SQLite、PostgreSQL、MySQL）的统一行为契约，
实现上层 RAG 服务与底层关系型数据库引擎的解耦。

@author Ateng
@since 2026-11-11
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MetadataCatalogAdapter(Protocol):
    """可插拔元数据目录引擎抽象协议接口。

    所有元数据目录实现（如内置 SQLite、PostgreSQL、MySQL）必须遵循此契约，
    以确保知识库、文档指纹、切片链表与文档图片资产的生命周期管理语义一致。
    """

    def create_kb(
        self,
        name: str,
        description: str = "",
        embedding_model: str = "",
        embedding_dimension: int = 0,
        owner: str = "default",
        visibility: str = "public",
    ) -> bool:
        """创建新知识库元数据记录。

        Args:
            name: 知识库唯一名称标识。
            description: 知识库描述说明。
            embedding_model: 绑定的嵌入模型标识。
            embedding_dimension: 绑定的向量维度，未确定时为 0。
            owner: 知识库所属创建者标识，默认 ``"default"``。
            visibility: 可见性范围（``"public"`` 或 ``"private"``），默认 ``"public"``。

        Returns:
            创建成功返回 ``True``；若同名知识库已存在则返回 ``False``。

        Raises:
            ValueError: 当 ``visibility`` 等扩展字段取值非法时抛出。
        """
        ...

    def get_kb(self, name: str) -> dict[str, Any] | None:
        """按名称查询单个知识库元数据及文档/切片聚合统计。

        Args:
            name: 目标知识库名称。

        Returns:
            包含 ``name``、``description``、``embedding_model``、``embedding_dimension``、
            ``owner``、``visibility``、``created_at``、``updated_at``、``document_count``、
            ``chunk_count`` 的字典；未找到时返回 ``None``。
        """
        ...

    def list_kbs(self) -> list[dict[str, Any]]:
        """枚举所有知识库列表及其文档与切片统计。

        Returns:
            按创建时间升序排列的知识库元数据字典列表；无记录时返回空列表 ``[]``。
        """
        ...

    def clear_kb(self, name: str) -> bool:
        """清空指定知识库的全部文档、切片与关联图片资产记录，保留知识库本身配置。

        Args:
            name: 目标知识库名称。

        Returns:
            知识库存在并成功清空返回 ``True``；知识库不存在返回 ``False``。
        """
        ...

    def update_kb_embedding_model(
        self,
        name: str,
        embedding_model: str,
        embedding_dimension: int = 0,
    ) -> None:
        """更新知识库绑定的嵌入模型标识与维度。

        Args:
            name: 目标知识库名称。
            embedding_model: 新的嵌入模型标识。
            embedding_dimension: 新的向量维度，默认 ``0`` 表示不指定或待探测。
        """
        ...

    def delete_kb(self, name: str) -> bool:
        """物理删除指定知识库及其关联的文档、切片与图片资产记录。

        Args:
            name: 待删除的知识库名称。

        Returns:
            知识库存在并成功删除返回 ``True``；知识库不存在返回 ``False``。
        """
        ...

    def get_document_by_title(self, kb_name: str, title: str) -> dict[str, Any] | None:
        """根据知识库名称与文档标题精确查找文档记录。

        Args:
            kb_name: 所属知识库名称。
            title: 文档标题。

        Returns:
            文档元数据字典；未匹配时返回 ``None``。
        """
        ...

    def get_document_by_id(self, kb_name: str, doc_id: str) -> dict[str, Any] | None:
        """根据知识库名称与文档唯一 ID 查找文档记录。

        Args:
            kb_name: 所属知识库名称。
            doc_id: 文档唯一标识符。

        Returns:
            文档元数据字典；未匹配时返回 ``None``。
        """
        ...

    def find_document(self, kb_name: str, doc_identifier: str) -> dict[str, Any] | None:
        """多策略查找文档：依次按文档 ID、规范化路径、源 URI 或标题匹配。

        Args:
            kb_name: 所属知识库名称。
            doc_identifier: 文档 ID、本地文件路径、源 URI 或文档标题。

        Returns:
            匹配的文档元数据字典；未找到时返回 ``None``。
        """
        ...

    def delete_document(self, kb_name: str, doc_id: str) -> list[str]:
        """删除指定文档及其关联切片与图片资产记录。

        Args:
            kb_name: 所属知识库名称。
            doc_id: 待删除的文档唯一标识符。

        Returns:
            被删除切片的唯一 ID 列表；若文档无切片或不存在则返回空列表 ``[]``。
        """
        ...

    def register_document_and_chunks(
        self,
        kb_name: str,
        doc_id: str,
        title: str,
        content_hash: str,
        source_uri: str,
        chunk_ids: list[str],
        file_size: int,
        chunk_contents: list[str] | None = None,
        chunk_images: list[list[str]] | None = None,
    ) -> None:
        """原子注册或更新文档及其切片链表记录。

        Args:
            kb_name: 所属知识库名称。
            doc_id: 文档唯一标识符。
            title: 文档标题。
            content_hash: 文档内容的 SHA-256 指纹摘要。
            source_uri: 文档来源路径或 URL。
            chunk_ids: 按顺序排列的切片 ID 列表，用于持久化 ``prev_chunk_id`` / ``next_chunk_id``。
            file_size: 原始文档字节大小。
            chunk_contents: 与 ``chunk_ids`` 等长的切片文本列表；为 ``None`` 时退化为空字符串。
            chunk_images: 与 ``chunk_ids`` 等长的切片关联图片路径列表；为 ``None`` 时为空列表。
        """
        ...

    def register_image_assets(
        self,
        kb_name: str,
        doc_id: str,
        assets: list[dict[str, Any]],
    ) -> None:
        """批量幂等登记文档提取出的图片资产记录。

        Args:
            kb_name: 所属知识库名称。
            doc_id: 所属文档唯一标识符。
            assets: 图片资产字典列表，支持 ``asset_id``、``filename``、``storage_uri``、
                ``mime_type`` 与 ``byte_size`` 字段。
        """
        ...

    def get_image_asset(
        self,
        kb_name: str,
        doc_id: str,
        filename: str,
    ) -> dict[str, Any] | None:
        """查询指定文档下某图片文件名的资产登记信息。

        Args:
            kb_name: 所属知识库名称。
            doc_id: 所属文档唯一标识符。
            filename: 图片文件名或相对标识。

        Returns:
            图片资产元数据字典；未找到时返回 ``None``。
        """
        ...

    def list_image_assets(
        self,
        kb_name: str,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出指定知识库（或限定文档）已登记的图片资产清单。

        Args:
            kb_name: 所属知识库名称。
            doc_id: 可选文档 ID；为 ``None`` 时返回整个知识库的图片资产列表。

        Returns:
            图片资产元数据字典列表；无匹配时返回空列表 ``[]``。
        """
        ...

    def count_kbs(self) -> int:
        """统计当前目录中的知识库总数。

        Returns:
            知识库数量（非负整数）。
        """
        ...

    def count_documents(self, kb_name: str | None = None) -> int:
        """统计文档总数。

        Args:
            kb_name: 可选知识库名称；为 ``None`` 时统计所有知识库的文档总数。

        Returns:
            文档数量（非负整数）。
        """
        ...

    def ping(self) -> bool:
        """探测元数据存储引擎的健康连通性。

        Returns:
            连通正常返回 ``True``，异常返回 ``False``。
        """
        ...
