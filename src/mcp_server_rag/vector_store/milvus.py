"""
基于 pymilvus 的企业级分布式向量存储适配器实现。

通过统一集合 `mcp_rag_chunks` 与标量 `kb_name` 过滤实现高性能多知识库隔离检索。

@author Ateng
@since 2026-11-11
"""

from typing import Any


class MilvusVectorStoreAdapter:
    """Milvus 企业级分布式向量存储适配器。"""

    DEFAULT_COLLECTION: str = "mcp_rag_chunks"
    DEFAULT_DIMENSION: int = 512

    @property
    def requires_dimension(self) -> bool:
        """Milvus 建表必须预先确定向量维度。"""
        return True

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str = "",
        collection_name: str = DEFAULT_COLLECTION,
        dimension: int = DEFAULT_DIMENSION,
        client: Any | None = None,
    ) -> None:
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.dimension = dimension
        self._client_instance = client

    def _get_client(self) -> Any:
        """获取或惰性初始化 pymilvus.MilvusClient 客户端。"""
        if self._client_instance is not None:
            return self._client_instance
        try:
            from pymilvus import MilvusClient  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "未检测到 pymilvus 依赖，无法使用 Milvus 向量存储。"
                "请运行 `pip install atengk-mcp-server-rag[milvus]` "
                "或 `uv pip install pymilvus` 进行安装。"
            ) from exc
        self._client_instance = MilvusClient(uri=self.uri, token=self.token)
        return self._client_instance

    def _ensure_collection(self, dim: int | None = None) -> None:
        """确保统一向量集合存在，若不存在则创建。"""
        client = self._get_client()
        if not client.has_collection(collection_name=self.collection_name):
            dimension = dim or self.dimension
            client.create_collection(
                collection_name=self.collection_name,
                dimension=dimension,
                primary_field_name="id",
                id_type="string",
                max_length=128,
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False,
            )

    def create_kb(self, name: str, dimension: int | None = None) -> None:
        """为知识库创建底层向量存储（统一物理集合，按实时探测维度初始化）。"""
        self._ensure_collection(dim=dimension)

    def clear_kb(self, name: str) -> None:
        """清空指定知识库的所有向量数据。"""
        client = self._get_client()
        if client.has_collection(collection_name=self.collection_name):
            client.delete(
                collection_name=self.collection_name,
                filter=f'kb_name == "{name}"',
            )

    def delete_kb(self, name: str) -> None:
        """物理删除指定知识库的向量切片数据。"""
        client = self._get_client()
        if client.has_collection(collection_name=self.collection_name):
            client.delete(
                collection_name=self.collection_name,
                filter=f'kb_name == "{name}"',
            )

    def add_chunks(
        self,
        kb_name: str,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """向 Milvus 统一集合批量写入向量切片。"""
        if not chunk_ids:
            return
        dim = len(embeddings[0]) if embeddings else self.dimension
        self._ensure_collection(dim=dim)
        client = self._get_client()

        rows: list[dict[str, Any]] = []
        for cid, txt, emb, meta in zip(chunk_ids, texts, embeddings, metadatas):
            safe_meta: dict[str, Any] = {}
            for k, v in (meta or {}).items():
                if isinstance(v, (str, int, float, bool)):
                    safe_meta[k] = v
                else:
                    safe_meta[k] = str(v)
            rows.append(
                {
                    "id": str(cid),
                    "vector": emb,
                    "kb_name": str(kb_name),
                    "text": str(txt or ""),
                    "metadata": safe_meta,
                }
            )
        client.insert(collection_name=self.collection_name, data=rows)

    def delete_chunks(self, kb_name: str, chunk_ids: list[str]) -> None:
        """从知识库中删除指定切片 ID 列表。"""
        if not chunk_ids:
            return
        client = self._get_client()
        if client.has_collection(collection_name=self.collection_name):
            client.delete(
                collection_name=self.collection_name,
                ids=chunk_ids,
            )

    def search(
        self,
        kb_name: str,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """在 Milvus 统一集合中执行向量相似度检索并按 kb_name 标量分区过滤。"""
        client = self._get_client()
        if not client.has_collection(collection_name=self.collection_name):
            return []

        filter_expr = f'kb_name == "{kb_name}"'
        if where:
            for k, v in where.items():
                if not k.startswith("$"):
                    if isinstance(v, (str, int, float, bool)):
                        filter_expr += f" and {k} == {repr(v)}"
                    elif isinstance(v, dict) and "$eq" in v:
                        filter_expr += f" and {k} == {repr(v['$eq'])}"

        try:
            search_res = client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                filter=filter_expr,
                limit=top_k,
                output_fields=["id", "text", "metadata", "kb_name"],
            )
        except Exception:
            return []

        hits = search_res[0] if search_res else []
        results: list[dict[str, Any]] = []
        for hit in hits:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
            cid = hit.get("id") or entity.get("id", "")
            text = entity.get("text") or hit.get("text", "")
            meta = entity.get("metadata") or hit.get("metadata", {})
            kbn = entity.get("kb_name") or hit.get("kb_name", kb_name)
            dist = hit.get("distance", hit.get("score", 0.0))
            clamped_score = round(max(0.0, min(1.0, float(dist))), 4)
            results.append(
                {
                    "chunk_id": str(cid),
                    "text": str(text),
                    "score": clamped_score,
                    "metadata": meta if isinstance(meta, dict) else {},
                    "kb_name": str(kbn),
                }
            )

        return results

    def count(self, name: str) -> int:
        """获取指定知识库在 Milvus 中的向量总数。"""
        client = self._get_client()
        if not client.has_collection(collection_name=self.collection_name):
            return 0
        try:
            res = client.query(
                collection_name=self.collection_name,
                filter=f'kb_name == "{name}"',
                output_fields=["count(*)"],
            )
            if res and isinstance(res, list) and isinstance(res[0], dict) and "count(*)" in res[0]:
                return int(res[0]["count(*)"])
            return len(res) if isinstance(res, list) else 0
        except Exception:
            return 0

    def ping(self) -> bool:
        """自检 Milvus 集群连通性与健康状态。"""
        try:
            client = self._get_client()
            client.list_collections()
            return True
        except Exception:
            return False

    def get_info(self) -> dict[str, Any]:
        """获取 Milvus 引擎信息与诊断状态。"""
        return {
            "type": "milvus",
            "uri": self.uri,
            "collection": self.collection_name,
            "status": "ok" if self.ping() else "error",
        }
