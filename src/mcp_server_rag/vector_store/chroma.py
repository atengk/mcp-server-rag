"""
基于 ChromaDB 的轻量嵌入式向量存储适配器实现。

通过本地 PersistentClient 进行多知识库集合的读写与生命周期管理。

@author Ateng
@since 2026-11-11
"""

from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.api import ClientAPI


class ChromaVectorStoreAdapter:
    """ChromaDB 本地持久化向量存储适配器。"""

    def __init__(self, data_dir: Path) -> None:
        self.chroma_dir = data_dir / "chroma"
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        # 初始化持久化客户端
        self._client: ClientAPI = chromadb.PersistentClient(path=str(self.chroma_dir))

    @staticmethod
    def _collection_name(kb_name: str) -> str:
        """格式化知识库对应的 Chroma 集合名称。

        Chroma 要求集合名称满足 3-63 字符，只含字母数字下划线连字符。
        """
        clean_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in kb_name)
        return f"mcp_kb_{clean_name}"[:63]

    def create_kb(self, name: str) -> None:
        """创建知识库对应集合。"""
        coll_name = self._collection_name(name)
        self._client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"},
        )

    def clear_kb(self, name: str) -> None:
        """清空知识库所有向量。"""
        coll_name = self._collection_name(name)
        try:
            self._client.delete_collection(name=coll_name)
        except Exception:
            pass
        self._client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_kb(self, name: str) -> None:
        """物理删除知识库集合。"""
        coll_name = self._collection_name(name)
        try:
            self._client.delete_collection(name=coll_name)
        except Exception:
            pass

    def add_chunks(
        self,
        kb_name: str,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """向知识库批量写入向量切片。"""
        if not chunk_ids:
            return
        coll_name = self._collection_name(kb_name)
        collection = self._client.get_or_create_collection(name=coll_name)
        safe_metadatas = []
        for m in metadatas:
            safe_m = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    safe_m[k] = v
                else:
                    safe_m[k] = str(v)
            safe_metadatas.append(safe_m)
        collection.add(
            ids=chunk_ids,
            documents=texts,
            embeddings=cast(Any, embeddings),
            metadatas=cast(Any, safe_metadatas),
        )

    def delete_chunks(self, kb_name: str, chunk_ids: list[str]) -> None:
        """从知识库中删除指定切片 ID 列表。"""
        if not chunk_ids:
            return
        coll_name = self._collection_name(kb_name)
        try:
            collection = self._client.get_collection(name=coll_name)
            collection.delete(ids=chunk_ids)
        except Exception:
            pass

    def count(self, name: str) -> int:
        """获取知识库中的向量总数。"""
        coll_name = self._collection_name(name)
        try:
            collection = self._client.get_collection(name=coll_name)
            return int(collection.count())
        except Exception:
            return 0

    def search(
        self,
        kb_name: str,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """在知识库对应集合中执行向量相似度检索。

        Args:
            kb_name: 知识库名称
            query_embedding: 查询的高维嵌入向量
            top_k: 返回候选切片数上限
            where: 可选的元数据过滤字典
            where_document: 可选的文档内容过滤字典

        Returns:
            list[dict[str, Any]]: 排序后的切片结果列表
        """
        coll_name = self._collection_name(kb_name)
        try:
            collection = self._client.get_collection(name=coll_name)
        except Exception:
            return []

        total_count = collection.count()
        if total_count == 0 or top_k <= 0:
            return []

        n_results = min(top_k, total_count)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        try:
            query_res = collection.query(**kwargs)
        except Exception:
            return []

        ids_list = query_res.get("ids", [[]])
        docs_list = query_res.get("documents", [[]])
        metas_list = query_res.get("metadatas", [[]])
        dists_list = query_res.get("distances", [[]])

        ids = ids_list[0] if ids_list else []
        docs = docs_list[0] if docs_list else []
        metas = metas_list[0] if metas_list else []
        dists = dists_list[0] if dists_list else []

        coll_meta = collection.metadata or {}
        is_cosine = coll_meta.get("hnsw:space") == "cosine"

        results: list[dict[str, Any]] = []
        for cid, text, m, dist in zip(ids, docs, metas, dists):
            if dist is None:
                score = 0.0
            elif is_cosine:
                score = 1.0 - float(dist)
            else:
                score = 1.0 - (float(dist) / 2.0)

            clamped_score = round(max(0.0, min(1.0, score)), 4)
            results.append(
                {
                    "chunk_id": str(cid),
                    "text": str(text or ""),
                    "score": clamped_score,
                    "metadata": m or {},
                    "kb_name": kb_name,
                }
            )

        return results

    def ping(self) -> bool:
        """测试 ChromaDB 客户端健康度。"""
        try:
            return bool(self._client.heartbeat() > 0)
        except Exception:
            return False

    def get_info(self) -> dict[str, Any]:
        """获取 ChromaDB 诊断信息。"""
        return {
            "type": "chromadb",
            "storage_path": str(self.chroma_dir),
            "status": "ok" if self.ping() else "error",
            "heartbeat": self._client.heartbeat(),
        }
