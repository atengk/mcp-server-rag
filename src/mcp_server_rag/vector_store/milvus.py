"""
基于 pymilvus 的企业级分布式向量存储适配器实现。

支持按向量维度自动分集合（如 `mcp_rag_chunks_d512`、`mcp_rag_chunks_d1024`）与标量 `kb_name` 过滤，
并兼容显式固定集合名及历史单集合 `mcp_rag_chunks` 平滑过渡。

@author Ateng
@since 2026-11-11
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class MilvusVectorStoreAdapter:
    """Milvus 企业级分布式向量存储适配器。"""

    DEFAULT_COLLECTION: str = "mcp_rag_chunks"
    DEFAULT_COLLECTION_PREFIX: str = "mcp_rag_chunks"
    DEFAULT_DIMENSION: int = 512

    @property
    def requires_dimension(self) -> bool:
        """Milvus 建表必须预先确定向量维度。"""
        return True

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str = "",
        collection_name: str | None = None,
        dimension: int = DEFAULT_DIMENSION,
        client: Any | None = None,
        collection_prefix: str = DEFAULT_COLLECTION_PREFIX,
    ) -> None:
        self.uri = uri
        self.token = token
        self.fixed_collection = (collection_name or "").strip()
        self.collection_prefix = (
            collection_prefix or self.DEFAULT_COLLECTION_PREFIX
        ).strip() or self.DEFAULT_COLLECTION_PREFIX
        self.dimension = dimension
        self.collection_name = (
            self.fixed_collection or f"{self.collection_prefix}_d{self.dimension}"
        )
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
                "请运行 `pip install ateng-mcp-server-rag[milvus]` "
                "或 `uv pip install pymilvus` 进行安装。"
            ) from exc
        self._client_instance = MilvusClient(uri=self.uri, token=self.token)
        return self._client_instance

    def _normalize_dimension(self, dim: int | None) -> int:
        """规范化目标向量维度，未指定或非正数时回退至适配器默认维度。"""
        if dim is not None and int(dim) > 0:
            return int(dim)
        return self.dimension

    @staticmethod
    def _parse_positive_int(raw: Any) -> int | None:
        """将任意维度字段值安全解析为正整数，无法解析时返回 None。"""
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return raw if raw > 0 else None
        if isinstance(raw, str) and raw.strip().isdigit():
            val = int(raw.strip())
            return val if val > 0 else None
        return None

    def _get_collection_dimension(self, client: Any, collection_name: str) -> int | None:
        """读取已有 Milvus 集合的向量维度，无法识别或客户端未提供元数据时返回 None。"""
        try:
            desc = client.describe_collection(collection_name=collection_name)
        except Exception as exc:
            logger.warning("读取 Milvus 集合 '%s' Schema 失败: %s", collection_name, exc)
            return None
        if not isinstance(desc, dict):
            return None

        fields = desc.get("fields")
        if isinstance(fields, list):
            for field in fields:
                if not isinstance(field, dict):
                    continue
                params = field.get("params")
                if isinstance(params, dict):
                    dim = self._parse_positive_int(params.get("dim") or params.get("dimension"))
                    if dim is not None:
                        return dim
                if field.get("name") == "vector":
                    dim = self._parse_positive_int(field.get("dim") or field.get("dimension"))
                    if dim is not None:
                        return dim

        return self._parse_positive_int(desc.get("dimension") or desc.get("dim"))

    def resolve_collection_name(self, dimension: int | None = None) -> str:
        """根据向量维度与配置策略解析目标 Milvus 物理集合名称。

        优先级：
        1. 若配置了显式固定集合名（``MCP_RAG_MILVUS_COLLECTION``），直接返回固定集合；
        2. 优先匹配按维度隔离的集合 ``{collection_prefix}_d{dimension}``；
        3. 若维度集合尚未创建，但历史单集合 ``{collection_prefix}`` 已存在且维度兼容（一致或未知），
           自动向后兼容复用历史单集合。
        """
        if self.fixed_collection:
            return self.fixed_collection

        target_dim = self._normalize_dimension(dimension)
        dim_collection = f"{self.collection_prefix}_d{target_dim}"
        client = self._get_client()
        if client.has_collection(collection_name=dim_collection):
            return dim_collection

        if self.collection_prefix != dim_collection and client.has_collection(
            collection_name=self.collection_prefix
        ):
            legacy_dim = self._get_collection_dimension(client, self.collection_prefix)
            if legacy_dim is None or legacy_dim == target_dim:
                return self.collection_prefix

        return dim_collection

    def check_dimension_compatibility(self, dimension: int | None = None) -> dict[str, Any]:
        """校验目标向量维度与 Milvus 目标集合实际 Schema 维度是否一致。"""
        target_dim = self._normalize_dimension(dimension)
        coll_name = self.resolve_collection_name(target_dim)
        strategy = "fixed" if self.fixed_collection else "by_dimension"
        client = self._get_client()

        if not client.has_collection(collection_name=coll_name):
            return {
                "collection": coll_name,
                "strategy": strategy,
                "target_dimension": target_dim,
                "collection_dimension": None,
                "status": "uninitialized",
                "compatible": True,
            }

        actual_dim = self._get_collection_dimension(client, coll_name)
        if actual_dim is None:
            return {
                "collection": coll_name,
                "strategy": strategy,
                "target_dimension": target_dim,
                "collection_dimension": None,
                "status": "unknown",
                "compatible": True,
            }

        if actual_dim != target_dim:
            return {
                "collection": coll_name,
                "strategy": strategy,
                "target_dimension": target_dim,
                "collection_dimension": actual_dim,
                "status": "mismatch",
                "compatible": False,
                "error": (
                    f"Milvus 集合 '{coll_name}' 现有维度 ({actual_dim}) "
                    f"与当前目标向量维度 ({target_dim}) 不一致；"
                    "请清空重建该集合、切换匹配维度模型，或取消 MCP_RAG_MILVUS_COLLECTION "
                    "以启用按维度隔离集合。"
                ),
            }

        return {
            "collection": coll_name,
            "strategy": strategy,
            "target_dimension": target_dim,
            "collection_dimension": actual_dim,
            "status": "ok",
            "compatible": True,
        }

    def diagnose(self, target_dimension: int | None = None) -> dict[str, Any]:
        """生成供 rag_doctor 消费的 Milvus 集合诊断信息与维度一致性结论。"""
        if target_dimension is None:
            info = self.get_info()
            return {
                "compatible": True,
                "error": None,
                "details": {
                    "milvus_collection": info.get("collection"),
                    "milvus_collection_strategy": info.get("collection_strategy"),
                    "milvus_collection_dimension": None,
                    "milvus_dimension_status": "unknown",
                },
            }

        dim_check = self.check_dimension_compatibility(target_dimension)
        return {
            "compatible": bool(dim_check.get("compatible", True)),
            "error": dim_check.get("error"),
            "details": {
                "milvus_collection": dim_check.get("collection"),
                "milvus_collection_strategy": dim_check.get("strategy"),
                "milvus_collection_dimension": dim_check.get("collection_dimension"),
                "milvus_dimension_status": dim_check.get("status"),
            },
        }

    def _managed_collections(self, client: Any) -> list[str]:
        """枚举当前适配器管辖的所有已存在 Milvus 物理集合（含跨维度分表与历史单表）。"""
        if self.fixed_collection:
            return (
                [self.fixed_collection]
                if client.has_collection(collection_name=self.fixed_collection)
                else []
            )

        dim_pattern = re.compile(rf"^{re.escape(self.collection_prefix)}_d\d+$")
        try:
            raw_colls = client.list_collections()
        except Exception as exc:
            logger.warning("枚举 Milvus 集合列表失败，回退至候选集合探测: %s", exc)
            raw_colls = None

        if isinstance(raw_colls, (list, tuple, set)):
            return [
                c
                for c in raw_colls
                if isinstance(c, str)
                and (c == self.collection_prefix or bool(dim_pattern.match(c)))
            ]

        candidates = list(dict.fromkeys([self.collection_name, self.collection_prefix]))
        return [c for c in candidates if client.has_collection(collection_name=c)]

    def _ensure_collection(self, dim: int | None = None) -> str:
        """确保目标维度对应的向量集合存在且维度一致，若不存在则创建。"""
        compat = self.check_dimension_compatibility(dim)
        if not compat["compatible"]:
            raise ValueError(str(compat["error"]))

        coll_name = str(compat["collection"])
        if compat["status"] == "uninitialized":
            client = self._get_client()
            client.create_collection(
                collection_name=coll_name,
                dimension=int(compat["target_dimension"]),
                primary_field_name="id",
                id_type="string",
                max_length=128,
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False,
            )
        return coll_name

    def _delete_kb_vectors(self, name: str) -> None:
        """按 kb_name 过滤从所有管辖集合中清理指定知识库的向量切片。"""
        client = self._get_client()
        for coll_name in self._managed_collections(client):
            client.delete(
                collection_name=coll_name,
                filter=f'kb_name == "{name}"',
            )

    def create_kb(self, name: str, dimension: int | None = None) -> None:
        """为知识库创建底层向量存储（按目标维度路由或校验对应物理集合）。"""
        _ = name
        self._ensure_collection(dim=dimension)

    def clear_kb(self, name: str) -> None:
        """清空指定知识库在所有管辖集合中的向量数据。"""
        self._delete_kb_vectors(name)

    def delete_kb(self, name: str) -> None:
        """物理删除指定知识库在所有管辖集合中的向量切片数据。"""
        self._delete_kb_vectors(name)

    def add_chunks(
        self,
        kb_name: str,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """向维度匹配的 Milvus 集合批量写入向量切片。"""
        if not chunk_ids:
            return
        dim = len(embeddings[0]) if embeddings else self.dimension
        coll_name = self._ensure_collection(dim=dim)
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
        client.insert(collection_name=coll_name, data=rows)

    def delete_chunks(self, kb_name: str, chunk_ids: list[str]) -> None:
        """从所有管辖集合中删除指定切片 ID 列表。"""
        if not chunk_ids:
            return
        _ = kb_name
        client = self._get_client()
        for coll_name in self._managed_collections(client):
            client.delete(
                collection_name=coll_name,
                ids=chunk_ids,
            )

    def _search_single_collection(
        self,
        client: Any,
        coll_name: str,
        kb_name: str,
        query_embedding: list[float],
        top_k: int,
        filter_expr: str,
    ) -> list[dict[str, Any]]:
        """在单个兼容集合中执行向量相似度检索并标准化命中结果。"""
        try:
            search_res = client.search(
                collection_name=coll_name,
                data=[query_embedding],
                filter=filter_expr,
                limit=top_k,
                output_fields=["id", "text", "metadata", "kb_name"],
            )
        except Exception as exc:
            logger.warning("在 Milvus 集合 '%s' 检索知识库 '%s' 失败: %s", coll_name, kb_name, exc)
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

    def search(
        self,
        kb_name: str,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """在维度匹配的 Milvus 集合中执行向量相似度检索并按 kb_name 标量分区过滤。"""
        _ = where_document
        client = self._get_client()
        dim = len(query_embedding) if query_embedding else self.dimension
        coll_name = self.resolve_collection_name(dim)
        if not client.has_collection(collection_name=coll_name):
            return []

        actual_dim = self._get_collection_dimension(client, coll_name)
        if actual_dim is not None and actual_dim != dim:
            return []

        filter_expr = f'kb_name == "{kb_name}"'
        if where:
            for k, v in where.items():
                if not k.startswith("$"):
                    if isinstance(v, (str, int, float, bool)):
                        filter_expr += f" and {k} == {repr(v)}"
                    elif isinstance(v, dict) and "$eq" in v:
                        filter_expr += f" and {k} == {repr(v['$eq'])}"

        results = self._search_single_collection(
            client, coll_name, kb_name, query_embedding, top_k, filter_expr
        )

        # 若维度分表已存在但当前未命中，且历史单表仍保留同维度数据，则平滑兜底回查
        if (
            not results
            and not self.fixed_collection
            and coll_name != self.collection_prefix
            and client.has_collection(collection_name=self.collection_prefix)
        ):
            legacy_dim = self._get_collection_dimension(client, self.collection_prefix)
            if legacy_dim is not None and legacy_dim == dim:
                results = self._search_single_collection(
                    client,
                    self.collection_prefix,
                    kb_name,
                    query_embedding,
                    top_k,
                    filter_expr,
                )

        return results

    def count(self, name: str) -> int:
        """获取指定知识库在 Milvus 管辖集合中的向量总数。"""
        client = self._get_client()
        total = 0
        for coll_name in self._managed_collections(client):
            try:
                res = client.query(
                    collection_name=coll_name,
                    filter=f'kb_name == "{name}"',
                    output_fields=["count(*)"],
                )
                if (
                    res
                    and isinstance(res, list)
                    and isinstance(res[0], dict)
                    and "count(*)" in res[0]
                ):
                    total += int(res[0]["count(*)"])
                elif isinstance(res, list):
                    total += len(res)
            except Exception as exc:
                logger.warning(
                    "统计 Milvus 集合 '%s' 知识库 '%s' 数量失败: %s", coll_name, name, exc
                )
                continue
        return total

    def ping(self) -> bool:
        """自检 Milvus 集群连通性与健康状态。"""
        try:
            client = self._get_client()
            client.list_collections()
            return True
        except Exception as exc:
            logger.warning("Milvus 集群连通性探针失败: %s", exc)
            return False

    def get_info(self) -> dict[str, Any]:
        """获取 Milvus 引擎信息与诊断状态。"""
        return {
            "type": "milvus",
            "uri": self.uri,
            "collection": self.collection_name,
            "collection_strategy": "fixed" if self.fixed_collection else "by_dimension",
            "collection_prefix": self.collection_prefix,
            "status": "ok" if self.ping() else "error",
        }
