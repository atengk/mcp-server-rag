"""
FastMCP 服务主实现模块。

注册并暴露知识库生命周期管理、健康自检探针与核心检索工具。

@author Ateng
@since 2026-11-11
"""

import hashlib
import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import httpx
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError
from mcp.server.mcpserver.resources.templates import ResourceTemplate
from mcp.server.mcpserver.resources.types import FunctionResource
from mcp.types import Resource as MCPResource

from mcp_server_rag.asset_gateway import (
    PreparedDocumentAssets,
    cleanup_document_gateway_assets,
    cleanup_kb_gateway_assets,
    cleanup_stale_gateway_assets,
    is_text_markup_document,
    list_gateway_asset_descriptors,
    persist_and_register_document_assets,
    prepare_document_assets,
    read_gateway_image_asset,
)
from mcp_server_rag.config import Settings, get_settings
from mcp_server_rag.embedding import get_embedding_provider
from mcp_server_rag.hybrid_search import HybridSearchOptions, execute_hybrid_search
from mcp_server_rag.metadata import MetadataCatalogAdapter, get_metadata_catalog
from mcp_server_rag.office import (
    ensure_office_dependency,
    is_office_document,
    parse_office_document,
)
from mcp_server_rag.parsers import (
    build_embedding_text,
    dispatch_and_split,
    extract_html_content,
    extract_images_from_text,
    split_markdown,
)
from mcp_server_rag.splitter import recursive_split_text
from mcp_server_rag.vector_store import VectorStoreAdapter, get_vector_store

# 兼容 FastMCP 类型别名
FastMCP = MCPServer

logger = logging.getLogger(__name__)

READ_ONLY_BLOCK_MSG: str = (
    "只读模式已启用，拒绝执行写操作 (Write operations disabled in read-only mode)"
)
MAX_EXPAND_CONTEXT: int = 5


def _get_services() -> tuple[MetadataCatalogAdapter, VectorStoreAdapter]:
    """获取当前配置对应的元数据目录与向量存储适配器实例。"""
    settings = get_settings()
    catalog = get_metadata_catalog(settings)
    vector_store = get_vector_store(settings)
    return catalog, vector_store


def _validate_kb_name(name: str) -> tuple[bool, str]:
    """校验知识库名称合规性（字母数字下划线与连字符，1~64 字符）。"""
    if not name or len(name) > 64:
        return False, "知识库名称长度必须在 1 到 64 个字符之间"
    if not re.match(r"^[a-zA-Z0-9_\-]+$", name):
        return False, "知识库名称仅支持英文字母、数字、下划线 (_) 与连字符 (-)"
    return True, ""


def _validate_kb_for_write(
    catalog: MetadataCatalogAdapter,
    kb_name: str,
    settings: Settings,
) -> dict[str, Any] | None:
    """校验目标知识库存在性及运行时 Embedding 模型一致性，防止跨模型向量污染。"""
    kb = catalog.get_kb(kb_name)
    if not kb:
        return {
            "success": False,
            "message": f"Knowledge base '{kb_name}' not found (知识库不存在)",
        }

    kb_model = str(kb.get("embedding_model") or "").strip()
    current_model = settings.embedding_model.strip()
    if not kb_model or kb_model == current_model:
        return None

    has_history = (int(kb.get("document_count") or 0) > 0) or (int(kb.get("chunk_count") or 0) > 0)
    if has_history:
        return {
            "success": False,
            "message": (
                f"知识库 '{kb_name}' 已绑定模型 '{kb_model}' 并包含历史数据，"
                f"与当前运行时模型 '{current_model}' 不一致，禁止跨模型混合写入以防向量空间污染；"
                "请切回原模型或新建知识库。"
            ),
        }

    # 空知识库允许随当前运行时模型更新绑定，并同步更新向量维度
    target_dim = settings.embedding_dimension
    if target_dim is None:
        try:
            target_dim = get_embedding_provider(settings).dimension
        except Exception as exc:
            logger.warning("空知识库 '%s' 切换模型时探测 Embedding 维度失败: %s", kb_name, exc)
            target_dim = 0
    catalog.update_kb_embedding_model(
        kb_name,
        current_model,
        embedding_dimension=int(target_dim or 0),
    )
    return None


@dataclass(frozen=True)
class _DocumentIndexSpec:
    """文档切片索引与双存储持久化入参载荷。"""

    kb_name: str
    doc_id: str
    title: str
    content_hash: str
    source_uri: str
    file_size: int
    chunks: list[str]
    chunk_ids: list[str]
    user_metadata: dict[str, Any] | None = None
    extra_metadata: dict[str, Any] | None = None
    precomputed_embeddings: list[list[float]] | None = None


def _compute_file_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """分块流式计算任意文件（含大型二进制文档）的 SHA-256 指纹摘要。"""
    hasher = hashlib.sha256()
    with file_path.open("rb") as fp:
        while block := fp.read(chunk_size):
            hasher.update(block)
    return hasher.hexdigest()


def _persist_indexed_document(
    settings: Settings,
    catalog: MetadataCatalogAdapter,
    vector_store: VectorStoreAdapter,
    doc: _DocumentIndexSpec,
) -> None:
    """统一执行 Alt 语义向量构建、图片清单提取、向量写入与元数据链表持久化。"""
    if doc.precomputed_embeddings is not None:
        embeddings = doc.precomputed_embeddings
    else:
        embedding_provider = get_embedding_provider(settings)
        embedding_texts = [build_embedding_text(chunk) for chunk in doc.chunks]
        embeddings = embedding_provider.embed_texts(embedding_texts)
    chunk_images = [extract_images_from_text(chunk) for chunk in doc.chunks]

    base_user_meta = doc.user_metadata or {}
    base_extra_meta = doc.extra_metadata or {}
    metadatas = [
        {
            **base_user_meta,
            **base_extra_meta,
            "doc_id": doc.doc_id,
            "title": doc.title,
            "kb_name": doc.kb_name,
            "chunk_index": i,
        }
        for i in range(len(doc.chunks))
    ]
    vector_store.add_chunks(
        kb_name=doc.kb_name,
        chunk_ids=doc.chunk_ids,
        texts=doc.chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    try:
        catalog.register_document_and_chunks(
            kb_name=doc.kb_name,
            doc_id=doc.doc_id,
            title=doc.title,
            content_hash=doc.content_hash,
            source_uri=doc.source_uri,
            chunk_ids=doc.chunk_ids,
            file_size=doc.file_size,
            chunk_contents=doc.chunks,
            chunk_images=chunk_images,
        )
    except Exception:
        try:
            vector_store.delete_chunks(doc.kb_name, doc.chunk_ids)
        except Exception as rollback_exc:
            logger.warning("元数据写入失败后清理刚入库向量失败 (%s): %s", doc.doc_id, rollback_exc)
        raise


def _commit_document_with_assets(
    settings: Settings,
    catalog: MetadataCatalogAdapter,
    vector_store: VectorStoreAdapter,
    doc: _DocumentIndexSpec,
    *,
    existing: dict[str, Any] | None,
    prepared_assets: PreparedDocumentAssets | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """协同写入文档元数据、向量切片与图片资产（支持异常补偿回滚与陈旧资产清理）。"""
    status = "indexed"
    old_assets: list[dict[str, Any]] = []
    if existing:
        old_assets = catalog.list_image_assets(doc.kb_name, doc_id=doc.doc_id)
        deleted_chunk_ids = catalog.delete_document(doc.kb_name, doc.doc_id)
        vector_store.delete_chunks(doc.kb_name, deleted_chunk_ids)
        status = "updated"

    written_assets: list[dict[str, Any]] = []
    try:
        _persist_indexed_document(settings, catalog, vector_store, doc)
        if prepared_assets is not None and prepared_assets.images:
            written_assets = persist_and_register_document_assets(
                settings,
                catalog,
                doc.kb_name,
                doc.doc_id,
                prepared_assets,
            )
    except Exception:
        try:
            rolled_chunk_ids = catalog.delete_document(doc.kb_name, doc.doc_id)
            vector_store.delete_chunks(doc.kb_name, rolled_chunk_ids or doc.chunk_ids)
        except Exception as rollback_exc:
            logger.warning(
                "索引资产失败回滚元数据/向量库异常 (%s): %s",
                doc.doc_id,
                rollback_exc,
            )
        cleanup_document_gateway_assets(
            settings,
            doc.kb_name,
            doc.doc_id,
            old_assets + written_assets,
        )
        raise

    if existing and old_assets:
        if not written_assets:
            cleanup_document_gateway_assets(settings, doc.kb_name, doc.doc_id, old_assets)
        else:
            cleanup_stale_gateway_assets(settings, old_assets, written_assets)

    return status, written_assets


def health_check() -> dict[str, str]:
    """检查 RAG MCP 服务基础运行状态。

    Returns:
        dict[str, str]: 包含服务状态、名称与版本的键值对字典。
    """
    return {
        "status": "healthy",
        "service": "mcp-server-rag",
        "version": "0.1.0",
    }


def rag_create_kb(
    name: str,
    description: str = "",
    owner: str = "default",
    visibility: str = "public",
) -> dict[str, Any]:
    """创建新的知识库命名空间与底层向量存储集合。

    Args:
        name: 知识库名称，全局唯一标识（仅限字母、数字、下划线、连字符）
        description: 知识库的详细描述说明信息
        owner: 知识库所属创建者标识，默认 ``"default"``
        visibility: 知识库可见性范围（``"public"`` 或 ``"private"``），默认 ``"public"``

    Returns:
        dict[str, Any]: 创建结果状态与说明
    """
    settings = get_settings()
    if settings.read_only:
        return {
            "success": False,
            "name": name,
            "message": READ_ONLY_BLOCK_MSG,
        }

    valid, err_msg = _validate_kb_name(name)
    if not valid:
        return {"success": False, "name": name, "message": err_msg}

    normalized_visibility = visibility.strip().lower()
    if normalized_visibility not in {"public", "private"}:
        return {
            "success": False,
            "name": name,
            "message": (
                f"非法的知识库可见性 visibility='{visibility}'，仅支持 'public' 或 'private'"
            ),
        }

    catalog, vector_store = _get_services()

    # 检查是否已存在
    existing = catalog.get_kb(name)
    if existing:
        return {
            "success": False,
            "name": name,
            "message": f"Knowledge base '{name}' already exists (知识库已存在)",
        }

    # 1. 获取实时向量维度（供强 Schema 向量引擎初始化物理集合）
    target_dim: int | None = settings.embedding_dimension
    if target_dim is None:
        try:
            target_dim = get_embedding_provider(settings).dimension
        except Exception as exc:
            logger.warning("初始化知识库 '%s' 时探测 Embedding 维度失败: %s", name, exc)
            if vector_store.requires_dimension:
                return {
                    "success": False,
                    "name": name,
                    "message": f"无法初始化向量存储集合：探测 Embedding 维度失败 ({exc})",
                }
            target_dim = None

    # 2. 在元数据目录中注册知识库扩展字段
    created = catalog.create_kb(
        name=name,
        description=description,
        embedding_model=settings.embedding_model,
        embedding_dimension=int(target_dim or 0),
        owner=owner,
        visibility=normalized_visibility,
    )
    if not created:
        return {
            "success": False,
            "name": name,
            "message": f"Failed to register knowledge base '{name}' in metadata catalog",
        }

    # 3. 在向量存储中初始化物理集合
    vector_store.create_kb(name, dimension=target_dim)

    return {
        "success": True,
        "name": name,
        "message": f"Knowledge base '{name}' created successfully",
    }


def rag_list_kbs() -> list[dict[str, Any]]:
    """枚举所有已创建的知识库清单及其文档数、切片统计与更新时间。

    Returns:
        list[dict[str, Any]]: 包含各知识库完整元数据的字典列表
    """
    catalog, _ = _get_services()
    return catalog.list_kbs()


def rag_clear_kb(name: str) -> dict[str, Any]:
    """清空指定知识库中的所有文档与切片数据，但保留知识库本身的基础配置与元数据。

    Args:
        name: 待清空的知识库名称

    Returns:
        dict[str, Any]: 清空结果与状态说明
    """
    settings = get_settings()
    if settings.read_only:
        return {
            "success": False,
            "name": name,
            "message": READ_ONLY_BLOCK_MSG,
        }

    catalog, vector_store = _get_services()

    # 检查知识库是否存在
    if not catalog.get_kb(name):
        return {
            "success": False,
            "name": name,
            "message": f"Knowledge base '{name}' not found (知识库不存在)",
        }

    # 1. 记录并清空 SQLite 元数据与本地物理资产
    kb_assets = catalog.list_image_assets(name)
    catalog.clear_kb(name)

    # 2. 清空向量库数据并清理物理图片文件及 S3 对象
    vector_store.clear_kb(name)
    cleanup_kb_gateway_assets(settings, name, kb_assets)

    return {
        "success": True,
        "name": name,
        "message": f"Knowledge base '{name}' cleared successfully",
    }


def rag_delete_kb(name: str) -> dict[str, Any]:
    """物理销毁指定知识库及其关联的所有文档元数据与向量索引集合。

    Args:
        name: 待删除的知识库名称

    Returns:
        dict[str, Any]: 删除结果与状态说明
    """
    settings = get_settings()
    if settings.read_only:
        return {
            "success": False,
            "name": name,
            "message": READ_ONLY_BLOCK_MSG,
        }

    catalog, vector_store = _get_services()

    # 检查知识库是否存在
    if not catalog.get_kb(name):
        return {
            "success": False,
            "name": name,
            "message": f"Knowledge base '{name}' not found (知识库不存在)",
        }

    # 1. 删除向量集合与登记图片资产清单
    kb_assets = catalog.list_image_assets(name)
    vector_store.delete_kb(name)

    # 2. 删除 SQLite 元数据并同步清理物理图片资产目录及 S3 对象
    catalog.delete_kb(name)
    cleanup_kb_gateway_assets(settings, name, kb_assets)

    return {
        "success": True,
        "name": name,
        "message": f"Knowledge base '{name}' deleted successfully",
    }


def rag_delete_document(kb_name: str, doc_id: str) -> dict[str, Any]:
    """从指定知识库中物理删除特定文档及其所有切片与向量索引。

    支持按文档 ID、标题或文件路径精确匹配。

    Args:
        kb_name: 目标知识库名称
        doc_id: 待删除的文档 ID、标题或文件路径

    Returns:
        dict[str, Any]: 删除操作结果状态字典
    """
    settings = get_settings()
    if settings.read_only:
        return {
            "success": False,
            "kb_name": kb_name,
            "doc_id": doc_id,
            "message": READ_ONLY_BLOCK_MSG,
        }

    clean_doc = doc_id.strip() if doc_id else ""
    if not clean_doc:
        return {
            "success": False,
            "kb_name": kb_name,
            "doc_id": doc_id,
            "message": "文档标识 (doc_id) 不能为空",
        }

    catalog, vector_store = _get_services()

    # 1. 校验知识库是否存在
    if not catalog.get_kb(kb_name):
        return {
            "success": False,
            "kb_name": kb_name,
            "doc_id": doc_id,
            "message": f"Knowledge base '{kb_name}' not found (知识库不存在)",
        }

    # 2. 匹配目标文档
    doc = catalog.find_document(kb_name, clean_doc)
    if not doc:
        return {
            "success": False,
            "kb_name": kb_name,
            "doc_id": doc_id,
            "message": f"Document '{doc_id}' not found in knowledge base '{kb_name}' (文档不存在)",
        }

    actual_doc_id = str(doc["id"])

    # 3. 在删除元数据前提取登记的本地资产并原子删除文档与切片记录
    doc_assets = catalog.list_image_assets(kb_name, doc_id=actual_doc_id)
    deleted_chunk_ids = catalog.delete_document(kb_name, actual_doc_id)

    # 4. 在向量数据库中删除对应向量切片并清理本地物理图片与 S3 对象资产
    if deleted_chunk_ids:
        vector_store.delete_chunks(kb_name, deleted_chunk_ids)
    cleanup_document_gateway_assets(settings, kb_name, actual_doc_id, doc_assets)

    return {
        "success": True,
        "kb_name": kb_name,
        "doc_id": actual_doc_id,
        "deleted_chunks_count": len(deleted_chunk_ids),
        "message": (
            f"Document '{actual_doc_id}' and {len(deleted_chunk_ids)} chunk(s) "
            f"successfully deleted from knowledge base '{kb_name}'"
        ),
    }


def rag_doctor() -> dict[str, Any]:
    """全维自检诊断探针，检查存储引擎连通性、本地模型配置、磁盘占用与系统环境。

    Returns:
        dict[str, Any]: 结构化诊断报告
    """
    settings = get_settings()
    catalog, vector_store = _get_services()

    # 检查 SQLite 与底层向量存储连通性
    sqlite_ok = catalog.ping()
    vector_store_ok = vector_store.ping()

    # 计算磁盘占用
    disk_usage_bytes = 0
    if settings.data_dir.exists():
        for f in settings.data_dir.glob("**/*"):
            if f.is_file():
                disk_usage_bytes += f.stat().st_size

    # 获取系统磁盘可用空间
    total, used, free = shutil.disk_usage(settings.data_dir.parent)

    kb_count = catalog.count_kbs()
    doc_count = catalog.count_documents()

    embedding_dimension: int | None = None
    embedding_status = "ok"
    embedding_error: str | None = None
    try:
        provider = get_embedding_provider(settings)
        embedding_dimension = provider.dimension
    except Exception as exc:
        logger.warning("rag_doctor 探测 Embedding 提供方失败: %s", exc)
        embedding_status = "error"
        embedding_error = str(exc)

    overall_status = (
        "healthy" if (sqlite_ok and vector_store_ok and embedding_status == "ok") else "degraded"
    )

    report: dict[str, Any] = {
        "status": overall_status,
        "data_dir": str(settings.data_dir),
        "data_dir_size_bytes": disk_usage_bytes,
        "disk_free_bytes": free,
        "vector_store": settings.vector_store,
        "vector_store_status": "ok" if vector_store_ok else "error",
        "metadata_store": settings.metadata_store,
        "metadata_catalog_status": "ok" if sqlite_ok else "error",
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": embedding_dimension,
        "embedding_status": embedding_status,
        "storage_type": settings.storage_type,
        "archive_remote_images": settings.archive_remote_images,
        "read_only": settings.read_only,
        "knowledge_bases_count": kb_count,
        "documents_count": doc_count,
    }
    if embedding_error:
        report["embedding_error"] = embedding_error
    return report


def rag_index_text(
    kb_name: str,
    text: str,
    title: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """对单段纯文本内容进行切片、计算高维向量嵌入并持久化索引到指定知识库中。

    支持基于 SHA-256 内容指纹校验的幂等更新：
    - 若相同 title 的文档内容未变动，秒级跳过；
    - 若内容发生修改，原子删除旧切片并插入新切片。

    Args:
        kb_name: 目标知识库名称
        text: 待索引的正文纯文本内容
        title: 文档标题或唯一标识（在知识库中作为 doc_id）
        metadata: 可选的附加结构化元数据（标签、分类、作者等）

    Returns:
        dict[str, Any]: 索引执行结果状态字典
    """
    settings = get_settings()
    if settings.read_only:
        return {
            "success": False,
            "message": READ_ONLY_BLOCK_MSG,
        }

    if not text or not text.strip():
        return {"success": False, "message": "正文内容不能为空"}
    if not title or not title.strip():
        return {"success": False, "message": "文档标题 (title) 不能为空"}

    catalog, vector_store = _get_services()

    # 1. 校验知识库是否存在
    err = _validate_kb_for_write(catalog, kb_name, settings)
    if err:
        return err

    doc_id = title.strip()
    raw_text = text.strip()
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    # 2. 幂等指纹比对
    existing = catalog.get_document_by_title(kb_name, doc_id)
    if existing and existing["content_hash"] == content_hash:
        return {
            "success": True,
            "status": "skipped",
            "doc_id": doc_id,
            "chunk_count": existing["chunk_count"],
            "content_hash": content_hash,
            "message": (
                f"Document '{doc_id}' content unchanged, skipped indexing (内容未变动，已跳过)"
            ),
        }

    # 3. 预处理图片外链安全归档、切分与向量计算（失败时不误删存量版本）
    prepared_assets = prepare_document_assets(
        settings,
        kb_name,
        doc_id,
        raw_text,
    )
    chunks = recursive_split_text(
        prepared_assets.processed_text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        chunks = [prepared_assets.processed_text]

    embedding_provider = get_embedding_provider(settings)
    embedding_texts = [build_embedding_text(chunk) for chunk in chunks]
    precomputed_embeddings = embedding_provider.embed_texts(embedding_texts)

    # 4. 协同写入文档元数据、向量切片与图片资产（含回滚与陈旧清理）
    chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    status, _ = _commit_document_with_assets(
        settings,
        catalog,
        vector_store,
        _DocumentIndexSpec(
            kb_name=kb_name,
            doc_id=doc_id,
            title=title,
            content_hash=content_hash,
            source_uri="",
            file_size=len(raw_text.encode("utf-8")),
            chunks=chunks,
            chunk_ids=chunk_ids,
            user_metadata=metadata,
            precomputed_embeddings=precomputed_embeddings,
        ),
        existing=existing,
        prepared_assets=prepared_assets,
    )

    return {
        "success": True,
        "status": status,
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "content_hash": content_hash,
        "message": f"Document '{title}' successfully {status}",
    }


def rag_index_file(
    kb_name: str,
    file_path: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """摄取本地单个文件，自动识别文件格式并进行结构感知切片、向量嵌入与持久化索引。

    支持 SHA-256 内容指纹校验与幂等更新（若文件未修改则秒级跳过）。

    Args:
        kb_name: 目标知识库名称
        file_path: 本地文件绝对或相对路径
        metadata: 可选的附加结构化元数据（标签、分类等）

    Returns:
        dict[str, Any]: 索引结果状态字典
    """
    settings = get_settings()
    if settings.read_only:
        return {
            "success": False,
            "message": READ_ONLY_BLOCK_MSG,
        }

    path = Path(file_path).resolve()
    if not path.exists():
        return {"success": False, "message": f"文件不存在: {file_path}"}
    if not path.is_file():
        return {"success": False, "message": f"目标路径不是普通文件: {file_path}"}

    is_office = is_office_document(path)
    if is_office:
        ensure_office_dependency(path)

    catalog, vector_store = _get_services()

    # 1. 校验知识库是否存在
    err = _validate_kb_for_write(catalog, kb_name, settings)
    if err:
        return err

    doc_id = str(path)
    title = path.name
    if is_office:
        content_hash = _compute_file_sha256(path)
    else:
        try:
            raw_content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                raw_content = path.read_text(encoding="gbk")
            except Exception:
                raw_content = path.read_text(encoding="latin-1", errors="replace")

        if not raw_content or not raw_content.strip():
            return {"success": False, "message": "文件正文为空"}
        content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()

    # 2. 幂等指纹比对
    existing = catalog.get_document_by_id(kb_name, doc_id)
    if existing and existing["content_hash"] == content_hash:
        return {
            "success": True,
            "status": "skipped",
            "doc_id": doc_id,
            "file_path": str(path),
            "chunk_count": existing["chunk_count"],
            "content_hash": content_hash,
            "message": f"File '{title}' content unchanged, skipped indexing",
        }

    prepared_assets: PreparedDocumentAssets | None = None
    ext = path.suffix.lower()
    if is_office:
        try:
            parsed_office = parse_office_document(path, kb_name=kb_name, doc_id=doc_id)
        except ImportError:
            raise
        except Exception as exc:
            logger.warning("解析二进制文档 '%s' 失败: %s", path.name, exc)
            return {"success": False, "message": f"解析文档 '{title}' 失败: {exc}"}
        raw_content = parsed_office.markdown_text
        if not raw_content or not raw_content.strip():
            return {"success": False, "message": "文档未提取到有效文本或图片内容"}
        file_size = path.stat().st_size
        prepared_assets = prepare_document_assets(
            settings,
            kb_name,
            doc_id,
            raw_content,
            pre_extracted_images=parsed_office.images,
        )
        chunks = dispatch_and_split(
            file_path=path,
            content=prepared_assets.processed_text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            chunks = [prepared_assets.processed_text]
    elif ext in {".html", ".htm"}:
        file_size = len(raw_content.encode("utf-8"))
        html_title, cleaned_html = extract_html_content(raw_content)
        normalized_html = (
            f"# {html_title}\n\n{cleaned_html}"
            if html_title and html_title not in cleaned_html
            else (cleaned_html or html_title.strip())
        )
        prepared_assets = prepare_document_assets(
            settings,
            kb_name,
            doc_id,
            normalized_html,
            base_dir=path.parent,
        )
        chunks = (
            recursive_split_text(
                prepared_assets.processed_text,
                settings.chunk_size,
                settings.chunk_overlap,
            )
            if prepared_assets.processed_text
            else []
        )
        if not chunks:
            chunks = [prepared_assets.processed_text or raw_content]
    elif is_text_markup_document(path):
        file_size = len(raw_content.encode("utf-8"))
        prepared_assets = prepare_document_assets(
            settings,
            kb_name,
            doc_id,
            raw_content,
            base_dir=path.parent,
        )
        chunks = dispatch_and_split(
            file_path=path,
            content=prepared_assets.processed_text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            chunks = [prepared_assets.processed_text]
    else:
        file_size = len(raw_content.encode("utf-8"))
        chunks = dispatch_and_split(
            file_path=path,
            content=raw_content,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            chunks = [raw_content]

    # 3. 提前完成向量计算，确保 S3 校验或嵌入失败时不误删存量版本
    embedding_provider = get_embedding_provider(settings)
    embedding_texts = [build_embedding_text(chunk) for chunk in chunks]
    precomputed_embeddings = embedding_provider.embed_texts(embedding_texts)

    # 4. 协同写入文档元数据、向量切片与图片资产（含失败补偿回滚及陈旧资产清理）
    path_hash = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]
    chunk_ids = [f"{path.stem}_{path_hash}_{i}" for i in range(len(chunks))]
    status, _ = _commit_document_with_assets(
        settings,
        catalog,
        vector_store,
        _DocumentIndexSpec(
            kb_name=kb_name,
            doc_id=doc_id,
            title=title,
            content_hash=content_hash,
            source_uri=str(path),
            file_size=file_size,
            chunks=chunks,
            chunk_ids=chunk_ids,
            user_metadata=metadata,
            extra_metadata={
                "file_path": str(path),
                "file_name": path.name,
                "file_extension": path.suffix.lower(),
            },
            precomputed_embeddings=precomputed_embeddings,
        ),
        existing=existing,
        prepared_assets=prepared_assets,
    )

    return {
        "success": True,
        "status": status,
        "doc_id": doc_id,
        "file_path": str(path),
        "chunk_count": len(chunks),
        "content_hash": content_hash,
        "message": f"File '{title}' successfully {status}",
    }


IGNORED_DIR_NAMES: set[str] = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".rag_data",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
}

IGNORED_FILE_EXTENSIONS: set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".svg",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".pyc",
    ".pyo",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".xlsx",
    ".pptx",
}


def rag_index_directory(
    kb_name: str,
    dir_path: str,
    glob_pattern: str = "**/*",
    max_files: int = 200,
    recursive: bool = True,
) -> dict[str, Any]:
    """批量扫描并摄取本地目录下的所有文件，支持递归遍历与 max_files 超时防御截断。

    Args:
        kb_name: 目标知识库名称
        dir_path: 目标目录绝对或相对路径
        glob_pattern: 匹配文件模式（默认为 "**/*"）
        max_files: 目录文件摄取上限，防止大目录导致 MCP 响应超时（默认 200）
        recursive: 是否递归扫描子目录（默认 True）

    Returns:
        dict[str, Any]: 目录批量摄取统计报告字典
    """
    settings = get_settings()
    if settings.read_only:
        return {
            "success": False,
            "message": READ_ONLY_BLOCK_MSG,
        }

    path = Path(dir_path).resolve()
    if not path.exists() or not path.is_dir():
        return {
            "success": False,
            "message": f"目录不存在或不是有效目录: {dir_path}",
        }

    catalog, _ = _get_services()
    err = _validate_kb_for_write(catalog, kb_name, settings)
    if err:
        return err

    # 1. 扫描候选文件
    candidate_files: list[Path] = []
    scan_iter = (
        path.glob(glob_pattern)
        if "**" in glob_pattern or not recursive
        else path.rglob(glob_pattern)
    )
    for p in scan_iter:
        if not p.is_file():
            continue
        # 排除隐藏目录与特殊忽略目录
        parts = p.relative_to(path).parts
        if any(part in IGNORED_DIR_NAMES or part.startswith(".") for part in parts[:-1]):
            continue
        # 排除以 . 开头的隐藏文件
        if p.name.startswith("."):
            continue
        # 排除已知二进制或非文本格式
        if p.suffix.lower() in IGNORED_FILE_EXTENSIONS:
            continue
        candidate_files.append(p)

    # 排序以保证多平台遍历结果的一致性
    candidate_files.sort()

    total_found = len(candidate_files)
    truncated = False
    files_to_process = candidate_files
    if total_found > max_files:
        files_to_process = candidate_files[:max_files]
        truncated = True

    # 2. 依次摄取各文件
    file_reports: list[dict[str, Any]] = []
    indexed_count = 0
    updated_count = 0
    skipped_count = 0
    failed_count = 0

    for file_p in files_to_process:
        try:
            res = rag_index_file(kb_name=kb_name, file_path=str(file_p))
        except ImportError as exc:
            res = {"success": False, "message": str(exc)}
        status = res.get("status", "failed") if res.get("success") else "failed"
        if status == "indexed":
            indexed_count += 1
        elif status == "updated":
            updated_count += 1
        elif status == "skipped":
            skipped_count += 1
        else:
            failed_count += 1

        file_reports.append(
            {
                "file_path": str(file_p),
                "status": status,
                "chunk_count": res.get("chunk_count", 0),
                "message": res.get("message", ""),
            }
        )

    msg = (
        f"目录扫描发现 {total_found} 个候选文件，"
        f"本次处理 {len(files_to_process)} 个文件 "
        f"(新增: {indexed_count}, 更新: {updated_count}, "
        f"跳过: {skipped_count}, 失败: {failed_count})。"
    )
    if truncated:
        msg += f" [警告: 超过 max_files={max_files} 限制，已执行截断]"

    return {
        "success": True,
        "dir_path": str(path),
        "total_found": total_found,
        "total_indexed": len(files_to_process),
        "indexed_count": indexed_count,
        "updated_count": updated_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "truncated": truncated,
        "files": file_reports,
        "message": msg,
    }


def rag_index_url(
    kb_name: str,
    url: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """通过 HTTP 请求抓取在线网页或在线文档，清洗 HTML 视觉噪点并进行向量索引。

    支持 SHA-256 内容指纹校验与幂等跳过，并具备网络超时与异常防护。

    Args:
        kb_name: 目标知识库名称
        url: 目标网页或在线文档的完整 URL（必须以 http:// 或 https:// 开头）
        metadata: 可选的附加结构化元数据字典

    Returns:
        dict[str, Any]: 索引结果状态字典
    """
    settings = get_settings()
    if settings.read_only:
        return {
            "success": False,
            "message": READ_ONLY_BLOCK_MSG,
        }

    clean_url = url.strip()
    if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
        return {
            "success": False,
            "message": f"URL 协议无效，必须以 http:// 或 https:// 开头: {url}",
        }

    settings = get_settings()
    catalog, vector_store = _get_services()
    err = _validate_kb_for_write(catalog, kb_name, settings)
    if err:
        return err

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Antigravity-RAG/0.1.0"
        )
    }
    try:
        resp = httpx.get(clean_url, headers=headers, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
    except httpx.TimeoutException as exc:
        return {
            "success": False,
            "message": f"请求 URL 网络超时: {exc}",
        }
    except httpx.HTTPStatusError as exc:
        return {
            "success": False,
            "message": f"HTTP 请求失败 ({exc.response.status_code}): {exc}",
        }
    except Exception as exc:
        return {
            "success": False,
            "message": f"获取 URL 内容失败: {exc}",
        }

    content_type = resp.headers.get("content-type", "").lower()
    raw_resp_text = resp.text
    if "html" in content_type or "<html" in raw_resp_text.lower()[:200]:
        extracted_title, text_content = extract_html_content(raw_resp_text)
        fallback_title = clean_url.rstrip("/").split("/")[-1] or clean_url
        title = extracted_title if extracted_title else fallback_title
    else:
        text_content = raw_resp_text.strip()
        title = clean_url.rstrip("/").split("/")[-1] or clean_url

    if not text_content:
        return {
            "success": False,
            "message": f"URL 页面提取正文为空: {clean_url}",
        }

    content_hash = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
    doc_id = clean_url

    # 幂等比对
    existing = catalog.get_document_by_id(kb_name, doc_id)
    if existing and existing["content_hash"] == content_hash:
        return {
            "success": True,
            "status": "skipped",
            "doc_id": doc_id,
            "url": clean_url,
            "title": title,
            "chunk_count": existing["chunk_count"],
            "content_hash": content_hash,
            "message": f"URL '{clean_url}' content unchanged, skipped indexing",
        }

    # 外链安全归档预处理与切片分发
    prepared_assets = prepare_document_assets(
        settings,
        kb_name,
        doc_id,
        text_content,
        source_url=clean_url,
    )
    if "markdown" in content_type:
        chunks = split_markdown(
            prepared_assets.processed_text,
            settings.chunk_size,
            settings.chunk_overlap,
        )
    else:
        chunks = recursive_split_text(
            prepared_assets.processed_text,
            settings.chunk_size,
            settings.chunk_overlap,
        )
    if not chunks:
        chunks = [prepared_assets.processed_text]

    embedding_provider = get_embedding_provider(settings)
    embedding_texts = [build_embedding_text(chunk) for chunk in chunks]
    precomputed_embeddings = embedding_provider.embed_texts(embedding_texts)

    url_hash = hashlib.md5(clean_url.encode("utf-8")).hexdigest()[:8]
    chunk_ids = [f"url_{url_hash}_{i}" for i in range(len(chunks))]
    status, _ = _commit_document_with_assets(
        settings,
        catalog,
        vector_store,
        _DocumentIndexSpec(
            kb_name=kb_name,
            doc_id=doc_id,
            title=title,
            content_hash=content_hash,
            source_uri=clean_url,
            file_size=len(text_content.encode("utf-8")),
            chunks=chunks,
            chunk_ids=chunk_ids,
            user_metadata=metadata,
            extra_metadata={
                "url": clean_url,
                "source_uri": clean_url,
                "content_type": content_type,
            },
            precomputed_embeddings=precomputed_embeddings,
        ),
        existing=existing,
        prepared_assets=prepared_assets,
    )

    return {
        "success": True,
        "status": status,
        "doc_id": doc_id,
        "url": clean_url,
        "title": title,
        "chunk_count": len(chunks),
        "content_hash": content_hash,
        "message": f"URL '{clean_url}' successfully {status}",
    }


def _build_chroma_where(metadata_filter: dict[str, Any] | None) -> dict[str, Any] | None:
    """将通用的元数据字典转换为 ChromaDB 合法的 where 过滤语句。"""
    if not metadata_filter:
        return None
    # 若已经包含 Chroma 原生逻辑操作符（如 $and, $or），直接透传
    if any(k.startswith("$") for k in metadata_filter):
        return metadata_filter

    if len(metadata_filter) == 1:
        k, v = next(iter(metadata_filter.items()))
        if isinstance(v, dict):
            return {k: v}
        return {k: {"$eq": v}}

    conditions: list[dict[str, Any]] = []
    for k, v in metadata_filter.items():
        if isinstance(v, dict):
            conditions.append({k: v})
        else:
            conditions.append({k: {"$eq": v}})
    return {"$and": conditions}


def _build_chroma_where_document(keyword: str | None) -> dict[str, Any] | None:
    """将精准关键词转换为 ChromaDB 合法的 where_document 过滤语句。"""
    if not keyword or not keyword.strip():
        return None
    kw = keyword.strip()
    cases = list(dict.fromkeys([kw, kw.lower(), kw.upper()]))
    if len(cases) == 1:
        return {"$contains": cases[0]}
    return {"$or": [{"$contains": c} for c in cases]}


def _resolve_target_kbs(
    catalog: MetadataCatalogAdapter,
    kb_name: str | list[str],
) -> tuple[list[str], bool]:
    """解析目标知识库清单并标识是否为跨库联合检索模式。

    Args:
        catalog: SQLite 元数据目录实例
        kb_name: 单知识库名称、知识库列表或通配符 "*"

    Returns:
        tuple[list[str], bool]: (有效知识库名称列表, 是否为跨库模式)
    """
    existing_kbs = {kb["name"] for kb in catalog.list_kbs()}
    if not existing_kbs:
        return [], False

    if isinstance(kb_name, str):
        trimmed = kb_name.strip()
        if trimmed == "*":
            return [kb["name"] for kb in catalog.list_kbs()], True
        if trimmed in existing_kbs:
            return [trimmed], False
        return [], False

    if isinstance(kb_name, list):
        if "*" in kb_name:
            return [kb["name"] for kb in catalog.list_kbs()], True
        target_list: list[str] = []
        for item in kb_name:
            if isinstance(item, str) and item.strip() in existing_kbs:
                target_list.append(item.strip())
        return target_list, True

    return [], False


def _hydrate_hits_images(
    catalog: MetadataCatalogAdapter,
    hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """从元数据目录批量回填命中切片的结构化 images URL 列表（无图片恒为空列表 []）。"""
    if not hits:
        return hits

    ids_by_kb: dict[str, list[str]] = {}
    for hit in hits:
        kb = str(hit.get("kb_name") or "")
        cid = str(hit.get("chunk_id") or "")
        if kb and cid:
            ids_by_kb.setdefault(kb, []).append(cid)

    chunk_map: dict[tuple[str, str], dict[str, Any]] = {}
    for kb, cids in ids_by_kb.items():
        for row in catalog.get_chunks_by_ids(kb, cids):
            chunk_map[(kb, str(row["id"]))] = row

    for hit in hits:
        kb = str(hit.get("kb_name") or "")
        cid = str(hit.get("chunk_id") or "")
        item_meta = hit.setdefault("metadata", {})
        chunk_row = chunk_map.get((kb, cid))
        if chunk_row is not None:
            item_meta["images"] = list(chunk_row.get("images") or [])
        else:
            item_meta["images"] = extract_images_from_text(str(hit.get("text") or ""))
    return hits


def _normalize_expand_context(expand_context: int) -> int:
    """将邻近上下文扩展步数钳位至安全区间 [0, MAX_EXPAND_CONTEXT]。"""
    return max(0, min(int(expand_context), MAX_EXPAND_CONTEXT))


@dataclass
class _ExpandedCluster:
    """同文档邻近窗口扩展与重叠合并后的切片聚合簇。"""

    chunk_ids: set[str]
    seed_hit: dict[str, Any]
    score: float
    fallbacks: dict[str, str] = field(default_factory=dict)


def _collect_chain(
    by_id: dict[str, dict[str, Any]],
    start_node: dict[str, Any],
    pointer_key: str,
    window: int,
    visited: set[str],
) -> list[dict[str, Any]]:
    """沿单侧链表指针（prev_chunk_id 或 next_chunk_id）采集带防环保护的邻接节点。"""
    chain: list[dict[str, Any]] = []
    curr = start_node
    for _ in range(window):
        target_id = str(curr.get(pointer_key) or "")
        if not target_id or target_id in visited or target_id not in by_id:
            break
        visited.add(target_id)
        node = by_id[target_id]
        chain.append(node)
        curr = node
    return chain


def _walk_linked_window(
    by_id: dict[str, dict[str, Any]],
    seed_id: str,
    window: int,
) -> list[dict[str, Any]]:
    """沿双向链表指针 prev_chunk_id/next_chunk_id 收集带防环保护的邻接切片窗口。"""
    seed_node = by_id.get(seed_id)
    if seed_node is None:
        return []

    visited: set[str] = {seed_id}
    prev_chain = _collect_chain(by_id, seed_node, "prev_chunk_id", window, visited)
    prev_chain.reverse()
    next_chain = _collect_chain(by_id, seed_node, "next_chunk_id", window, visited)
    return [*prev_chain, seed_node, *next_chain]


def _materialize_cluster(
    cluster: _ExpandedCluster,
    by_id: dict[str, dict[str, Any]],
    pos_by_id: dict[str, int],
) -> dict[str, Any]:
    """按文档原始阅读顺序缝合簇内切片文本，并去重聚合关联图片清单。"""
    ordered_ids = sorted(cluster.chunk_ids, key=lambda cid: pos_by_id.get(cid, 0))

    pieces: list[str] = []
    merged_images: list[str] = []
    seen_images: set[str] = set()

    for cid in ordered_ids:
        row = by_id.get(cid)
        if row is None:
            continue
        piece_text = str(row.get("content") or "").strip()
        if not piece_text:
            piece_text = cluster.fallbacks.get(cid, "").strip()
        if piece_text:
            pieces.append(piece_text)

        row_images = row.get("images")
        raw_imgs: list[str] = (
            [str(u) for u in row_images if u]
            if isinstance(row_images, list)
            else extract_images_from_text(piece_text)
        )
        for img_url in raw_imgs:
            if img_url not in seen_images:
                seen_images.add(img_url)
                merged_images.append(img_url)

    seed_hit = cluster.seed_hit
    stitched_text = "\n\n".join(pieces) if pieces else str(seed_hit.get("text") or "")
    expanded_meta = dict(seed_hit.get("metadata") or {})
    expanded_meta["images"] = merged_images

    return {
        **seed_hit,
        "text": stitched_text,
        "score": cluster.score,
        "metadata": expanded_meta,
    }


def _expand_and_deduplicate_hits(
    catalog: MetadataCatalogAdapter,
    hits: list[dict[str, Any]],
    window: int,
) -> list[dict[str, Any]]:
    """展开各命中切片的邻近上下文窗口，并融合同文档重叠区间以消除重复文本。"""
    if window <= 0 or not hits:
        return hits

    # 1. 预先查询缺少 doc_id 的命中切片，尽量还原文档归属
    missing_doc_ids_by_kb: dict[str, list[str]] = {}
    for hit in hits:
        kb = str(hit.get("kb_name") or "")
        cid = str(hit.get("chunk_id") or "")
        meta = hit.get("metadata") or {}
        if kb and cid and not meta.get("doc_id"):
            missing_doc_ids_by_kb.setdefault(kb, []).append(cid)

    seed_doc_map: dict[tuple[str, str], str] = {}
    for kb, cids in missing_doc_ids_by_kb.items():
        for row in catalog.get_chunks_by_ids(kb, cids):
            seed_doc_map[(kb, str(row["id"]))] = str(row.get("doc_id") or "")

    # 2. 按 (kb_name, doc_id) 缓存有序文档切片索引
    doc_cache: dict[tuple[str, str], tuple[dict[str, dict[str, Any]], dict[str, int]]] = {}
    clusters_by_doc: dict[tuple[str, str], list[_ExpandedCluster]] = {}
    passthrough_hits: list[dict[str, Any]] = []

    for hit in hits:
        kb = str(hit.get("kb_name") or "")
        seed_id = str(hit.get("chunk_id") or "")
        hit_meta = hit.get("metadata") or {}
        doc_id = str(hit_meta.get("doc_id") or seed_doc_map.get((kb, seed_id), ""))

        if not kb or not doc_id or not seed_id:
            passthrough_hits.append(hit)
            continue

        doc_key = (kb, doc_id)
        if doc_key not in doc_cache:
            doc_rows = catalog.get_chunks_by_doc(kb, doc_id)
            by_id = {str(r["id"]): r for r in doc_rows}
            pos_by_id = {str(r["id"]): idx for idx, r in enumerate(doc_rows)}
            doc_cache[doc_key] = (by_id, pos_by_id)

        by_id, _ = doc_cache[doc_key]
        window_nodes = _walk_linked_window(by_id, seed_id, window)
        if not window_nodes:
            passthrough_hits.append(hit)
            continue

        window_ids = {str(node["id"]) for node in window_nodes}
        hit_score = float(hit.get("score") or 0.0)
        hit_text = str(hit.get("text") or "")

        existing = clusters_by_doc.setdefault(doc_key, [])
        overlapping = [idx for idx, cl in enumerate(existing) if cl.chunk_ids & window_ids]

        if not overlapping:
            existing.append(
                _ExpandedCluster(
                    chunk_ids=set(window_ids),
                    seed_hit=hit,
                    score=hit_score,
                    fallbacks={seed_id: hit_text},
                )
            )
            continue

        # 合并所有与当前窗口重叠的簇，保留最高分种子命中及全部文本兜底
        merged_cluster = _ExpandedCluster(
            chunk_ids=set(window_ids),
            seed_hit=hit,
            score=hit_score,
            fallbacks={seed_id: hit_text},
        )
        remaining: list[_ExpandedCluster] = []
        overlap_set = set(overlapping)
        for idx, cl in enumerate(existing):
            if idx not in overlap_set:
                remaining.append(cl)
                continue
            merged_cluster.chunk_ids.update(cl.chunk_ids)
            merged_cluster.fallbacks.update(cl.fallbacks)
            if cl.score >= merged_cluster.score:
                merged_cluster.score = cl.score
                merged_cluster.seed_hit = cl.seed_hit

        remaining.append(merged_cluster)
        clusters_by_doc[doc_key] = remaining

    # 3. 实体化合并后的文档窗口并维持全局降序排列
    expanded_results: list[dict[str, Any]] = list(passthrough_hits)
    for doc_key, clusters in clusters_by_doc.items():
        by_id, pos_by_id = doc_cache[doc_key]
        for cluster in clusters:
            expanded_results.append(_materialize_cluster(cluster, by_id, pos_by_id))

    expanded_results.sort(
        key=lambda x: (
            float(x.get("score") or 0.0),
            float((x.get("metadata") or {}).get("rrf_score") or 0.0),
        ),
        reverse=True,
    )
    return expanded_results


def rag_search(
    kb_name: str | list[str],
    query: str,
    top_k: int = 5,
    threshold: float = 0.0,
    keyword: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    expand_context: int = 0,
) -> list[dict[str, Any]]:
    """在指定知识库或跨多个知识库中执行向量语义检索与原生 FTS 倒排混合检索（RRF 融合）。

    Args:
        kb_name: 目标知识库名称，支持单库名称、知识库列表（如 ["docs", "code"]）或 "*" 跨全库召回
        query: 检索查询文本
        top_k: 返回相似切片的最大数量（默认为 5）
        threshold: 相似度截断阈值，过滤掉低于此分数的切片（范围 0.0 ~ 1.0）
        keyword: 可选的精准关键词，切片正文必须包含此词（大小写无关）
        metadata_filter: 可选的元数据字典过滤条件（如 {"category": "tech"}）
        expand_context: 邻近上下文窗口扩展步数（默认 0 不展开，允许安全范围 0 ~ 5）

    Returns:
        list[dict[str, Any]]: 排序后的相似切片列表，
            每个切片包含 chunk_id, text, score, metadata（含结构化 images 列表）, kb_name
    """
    if not query or not query.strip():
        return []

    settings = get_settings()
    catalog, vector_store = _get_services()

    # 1. 解析目标知识库清单
    target_kbs, _ = _resolve_target_kbs(catalog, kb_name)
    if not target_kbs:
        return []

    # 2. 构造底层向量数据库过滤条件
    where = _build_chroma_where(metadata_filter)
    where_document = _build_chroma_where_document(keyword)

    # 3. 计算查询向量
    embedding_provider = get_embedding_provider(settings)
    query_vector = embedding_provider.embed_query(query.strip())

    # 4. 执行向量检索 + 原生数据库 FTS 倒排双路召回与 RRF 融合重排
    fused_hits = execute_hybrid_search(
        catalog=catalog,
        vector_store=vector_store,
        target_kbs=target_kbs,
        query=query,
        query_vector=query_vector,
        options=HybridSearchOptions(
            top_k=top_k,
            threshold=threshold,
            keyword=keyword,
            metadata_filter=metadata_filter,
            where=where,
            where_document=where_document,
        ),
    )

    # 5. 补充结构化图片元数据并按需执行双向链表上下文扩展
    top_hits = _hydrate_hits_images(catalog, fused_hits)
    window = _normalize_expand_context(expand_context)
    if window > 0:
        return _expand_and_deduplicate_hits(catalog, top_hits, window)
    return top_hits


def rag_query_with_context(
    kb_name: str | list[str],
    query: str,
    top_k: int = 5,
    threshold: float = 0.0,
    keyword: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
    expand_context: int = 0,
) -> str:
    """检索知识库相关上下文并组装为带来源引用的 Markdown 增强提示词。

    Args:
        kb_name: 目标知识库名称（支持单库名、知识库名称列表或通配符 "*" 跨库检索）
        query: 检索查询问题或主题
        top_k: 返回参考切片的最大数量（默认为 5）
        threshold: 相似度截断阈值，过滤掉低于此分数的切片
        keyword: 可选的精准关键词
        metadata_filter: 可选的元数据字典过滤条件
        expand_context: 邻近上下文窗口扩展步数（默认 0 不展开，允许安全范围 0 ~ 5）

    Returns:
        str: 组装后的 Markdown 提示词字符串；若无匹配内容则返回未检索到上下文提示
    """
    chunks = rag_search(
        kb_name=kb_name,
        query=query,
        top_k=top_k,
        threshold=threshold,
        keyword=keyword,
        metadata_filter=metadata_filter,
        expand_context=expand_context,
    )

    if not chunks:
        return "未检索到与查询相关的上下文内容。"

    is_cross = kb_name == "*" or isinstance(kb_name, list)

    context_sections: list[str] = ["根据以下检索到的相关知识库参考资料回答问题：\n"]
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        title = meta.get("title") or meta.get("doc_id") or chunk.get("chunk_id", f"文档_{i}")
        score = chunk.get("score", 0.0)
        source_kb = chunk.get("kb_name") or meta.get("kb_name", "")
        raw_text = chunk.get("text", "").strip()
        quoted_text = (
            "\n".join(f"> {line}" for line in raw_text.splitlines()) if raw_text else "> (空内容)"
        )
        if is_cross and source_kb:
            section = (
                f"### 📚 参考片段 [{i}] (来源: {title} [知识库: {source_kb}], "
                f"相似度: {score:.4f})\n{quoted_text}\n"
            )
        else:
            section = f"### 📚 参考片段 [{i}] (来源: {title}, 相似度: {score:.4f})\n{quoted_text}\n"
        context_sections.append(section)

    return "\n".join(context_sections).strip()


class _ImageAssetResourceTemplate(ResourceTemplate):
    """按资产 URI 模板动态加载图片二进制字节与对应 MIME 类型的 MCP Resource 模板。"""

    async def create_resource(
        self,
        uri: str,
        params: dict[str, Any],
        context: Any,
    ) -> FunctionResource:
        """依据模板提取的单级路径参数读取图片载荷并封装精确 MIME 响应。"""
        _ = context
        kb_name = str(params.get("kb_name") or "")
        doc_id = str(params.get("doc_id") or "")
        filename = str(params.get("filename") or "")

        try:
            payload_raw, mime_raw = await anyio.to_thread.run_sync(
                self.fn,
                kb_name,
                doc_id,
                filename,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ResourceNotFoundError(str(exc)) from exc
        except ImportError as exc:
            raise ResourceError(str(exc)) from exc

        payload = bytes(payload_raw)
        mime_type = str(mime_raw)

        def _read_payload() -> bytes:
            return payload

        return FunctionResource(
            uri=uri,
            name=filename,
            title=self.title,
            description=self.description,
            mime_type=mime_type,
            icons=self.icons,
            annotations=self.annotations,
            meta=self.meta,
            fn=_read_payload,
        )


class _RagMCPServer(MCPServer[Any]):
    """支持从 Metadata Catalog 动态列举图片资产资源的 FastMCP 扩展实例。"""

    def __init__(
        self,
        *args: Any,
        dynamic_resources_loader: Callable[[], list[MCPResource]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._dynamic_resources_loader = dynamic_resources_loader

    async def list_resources(self) -> list[MCPResource]:
        """合并静态注册资源与 Metadata Catalog 中登记的实时图片资产清单。"""
        static_resources = await super().list_resources()
        if self._dynamic_resources_loader is None:
            return static_resources
        dynamic_resources = await anyio.to_thread.run_sync(self._dynamic_resources_loader)
        existing_uris = {item.uri for item in static_resources}
        return static_resources + [
            item for item in dynamic_resources if item.uri not in existing_uris
        ]


def create_server(settings: Settings | None = None) -> FastMCP:
    """构建并配置 FastMCP 服务端实例。

    根据运行时 Settings 配置（如 MCP_RAG_READ_ONLY）决定是否暴露写操作类工具：
    - 当 read_only=True 时，仅暴露只读查询、自检工具与 MCP Resource 只读图片网关；
    - 当 read_only=False 时，额外注册写操作、索引、建库与删除类工具。

    Args:
        settings: 可选的自定义运行时配置，缺省时按调用时机动态读取当前全局配置。

    Returns:
        FastMCP: 配置并注册完成的 FastMCP 服务端实例。
    """
    cfg = settings or get_settings()
    resolve_cfg: Callable[[], Settings] = (
        (lambda: settings) if settings is not None else get_settings
    )

    def _list_dynamic_image_resources() -> list[MCPResource]:
        active_cfg = resolve_cfg()
        catalog = get_metadata_catalog(active_cfg)
        return [MCPResource(**entry) for entry in list_gateway_asset_descriptors(catalog)]

    def _read_image_asset_sync(kb_name: str, doc_id: str, filename: str) -> tuple[bytes, str]:
        active_cfg = resolve_cfg()
        catalog = get_metadata_catalog(active_cfg)
        return read_gateway_image_asset(active_cfg, catalog, kb_name, doc_id, filename)

    server = _RagMCPServer(
        name="mcp-server-rag",
        instructions="Model Context Protocol Server for Retrieval-Augmented Generation (RAG).",
        dynamic_resources_loader=_list_dynamic_image_resources,
    )

    # 0. 注册多模态图片资产 MCP Resource Gateway（只读与读写模式下均保持可用）
    asset_template = _ImageAssetResourceTemplate.from_function(
        fn=_read_image_asset_sync,
        uri_template="rag://assets/{kb_name}/{doc_id}/{filename}",
        name="rag_image_asset",
        title="RAG 图片资产共享网关",
        description="按知识库、文档标识与文件名读取已登记的多模态图片二进制流与 MIME 类型",
        mime_type="application/octet-stream",
    )
    server._resource_manager._templates[asset_template.uri_template] = asset_template

    # 1. 注册核心只读工具
    server.add_tool(health_check)
    server.add_tool(rag_list_kbs)
    server.add_tool(rag_doctor)
    server.add_tool(rag_search)
    server.add_tool(rag_query_with_context)

    # 2. 若处于读写模式，注册写操作与索引工具；只读模式下物理隐藏
    if not cfg.read_only:
        server.add_tool(rag_create_kb)
        server.add_tool(rag_clear_kb)
        server.add_tool(rag_delete_kb)
        server.add_tool(rag_delete_document)
        server.add_tool(rag_index_text)
        server.add_tool(rag_index_file)
        server.add_tool(rag_index_directory)
        server.add_tool(rag_index_url)

    return server


# 默认全局 MCP 服务实例
mcp: FastMCP = create_server()


def main() -> None:
    """运行 MCP 服务主入口。"""
    settings = get_settings()
    server = create_server(settings)
    server.run()


if __name__ == "__main__":
    main()
