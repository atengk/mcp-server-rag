"""
FastMCP 服务主实现模块。

注册并暴露知识库生命周期管理、健康自检探针与核心检索工具。

@author Ateng
@since 2026-11-11
"""

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any

import httpx
from mcp.server import MCPServer

from mcp_server_rag.config import get_settings
from mcp_server_rag.embedding import get_embedding_provider
from mcp_server_rag.metadata import MetadataCatalog
from mcp_server_rag.parsers import (
    dispatch_and_split,
    extract_html_content,
    split_markdown,
)
from mcp_server_rag.splitter import recursive_split_text
from mcp_server_rag.vector_store.chroma import ChromaVectorStoreAdapter

# 兼容 FastMCP 类型别名
FastMCP = MCPServer

# 初始化 MCP 服务实例
mcp = FastMCP(
    name="mcp-server-rag",
    instructions="Model Context Protocol Server for Retrieval-Augmented Generation (RAG).",
)


def _get_services() -> tuple[MetadataCatalog, ChromaVectorStoreAdapter]:
    """获取当前配置对应的元数据目录与向量存储适配器实例。"""
    settings = get_settings()
    catalog = MetadataCatalog(settings.data_dir)
    vector_store = ChromaVectorStoreAdapter(settings.data_dir)
    return catalog, vector_store


def _validate_kb_name(name: str) -> tuple[bool, str]:
    """校验知识库名称合规性（字母数字下划线与连字符，1~64 字符）。"""
    if not name or len(name) > 64:
        return False, "知识库名称长度必须在 1 到 64 个字符之间"
    if not re.match(r"^[a-zA-Z0-9_\-]+$", name):
        return False, "知识库名称仅支持英文字母、数字、下划线 (_) 与连字符 (-)"
    return True, ""


@mcp.tool()
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


@mcp.tool()
def rag_create_kb(name: str, description: str = "") -> dict[str, Any]:
    """创建新的知识库命名空间与底层向量存储集合。

    Args:
        name: 知识库名称，全局唯一标识（仅限字母、数字、下划线、连字符）
        description: 知识库的详细描述说明信息

    Returns:
        dict[str, Any]: 创建结果状态与说明
    """
    valid, err_msg = _validate_kb_name(name)
    if not valid:
        return {"success": False, "name": name, "message": err_msg}

    settings = get_settings()
    catalog, vector_store = _get_services()

    # 检查是否已存在
    existing = catalog.get_kb(name)
    if existing:
        return {
            "success": False,
            "name": name,
            "message": f"Knowledge base '{name}' already exists (知识库已存在)",
        }

    # 1. 在 SQLite 中注册知识库元数据
    created = catalog.create_kb(
        name=name,
        description=description,
        embedding_model=settings.embedding_model,
    )
    if not created:
        return {
            "success": False,
            "name": name,
            "message": f"Failed to register knowledge base '{name}' in metadata catalog",
        }

    # 2. 在向量存储中初始化物理集合
    vector_store.create_kb(name)

    return {
        "success": True,
        "name": name,
        "message": f"Knowledge base '{name}' created successfully",
    }


@mcp.tool()
def rag_list_kbs() -> list[dict[str, Any]]:
    """枚举所有已创建的知识库清单及其文档数、切片统计与更新时间。

    Returns:
        list[dict[str, Any]]: 包含各知识库完整元数据的字典列表
    """
    catalog, _ = _get_services()
    return catalog.list_kbs()


@mcp.tool()
def rag_clear_kb(name: str) -> dict[str, Any]:
    """清空指定知识库中的所有文档与切片数据，但保留知识库本身的基础配置与元数据。

    Args:
        name: 待清空的知识库名称

    Returns:
        dict[str, Any]: 清空结果与状态说明
    """
    catalog, vector_store = _get_services()

    # 检查知识库是否存在
    if not catalog.get_kb(name):
        return {
            "success": False,
            "name": name,
            "message": f"Knowledge base '{name}' not found (知识库不存在)",
        }

    # 1. 清空 SQLite 元数据记录
    catalog.clear_kb(name)

    # 2. 清空向量库数据
    vector_store.clear_kb(name)

    return {
        "success": True,
        "name": name,
        "message": f"Knowledge base '{name}' cleared successfully",
    }


@mcp.tool()
def rag_delete_kb(name: str) -> dict[str, Any]:
    """物理销毁指定知识库及其关联的所有文档元数据与向量索引集合。

    Args:
        name: 待删除的知识库名称

    Returns:
        dict[str, Any]: 删除结果与状态说明
    """
    catalog, vector_store = _get_services()

    # 检查知识库是否存在
    if not catalog.get_kb(name):
        return {
            "success": False,
            "name": name,
            "message": f"Knowledge base '{name}' not found (知识库不存在)",
        }

    # 1. 删除向量集合
    vector_store.delete_kb(name)

    # 2. 删除 SQLite 元数据
    catalog.delete_kb(name)

    return {
        "success": True,
        "name": name,
        "message": f"Knowledge base '{name}' deleted successfully",
    }


@mcp.tool()
def rag_doctor() -> dict[str, Any]:
    """全维自检诊断探针，检查存储引擎连通性、本地模型配置、磁盘占用与系统环境。

    Returns:
        dict[str, Any]: 结构化诊断报告
    """
    settings = get_settings()
    catalog, vector_store = _get_services()

    # 检查 SQLite 与 ChromaDB 连通性
    sqlite_ok = catalog.ping()
    chroma_ok = vector_store.ping()

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

    overall_status = "healthy" if (sqlite_ok and chroma_ok) else "degraded"

    return {
        "status": overall_status,
        "data_dir": str(settings.data_dir),
        "data_dir_size_bytes": disk_usage_bytes,
        "disk_free_bytes": free,
        "vector_store": settings.vector_store,
        "vector_store_status": "ok" if chroma_ok else "error",
        "metadata_catalog_status": "ok" if sqlite_ok else "error",
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "read_only": settings.read_only,
        "knowledge_bases_count": kb_count,
        "documents_count": doc_count,
    }


@mcp.tool()
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
    if not text or not text.strip():
        return {"success": False, "message": "正文内容不能为空"}
    if not title or not title.strip():
        return {"success": False, "message": "文档标题 (title) 不能为空"}

    settings = get_settings()
    catalog, vector_store = _get_services()

    # 1. 校验知识库是否存在
    kb = catalog.get_kb(kb_name)
    if not kb:
        return {
            "success": False,
            "message": f"Knowledge base '{kb_name}' not found (知识库不存在)",
        }

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

    # 3. 若为更新，先清理旧切片
    status = "indexed"
    if existing:
        deleted_chunk_ids = catalog.delete_document(kb_name, doc_id)
        vector_store.delete_chunks(kb_name, deleted_chunk_ids)
        status = "updated"

    # 4. 递归切片
    chunks = recursive_split_text(
        raw_text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        chunks = [raw_text]

    # 5. 计算嵌入向量
    chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    embedding_provider = get_embedding_provider(settings)
    embeddings = embedding_provider.embed_texts(chunks)

    # 6. 构造元数据并写入向量存储
    user_metadata = metadata or {}
    metadatas = [
        {
            **user_metadata,
            "doc_id": doc_id,
            "title": title,
            "kb_name": kb_name,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]
    vector_store.add_chunks(
        kb_name=kb_name,
        chunk_ids=chunk_ids,
        texts=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # 7. 写入元数据目录
    catalog.register_document_and_chunks(
        kb_name=kb_name,
        doc_id=doc_id,
        title=title,
        content_hash=content_hash,
        source_uri="",
        chunk_ids=chunk_ids,
        file_size=len(raw_text.encode("utf-8")),
    )

    return {
        "success": True,
        "status": status,
        "doc_id": doc_id,
        "chunk_count": len(chunks),
        "content_hash": content_hash,
        "message": f"Document '{title}' successfully {status}",
    }


@mcp.tool()
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
    path = Path(file_path).resolve()
    if not path.exists():
        return {"success": False, "message": f"文件不存在: {file_path}"}
    if not path.is_file():
        return {"success": False, "message": f"目标路径不是普通文件: {file_path}"}

    try:
        raw_content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            raw_content = path.read_text(encoding="gbk")
        except Exception:
            raw_content = path.read_text(encoding="latin-1", errors="replace")

    if not raw_content or not raw_content.strip():
        return {"success": False, "message": "文件正文为空"}

    settings = get_settings()
    catalog, vector_store = _get_services()

    # 1. 校验知识库是否存在
    kb = catalog.get_kb(kb_name)
    if not kb:
        return {
            "success": False,
            "message": f"Knowledge base '{kb_name}' not found (知识库不存在)",
        }

    doc_id = str(path)
    title = path.name
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

    # 3. 若为更新，先清理旧切片
    status = "indexed"
    if existing:
        deleted_chunk_ids = catalog.delete_document(kb_name, doc_id)
        vector_store.delete_chunks(kb_name, deleted_chunk_ids)
        status = "updated"

    # 4. 结构感知语法切片分发
    chunks = dispatch_and_split(
        file_path=path,
        content=raw_content,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        chunks = [raw_content]

    # 5. 生成全局唯一切片 ID 并计算向量嵌入
    path_hash = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]
    chunk_ids = [f"{path.stem}_{path_hash}_{i}" for i in range(len(chunks))]
    embedding_provider = get_embedding_provider(settings)
    embeddings = embedding_provider.embed_texts(chunks)

    # 6. 构造元数据并写入向量存储
    user_metadata = metadata or {}
    metadatas = [
        {
            **user_metadata,
            "doc_id": doc_id,
            "title": title,
            "file_path": str(path),
            "file_name": path.name,
            "file_extension": path.suffix.lower(),
            "kb_name": kb_name,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]
    vector_store.add_chunks(
        kb_name=kb_name,
        chunk_ids=chunk_ids,
        texts=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # 7. 写入元数据目录
    catalog.register_document_and_chunks(
        kb_name=kb_name,
        doc_id=doc_id,
        title=title,
        content_hash=content_hash,
        source_uri=str(path),
        chunk_ids=chunk_ids,
        file_size=len(raw_content.encode("utf-8")),
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
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
}


@mcp.tool()
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
    path = Path(dir_path).resolve()
    if not path.exists() or not path.is_dir():
        return {
            "success": False,
            "message": f"目录不存在或不是有效目录: {dir_path}",
        }

    catalog, _ = _get_services()
    kb = catalog.get_kb(kb_name)
    if not kb:
        return {
            "success": False,
            "message": f"Knowledge base '{kb_name}' not found (知识库不存在)",
        }

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
        res = rag_index_file(kb_name=kb_name, file_path=str(file_p))
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


@mcp.tool()
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
    clean_url = url.strip()
    if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
        return {
            "success": False,
            "message": f"URL 协议无效，必须以 http:// 或 https:// 开头: {url}",
        }

    settings = get_settings()
    catalog, vector_store = _get_services()
    kb = catalog.get_kb(kb_name)
    if not kb:
        return {
            "success": False,
            "message": f"Knowledge base '{kb_name}' not found (知识库不存在)",
        }

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

    status = "indexed"
    if existing:
        deleted_chunk_ids = catalog.delete_document(kb_name, doc_id)
        vector_store.delete_chunks(kb_name, deleted_chunk_ids)
        status = "updated"

    # 切片分发
    if "markdown" in content_type:
        chunks = split_markdown(text_content, settings.chunk_size, settings.chunk_overlap)
    else:
        chunks = recursive_split_text(text_content, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        chunks = [text_content]

    url_hash = hashlib.md5(clean_url.encode("utf-8")).hexdigest()[:8]
    chunk_ids = [f"url_{url_hash}_{i}" for i in range(len(chunks))]
    embedding_provider = get_embedding_provider(settings)
    embeddings = embedding_provider.embed_texts(chunks)

    user_metadata = metadata or {}
    metadatas = [
        {
            **user_metadata,
            "doc_id": doc_id,
            "title": title,
            "url": clean_url,
            "source_uri": clean_url,
            "content_type": content_type,
            "kb_name": kb_name,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]
    vector_store.add_chunks(
        kb_name=kb_name,
        chunk_ids=chunk_ids,
        texts=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    catalog.register_document_and_chunks(
        kb_name=kb_name,
        doc_id=doc_id,
        title=title,
        content_hash=content_hash,
        source_uri=clean_url,
        chunk_ids=chunk_ids,
        file_size=len(text_content.encode("utf-8")),
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


@mcp.tool()
def rag_search(
    kb_name: str,
    query: str,
    top_k: int = 5,
    threshold: float = 0.0,
    keyword: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """在指定知识库中执行高维向量语义搜索，支持阈值过滤、精准关键词过滤与元数据过滤。

    Args:
        kb_name: 目标知识库名称
        query: 检索查询文本
        top_k: 返回相似切片的最大数量（默认为 5）
        threshold: 相似度截断阈值，过滤掉低于此分数的切片（范围 0.0 ~ 1.0）
        keyword: 可选的精准关键词，切片正文必须包含此词（大小写无关）
        metadata_filter: 可选的元数据字典过滤条件（如 {"category": "tech"}）

    Returns:
        list[dict[str, Any]]: 排序后的相似切片列表，
            每个切片包含 chunk_id, text, score, metadata, kb_name
    """
    if not query or not query.strip():
        return []

    settings = get_settings()
    catalog, vector_store = _get_services()

    # 1. 校验知识库是否存在
    kb = catalog.get_kb(kb_name)
    if not kb:
        return []

    # 2. 构造底层向量数据库过滤条件
    where = _build_chroma_where(metadata_filter)
    where_document = _build_chroma_where_document(keyword)

    # 3. 计算查询向量
    embedding_provider = get_embedding_provider(settings)
    query_vector = embedding_provider.embed_query(query.strip())

    # 4. 执行向量检索召回候选切片
    candidates = vector_store.search(
        kb_name=kb_name,
        query_embedding=query_vector,
        top_k=top_k,
        where=where,
        where_document=where_document,
    )

    # 5. 精确过滤（阈值截断、关键词与元数据二次校验）
    filtered: list[dict[str, Any]] = []
    clean_kw = keyword.strip().lower() if keyword and keyword.strip() else None

    for item in candidates:
        # 相似度阈值过滤
        if item["score"] < threshold:
            continue
        # 精准关键词严格匹配验证
        if clean_kw and clean_kw not in item["text"].lower():
            continue
        # 元数据严格匹配二次校验
        if metadata_filter:
            item_meta = item.get("metadata", {})
            match = True
            for mk, mv in metadata_filter.items():
                if not mk.startswith("$") and item_meta.get(mk) != mv:
                    match = False
                    break
            if not match:
                continue
        filtered.append(item)

    return filtered[:top_k]


@mcp.tool()
def rag_query_with_context(
    kb_name: str,
    query: str,
    top_k: int = 5,
    threshold: float = 0.0,
    keyword: str | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> str:
    """检索知识库相关上下文并组装为带来源引用的 Markdown 增强提示词。

    Args:
        kb_name: 目标知识库名称
        query: 检索查询问题或主题
        top_k: 返回参考切片的最大数量（默认为 5）
        threshold: 相似度截断阈值，过滤掉低于此分数的切片
        keyword: 可选的精准关键词
        metadata_filter: 可选的元数据字典过滤条件

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
    )

    if not chunks:
        return "未检索到与查询相关的上下文内容。"

    context_sections: list[str] = ["根据以下检索到的相关知识库参考资料回答问题：\n"]
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        title = meta.get("title") or meta.get("doc_id") or chunk.get("chunk_id", f"文档_{i}")
        score = chunk.get("score", 0.0)
        raw_text = chunk.get("text", "").strip()
        quoted_text = (
            "\n".join(f"> {line}" for line in raw_text.splitlines()) if raw_text else "> (空内容)"
        )
        section = f"### 📚 参考片段 [{i}] (来源: {title}, 相似度: {score:.4f})\n{quoted_text}\n"
        context_sections.append(section)

    return "\n".join(context_sections).strip()


def main() -> None:
    """运行 MCP 服务主入口。"""
    mcp.run()


if __name__ == "__main__":
    main()
