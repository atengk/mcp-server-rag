"""
FastMCP 服务主实现模块。

注册并暴露知识库生命周期管理、健康自检探针与核心检索工具。

@author Ateng
@since 2026-11-11
"""

import re
import shutil
from typing import Any

from mcp.server import MCPServer

from mcp_server_rag.config import get_settings
from mcp_server_rag.metadata import MetadataCatalog
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


def main() -> None:
    """运行 MCP 服务主入口。"""
    mcp.run()


if __name__ == "__main__":
    main()
