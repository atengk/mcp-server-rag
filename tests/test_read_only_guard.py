"""
端到端与接缝测试：只读安全守卫与全局配置规范 (Issue #9)。

在单一最高接缝（FastMCP Tool 协议执行层边界与 MCPServer 工具注册清单）测试：
1. 当 MCP_RAG_READ_ONLY=true 时，FastMCP 协议握手层物理隐藏全部写操作工具；
2. 仅暴露只读查询与自检工具（rag_search, rag_query_with_context, rag_doctor 等）；
3. 写操作在只读模式下被双层防线严格阻断；
4. 全局所有 MCP_RAG_* 环境变量契约解析与缺省值回退。

@author Ateng
@since 2026-11-11
"""

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from mcp_server_rag.config import Settings, get_settings
from mcp_server_rag.server import (
    create_server,
    rag_clear_kb,
    rag_create_kb,
    rag_delete_document,
    rag_delete_kb,
    rag_doctor,
    rag_index_directory,
    rag_index_file,
    rag_index_text,
    rag_index_url,
)


@pytest.fixture(autouse=True)
def test_data_dir() -> Generator[str, None, None]:
    """为每个测试用例隔离独立的临时数据目录，并在测试后清理。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_guard_")
    old_data_dir = os.environ.get("MCP_RAG_DATA_DIR")
    old_read_only = os.environ.get("MCP_RAG_READ_ONLY")
    os.environ["MCP_RAG_DATA_DIR"] = temp_dir
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    yield temp_dir
    if old_data_dir is not None:
        os.environ["MCP_RAG_DATA_DIR"] = old_data_dir
    else:
        os.environ.pop("MCP_RAG_DATA_DIR", None)

    if old_read_only is not None:
        os.environ["MCP_RAG_READ_ONLY"] = old_read_only
    else:
        os.environ.pop("MCP_RAG_READ_ONLY", None)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.anyio
async def test_read_only_guard_physically_hides_write_tools() -> None:
    """验证当 MCP_RAG_READ_ONLY=true 时，协议握手层仅暴露只读类工具，写工具彻底隐藏。"""
    os.environ["MCP_RAG_READ_ONLY"] = "true"
    settings = get_settings()
    assert settings.read_only is True

    server = create_server(settings)
    tools = await server.list_tools()
    tool_names = {t.name for t in tools}

    # 1. 预期暴露的只读工具集合
    expected_read_tools = {
        "health_check",
        "rag_list_kbs",
        "rag_doctor",
        "rag_search",
        "rag_query_with_context",
    }
    assert tool_names == expected_read_tools

    # 2. 预期物理隐藏的写工具集合
    forbidden_write_tools = {
        "rag_create_kb",
        "rag_clear_kb",
        "rag_delete_kb",
        "rag_delete_document",
        "rag_index_text",
        "rag_index_file",
        "rag_index_directory",
        "rag_index_url",
    }
    for wt in forbidden_write_tools:
        assert wt not in tool_names


@pytest.mark.anyio
async def test_writable_mode_registers_all_tools() -> None:
    """验证当 MCP_RAG_READ_ONLY=false 时，注册全部 13 个工具。"""
    os.environ["MCP_RAG_READ_ONLY"] = "false"
    settings = get_settings()
    assert settings.read_only is False

    server = create_server(settings)
    tools = await server.list_tools()
    tool_names = {t.name for t in tools}

    all_expected_tools = {
        "health_check",
        "rag_create_kb",
        "rag_list_kbs",
        "rag_clear_kb",
        "rag_delete_kb",
        "rag_delete_document",
        "rag_doctor",
        "rag_index_text",
        "rag_index_file",
        "rag_index_directory",
        "rag_index_url",
        "rag_search",
        "rag_query_with_context",
    }
    assert tool_names == all_expected_tools


def test_read_only_direct_call_defense_in_depth(tmp_path: Path) -> None:
    """验证在只读模式下，若写工具函数被直接调用，提供第二层主动拦截与防御提示。"""
    os.environ["MCP_RAG_READ_ONLY"] = "true"

    # 尝试建库
    res_create = rag_create_kb("kb_guard_test")
    assert res_create["success"] is False
    assert "只读模式" in res_create["message"]

    # 尝试清库
    res_clear = rag_clear_kb("kb_guard_test")
    assert res_clear["success"] is False
    assert "只读模式" in res_clear["message"]

    # 尝试删库
    res_del_kb = rag_delete_kb("kb_guard_test")
    assert res_del_kb["success"] is False
    assert "只读模式" in res_del_kb["message"]

    # 尝试删文档
    res_del_doc = rag_delete_document("kb_guard_test", "doc_1")
    assert res_del_doc["success"] is False
    assert "只读模式" in res_del_doc["message"]

    # 尝试索引文本
    res_text = rag_index_text("kb_guard_test", "测试文本", "标题")
    assert res_text["success"] is False
    assert "只读模式" in res_text["message"]

    # 尝试索引文件
    test_f = tmp_path / "sample.txt"
    test_f.write_text("hello", encoding="utf-8")
    res_file = rag_index_file("kb_guard_test", str(test_f))
    assert res_file["success"] is False
    assert "只读模式" in res_file["message"]

    # 尝试索引目录
    res_dir = rag_index_directory("kb_guard_test", str(tmp_path))
    assert res_dir["success"] is False
    assert "只读模式" in res_dir["message"]

    # 尝试索引 URL
    res_url = rag_index_url("kb_guard_test", "https://example.com")
    assert res_url["success"] is False
    assert "只读模式" in res_url["message"]

    # 验证 rag_doctor 诊断输出正确反映 read_only 状态
    doc_report = rag_doctor()
    assert doc_report["read_only"] is True


def test_mcp_rag_configuration_contract_and_defaults() -> None:
    """验证所有 MCP_RAG_* 环境变量契约、缺省值与容错回退机制。"""
    # 1. 默认缺省值校验
    default_settings = Settings()
    assert default_settings.vector_store == "chromadb"
    assert default_settings.embedding_provider == "fastembed"
    assert default_settings.embedding_model == "BAAI/bge-small-zh-v1.5"
    assert default_settings.read_only is False
    assert default_settings.chunk_size == 600
    assert default_settings.chunk_overlap == 80
    assert default_settings.max_files_batch == 200
    assert default_settings.milvus_uri == "http://localhost:19530"
    assert default_settings.milvus_token == ""

    # 2. 畸变非法数值安全回退（防系统崩溃）
    os.environ["MCP_RAG_CHUNK_SIZE"] = "invalid_number"
    os.environ["MCP_RAG_CHUNK_OVERLAP"] = "-50"
    os.environ["MCP_RAG_MAX_FILES_BATCH"] = "abc"
    resilient_settings = get_settings()
    assert resilient_settings.chunk_size == 600
    assert resilient_settings.chunk_overlap == 80
    assert resilient_settings.max_files_batch == 200

    # 3. 多种布尔值表现形式解析校验
    for truthy_val in ("1", "true", "True", "YES", "on"):
        os.environ["MCP_RAG_READ_ONLY"] = truthy_val
        assert get_settings().read_only is True

    for falsy_val in ("0", "false", "False", "NO", "off", ""):
        os.environ["MCP_RAG_READ_ONLY"] = falsy_val
        assert get_settings().read_only is False
