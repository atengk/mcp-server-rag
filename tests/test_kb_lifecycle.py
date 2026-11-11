"""
端到端测试：知识库生命周期管理与健康自检探针 (Issue #3)。

在单一最高接缝（FastMCP Tool 协议执行层边界）测试知识库的
创建、枚举、清空、删除与健康诊断行为。

@author Ateng
@since 2026-11-11
"""

import os
import shutil
import tempfile
from collections.abc import Generator
from typing import Any

import pytest

from mcp_server_rag.server import (
    rag_clear_kb,
    rag_create_kb,
    rag_delete_kb,
    rag_doctor,
    rag_list_kbs,
)


@pytest.fixture(autouse=True)
def test_data_dir() -> Generator[str, None, None]:
    """为每个测试用例隔离独立的临时数据目录，并在测试后清理。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_data_")
    old_data_dir = os.environ.get("MCP_RAG_DATA_DIR")
    os.environ["MCP_RAG_DATA_DIR"] = temp_dir
    yield temp_dir
    if old_data_dir is not None:
        os.environ["MCP_RAG_DATA_DIR"] = old_data_dir
    else:
        os.environ.pop("MCP_RAG_DATA_DIR", None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_create_and_list_kb() -> None:
    """验证创建知识库并在列表中查询到其完整元数据。"""
    # 1. 初始列表应为空
    kbs: list[dict[str, Any]] = rag_list_kbs()
    assert len(kbs) == 0

    # 2. 创建一个新知识库
    res = rag_create_kb(name="project_docs", description="项目核心技术文档库")
    assert res["success"] is True
    assert res["name"] == "project_docs"

    # 3. 再次查询列表，应包含该知识库
    kbs = rag_list_kbs()
    assert len(kbs) == 1
    kb = kbs[0]
    assert kb["name"] == "project_docs"
    assert kb["description"] == "项目核心技术文档库"
    assert kb["document_count"] == 0
    assert kb["chunk_count"] == 0
    assert "created_at" in kb

    # 4. 重复创建同名知识库应友好拦截
    dup_res = rag_create_kb(name="project_docs")
    assert dup_res["success"] is False
    assert "already exists" in dup_res["message"].lower() or "已存在" in dup_res["message"]


def test_clear_kb() -> None:
    """验证清空知识库切片数据但保留知识库本身。"""
    rag_create_kb(name="temp_kb", description="待清空知识库")

    # 清空知识库
    clear_res = rag_clear_kb(name="temp_kb")
    assert clear_res["success"] is True
    assert clear_res["name"] == "temp_kb"

    # 知识库仍应存在于列表中
    kbs = rag_list_kbs()
    assert any(kb["name"] == "temp_kb" for kb in kbs)

    # 清空不存在的知识库应失败
    bad_res = rag_clear_kb(name="non_existent_kb")
    assert bad_res["success"] is False


def test_delete_kb() -> None:
    """验证物理销毁知识库。"""
    rag_create_kb(name="obsolete_kb", description="过时知识库")
    assert any(kb["name"] == "obsolete_kb" for kb in rag_list_kbs())

    # 物理删除
    del_res = rag_delete_kb(name="obsolete_kb")
    assert del_res["success"] is True

    # 删除后列表中不再包含该库
    assert not any(kb["name"] == "obsolete_kb" for kb in rag_list_kbs())

    # 再次删除应提示未找到
    del_again = rag_delete_kb(name="obsolete_kb")
    assert del_again["success"] is False


def test_invalid_kb_name() -> None:
    """验证非法知识库名称被正确拦截。"""
    res = rag_create_kb(name="invalid/name with spaces!")
    assert res["success"] is False
    assert "仅支持" in res["message"] or "英文字母" in res["message"]

    empty_res = rag_create_kb(name="")
    assert empty_res["success"] is False


def test_doctor_probe() -> None:
    """验证系统自检诊断探针报告输出。"""
    rag_create_kb(name="health_check_kb")

    report = rag_doctor()
    assert report["status"] == "healthy"
    assert report["vector_store_status"] == "ok"
    assert report["metadata_catalog_status"] == "ok"
    assert report["knowledge_bases_count"] == 1
    assert "data_dir" in report
