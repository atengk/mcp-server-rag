"""
端到端测试：纯文本切片摄取、内容指纹与幂等更新 (Issue #4)。

在单一最高接缝（FastMCP Tool 协议执行层边界）测试 rag_index_text
的内容指纹去重（SHA-256）、向量嵌入、SQLite 元数据与 ChromaDB 写入。

@author Ateng
@since 2026-11-11
"""

import os
import shutil
import tempfile
from collections.abc import Generator
from typing import Any

import pytest

from mcp_server_rag.server import rag_create_kb, rag_index_text, rag_list_kbs


@pytest.fixture(autouse=True)
def test_data_dir() -> Generator[str, None, None]:
    """为每个测试用例隔离独立的临时数据目录，并在测试后清理。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_ingest_")
    old_data_dir = os.environ.get("MCP_RAG_DATA_DIR")
    os.environ["MCP_RAG_DATA_DIR"] = temp_dir
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    yield temp_dir
    if old_data_dir is not None:
        os.environ["MCP_RAG_DATA_DIR"] = old_data_dir
    else:
        os.environ.pop("MCP_RAG_DATA_DIR", None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_index_text_basic() -> None:
    """验证纯文本切片、嵌入与写入知识库的端到端能力。"""
    rag_create_kb(name="tech_docs", description="技术知识库")

    sample_text = (
        "Model Context Protocol (MCP) 是用于大模型连接外部工具的标准协议。\n\n"
        "检索增强生成 (RAG) 通过将私有文档转化为高维向量并建立近邻索引，"
        "在模型推理时按需召回最相关的上下文知识片断，从而有效克服大模型的幻觉问题。\n\n"
        "FastMCP 是现代高性能 Python MCP 服务端开发框架。"
    )

    res = rag_index_text(
        kb_name="tech_docs",
        text=sample_text,
        title="mcp_and_rag_intro",
        metadata={"author": "Ateng", "category": "architecture"},
    )

    assert res["success"] is True
    assert res["status"] == "indexed"
    assert res["doc_id"] == "mcp_and_rag_intro"
    assert res["chunk_count"] >= 1
    assert "content_hash" in res

    # 验证知识库统计更新
    kbs: list[dict[str, Any]] = rag_list_kbs()
    kb = next(k for k in kbs if k["name"] == "tech_docs")
    assert kb["document_count"] == 1
    assert kb["chunk_count"] == res["chunk_count"]


def test_index_text_idempotent_skip() -> None:
    """验证相同内容再次索引时，通过 SHA-256 哈希比对秒级跳过（零算力消耗）。"""
    rag_create_kb(name="idempotent_kb")
    text = "这是一篇关于持续集成流水线规范的测试文档内容。"

    # 1. 首次索引
    res1 = rag_index_text(kb_name="idempotent_kb", text=text, title="ci_spec")
    assert res1["success"] is True
    assert res1["status"] == "indexed"

    # 2. 相同标题和内容重复索引
    res2 = rag_index_text(kb_name="idempotent_kb", text=text, title="ci_spec")
    assert res2["success"] is True
    assert res2["status"] == "skipped"
    assert "unchanged" in res2["message"].lower() or "未变动" in res2["message"]

    # 3. 验证文档与切片总数未翻倍
    kbs = rag_list_kbs()
    kb = next(k for k in kbs if k["name"] == "idempotent_kb")
    assert kb["document_count"] == 1
    assert kb["chunk_count"] == res1["chunk_count"]


def test_index_text_modified_replaces_old_chunks() -> None:
    """验证内容被修改时，系统原子删除旧切片并替换为新切片。"""
    rag_create_kb(name="update_kb")

    initial_text = "版本 1.0 的初始简短内容。"
    res1 = rag_index_text(kb_name="update_kb", text=initial_text, title="doc_v1")
    assert res1["status"] == "indexed"

    # 更新文本（大幅扩展内容）
    updated_text = (
        "版本 2.0 的全面重构内容。\n\n"
        + "这是详细的第二章节段落，包含了大量的扩展架构说明与系统配置要点。\n\n" * 5
    )
    res2 = rag_index_text(kb_name="update_kb", text=updated_text, title="doc_v1")
    assert res2["success"] is True
    assert res2["status"] == "updated"
    assert res2["content_hash"] != res1["content_hash"]

    # 文档数量仍为 1，切片数更新为新切片数
    kbs = rag_list_kbs()
    kb = next(k for k in kbs if k["name"] == "update_kb")
    assert kb["document_count"] == 1
    assert kb["chunk_count"] == res2["chunk_count"]


def test_index_text_nonexistent_kb() -> None:
    """验证向不存在的知识库索引时返回友好错误提示。"""
    res = rag_index_text(
        kb_name="nonexistent_kb",
        text="某些内容",
        title="some_doc",
    )
    assert res["success"] is False
    assert "not found" in res["message"].lower() or "不存在" in res["message"]
