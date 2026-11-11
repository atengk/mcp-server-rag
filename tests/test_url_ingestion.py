"""
端到端测试：网页与在线 URL 文档抓取摄取 (Issue #7)。

在单一最高接缝（FastMCP Tool 协议执行层边界）测试 rag_index_url
的 HTML 网页正文抽取、标题解析、内容指纹去重、超时与异常网络防护。

@author Ateng
@since 2026-11-11
"""

import os
import shutil
import tempfile
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from mcp_server_rag.server import (
    rag_create_kb,
    rag_index_url,
    rag_search,
)


@pytest.fixture(autouse=True)
def test_data_dir() -> Generator[str, None, None]:
    """为每个测试用例隔离独立的临时数据目录，并在测试后清理。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_url_")
    old_data_dir = os.environ.get("MCP_RAG_DATA_DIR")
    os.environ["MCP_RAG_DATA_DIR"] = temp_dir
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    yield temp_dir
    if old_data_dir is not None:
        os.environ["MCP_RAG_DATA_DIR"] = old_data_dir
    else:
        os.environ.pop("MCP_RAG_DATA_DIR", None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_index_url_html_success() -> None:
    """验证从在线 URL 抓取 HTML 网页、提取标题与正文并完成向量切片索引。"""
    rag_create_kb(name="kb_web", description="网页知识库")

    sample_html = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        "    <title>Model Context Protocol 官方架构说明</title>\n"
        "    <script>console.log('ignored script');</script>\n"
        "    <style>.nav { color: red; }</style>\n"
        "</head>\n"
        "<body>\n"
        "    <header><nav>Home | Docs | Contact</nav></header>\n"
        "    <h1>MCP 协议概述</h1>\n"
        "    <p>Model Context Protocol (MCP) 是一项开放规范，"
        "用于大模型与外部知识库及执行工具安全集成。</p>\n"
        "    <p>通过统一的客户端与服务端协议，Agent 能够按需检索领域文档切片。</p>\n"
        "    <footer>Copyright 2026 Example Corp</footer>\n"
        "</body>\n"
        "</html>"
    )

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = sample_html
    mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
    mock_resp.raise_for_status = MagicMock()

    target_url = "https://example.com/docs/mcp-overview"

    with patch("httpx.get", return_value=mock_resp):
        res: dict[str, Any] = rag_index_url(
            kb_name="kb_web",
            url=target_url,
            metadata={"category": "official_doc"},
        )

    assert res["success"] is True
    assert res["status"] == "indexed"
    assert res["doc_id"] == target_url
    assert res["url"] == target_url
    assert "Model Context Protocol 官方架构说明" in res["title"]
    assert res["chunk_count"] >= 1
    assert "content_hash" in res

    # 验证可通过向量检索召回抓取到的网页正文
    search_results = rag_search(
        kb_name="kb_web",
        query="什么是 MCP 协议规范",
        top_k=1,
    )
    assert len(search_results) == 1
    top_chunk = search_results[0]
    assert "Model Context Protocol" in top_chunk["text"]
    assert "大模型与外部知识库" in top_chunk["text"]
    # 验证视觉噪点与脚本已被清洗
    assert "ignored script" not in top_chunk["text"]
    assert "Copyright 2026" not in top_chunk["text"]


def test_index_url_idempotent_skip() -> None:
    """验证相同 URL 再次摄取时，内容未变动秒级跳过。"""
    rag_create_kb(name="kb_url_skip")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = (
        "<html><head><title>Test</title></head><body><p>Hello World Content</p></body></html>"
    )
    mock_resp.headers = {"content-type": "text/html"}
    mock_resp.raise_for_status = MagicMock()

    url = "https://example.com/page1"

    with patch("httpx.get", return_value=mock_resp):
        res1 = rag_index_url(kb_name="kb_url_skip", url=url)
        assert res1["success"] is True
        assert res1["status"] == "indexed"

        res2 = rag_index_url(kb_name="kb_url_skip", url=url)
        assert res2["success"] is True
        assert res2["status"] == "skipped"


def test_index_url_http_error_handling() -> None:
    """验证非 2xx HTTP 错误码妥善处理并返回错误提示，不引起服务崩溃。"""
    rag_create_kb(name="kb_url_err")

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        message="404 Not Found",
        request=MagicMock(),
        response=mock_resp,
    )

    with patch("httpx.get", return_value=mock_resp):
        res = rag_index_url(kb_name="kb_url_err", url="https://example.com/not-found")

    assert res["success"] is False
    assert "404" in res["message"] or "失败" in res["message"]


def test_index_url_network_timeout_handling() -> None:
    """验证网络连接超时与连接中断异常被安全捕获。"""
    rag_create_kb(name="kb_url_timeout")

    with patch("httpx.get", side_effect=httpx.ConnectTimeout("Connection timed out")):
        res = rag_index_url(kb_name="kb_url_timeout", url="https://timeout.example.com")

    assert res["success"] is False
    assert "超时" in res["message"] or "timeout" in res["message"].lower()


def test_index_url_invalid_scheme() -> None:
    """验证非法 URL 协议方案（如 ftp:// 或裸字符）被直接拦截。"""
    rag_create_kb(name="kb_url_scheme")

    res = rag_index_url(kb_name="kb_url_scheme", url="ftp://example.com/file.txt")
    assert res["success"] is False
    assert "http://" in res["message"] or "https://" in res["message"]
