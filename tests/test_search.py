"""
端到端测试：语义搜索、精准关键词过滤与 Markdown 上下文注入 (Issue #5)。

在单一最高接缝（FastMCP Tool 协议执行层边界）测试 rag_search 与 rag_query_with_context
的高维向量相似度召回、相似度评分阈值截断、关键词过滤、元数据过滤与 Markdown 格式化输出。

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
    rag_create_kb,
    rag_index_text,
    rag_query_with_context,
    rag_search,
)


@pytest.fixture(autouse=True)
def test_data_dir() -> Generator[str, None, None]:
    """为每个测试用例隔离独立的临时数据目录，并在测试后清理。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_search_")
    old_data_dir = os.environ.get("MCP_RAG_DATA_DIR")
    os.environ["MCP_RAG_DATA_DIR"] = temp_dir
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    yield temp_dir
    if old_data_dir is not None:
        os.environ["MCP_RAG_DATA_DIR"] = old_data_dir
    else:
        os.environ.pop("MCP_RAG_DATA_DIR", None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_rag_search_semantic_ordering() -> None:
    """验证语义搜索能按相似度降序返回 Top-K 切片及其分数与元数据。"""
    rag_create_kb(name="kb_semantic", description="语义检索知识库")

    # 写入三篇不同领域的文档
    rag_index_text(
        kb_name="kb_semantic",
        title="python_async",
        text="Python 是一种高级编程语言，通过 asyncio 模块提供原生协程与异步并发支持。",
        metadata={"category": "python", "priority": "high"},
    )
    rag_index_text(
        kb_name="kb_semantic",
        title="k8s_cluster",
        text="Kubernetes 是容器集群自动部署、扩缩和管理的开源系统，提供 Pod 编排与服务发现机制。",
        metadata={"category": "cloud", "priority": "medium"},
    )
    rag_index_text(
        kb_name="kb_semantic",
        title="cooking_recipe",
        text="红烧肉是一道著名的大众家常菜，选用五花肉为主料，加入冰糖老抽慢火焖炖入味。",
        metadata={"category": "cooking", "priority": "low"},
    )

    # 针对 Python 异步编程执行检索
    results: list[dict[str, Any]] = rag_search(
        kb_name="kb_semantic",
        query="Python 异步并发编程与协程",
        top_k=3,
    )

    assert len(results) >= 1
    # 验证返回切片的数据契约
    top_hit = results[0]
    assert "chunk_id" in top_hit
    assert "text" in top_hit
    assert "score" in top_hit
    assert "metadata" in top_hit
    assert "kb_name" in top_hit
    assert top_hit["kb_name"] == "kb_semantic"
    assert "asyncio" in top_hit["text"]
    assert top_hit["metadata"].get("category") == "python"

    # 验证相似度排序降序
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert top_hit["score"] > 0.4


def test_rag_search_threshold_filtering() -> None:
    """验证相似度评分阈值截断（threshold），过滤掉低于指定分数的无关切片。"""
    rag_create_kb(name="kb_threshold")

    rag_index_text(
        kb_name="kb_threshold",
        title="python_doc",
        text="FastMCP 是现代高性能 Python MCP 服务端开发框架，简化了工具与资源注入流程。",
    )
    rag_index_text(
        kb_name="kb_threshold",
        title="cake_doc",
        text="制作提拉米苏蛋糕需要准备马斯卡彭芝士、手指饼干、咖啡力娇酒与可可粉。",
    )

    # 1. 阈值极高（0.999），期望过滤所有结果
    high_threshold_results = rag_search(
        kb_name="kb_threshold",
        query="Python MCP 框架",
        top_k=5,
        threshold=0.999,
    )
    assert high_threshold_results == []

    # 2. 合理阈值，仅召回强相关的 Python 文档，剔除无关蛋糕文档
    normal_results = rag_search(
        kb_name="kb_threshold",
        query="Python MCP 框架",
        top_k=5,
        threshold=0.4,
    )
    assert len(normal_results) >= 1
    for r in normal_results:
        assert r["score"] >= 0.4
        assert "蛋糕" not in r["text"]


def test_rag_search_keyword_exact_filter() -> None:
    """验证精准关键词（keyword）过滤，解决专用错误码漏召回与混淆问题。"""
    rag_create_kb(name="kb_keyword")

    rag_index_text(
        kb_name="kb_keyword",
        title="err_timeout",
        text="服务通信发生故障，抛出错误代码 ERR_GATEWAY_TIMEOUT_504，请排查反向代理超时时间配置。",
    )
    rag_index_text(
        kb_name="kb_keyword",
        title="err_internal",
        text="服务通信发生故障，抛出错误代码 ERR_INTERNAL_SERVER_500，服务端发生未知未捕获异常。",
    )

    # 语义查询“服务故障”，但精准锁定 504 错误码
    results = rag_search(
        kb_name="kb_keyword",
        query="服务通信故障排查",
        top_k=5,
        keyword="ERR_GATEWAY_TIMEOUT_504",
    )

    assert len(results) == 1
    assert "ERR_GATEWAY_TIMEOUT_504" in results[0]["text"]
    assert "ERR_INTERNAL_SERVER_500" not in results[0]["text"]


def test_rag_search_metadata_filter() -> None:
    """验证元数据字典过滤（metadata_filter），根据属性限制检索范围。"""
    rag_create_kb(name="kb_metadata")

    rag_index_text(
        kb_name="kb_metadata",
        title="backend_doc",
        text="Spring Cloud 与 FastMCP 微服务通信架构与协议解析。",
        metadata={"category": "backend", "level": "advanced"},
    )
    rag_index_text(
        kb_name="kb_metadata",
        title="frontend_doc",
        text="React 与 Vue3 前端组件化状态管理与单向数据流。",
        metadata={"category": "frontend", "level": "intermediate"},
    )

    # 单条件过滤 category = backend
    results_backend = rag_search(
        kb_name="kb_metadata",
        query="微服务与前端架构设计",
        metadata_filter={"category": "backend"},
    )
    assert len(results_backend) == 1
    assert results_backend[0]["metadata"]["category"] == "backend"

    # 多条件复合过滤
    results_multi = rag_search(
        kb_name="kb_metadata",
        query="组件化通信设计",
        metadata_filter={"category": "frontend", "level": "intermediate"},
    )
    assert len(results_multi) == 1
    assert results_multi[0]["metadata"]["category"] == "frontend"

    # 无匹配元数据条件
    results_none = rag_search(
        kb_name="kb_metadata",
        query="微服务",
        metadata_filter={"category": "database"},
    )
    assert results_none == []


def test_rag_search_nonexistent_and_empty_kb() -> None:
    """验证知识库不存在或为空时的空安全返回契约（严禁返回 None 或抛异常）。"""
    # 1. 不存在的知识库
    res_nonexistent = rag_search(
        kb_name="not_exist_kb",
        query="任意查询内容",
    )
    assert res_nonexistent == []

    # 2. 存在的空知识库
    rag_create_kb(name="empty_kb")
    res_empty = rag_search(
        kb_name="empty_kb",
        query="任意查询内容",
    )
    assert res_empty == []


def test_rag_query_with_context_empty() -> None:
    """验证当知识库中未检索到任何匹配切片时，返回友好的无内容提示。"""
    rag_create_kb(name="kb_empty_context")

    output = rag_query_with_context(
        kb_name="kb_empty_context",
        query="任意查询",
    )
    assert "未检索到与查询相关的上下文内容" in output


def test_rag_query_with_context_success() -> None:
    """验证成功召回时输出标准 Markdown 引用格式的增强上下文。"""
    rag_create_kb(name="kb_ctx")

    rag_index_text(
        kb_name="kb_ctx",
        title="mcp_spec",
        text=(
            "Model Context Protocol (MCP) 是开放式通用协议。\n"
            "它允许大语言模型安全地访问私有数据源与外部工具。"
        ),
    )

    prompt = rag_query_with_context(
        kb_name="kb_ctx",
        query="什么是 MCP 协议",
        top_k=2,
    )

    assert "根据以下检索到的相关知识库参考资料回答问题：" in prompt
    assert "### 📚 参考片段 [1] (来源: mcp_spec, 相似度:" in prompt
    assert "> Model Context Protocol (MCP)" in prompt
