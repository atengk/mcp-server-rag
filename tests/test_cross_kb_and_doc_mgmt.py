"""
端到端测试：跨知识库联合召回与细粒度文档清理 (Issue #8)。

在单一最高接缝（FastMCP Tool 协议执行层边界）测试：
1. 细粒度文档清理 (rag_delete_document) 的原子删除、计数同步与向量清理；
2. 跨知识库联合语义召回 (rag_search 传入 kb_name="*" 或列表)；
3. 跨知识库增强上下文组装 (rag_query_with_context)。

@author Ateng
@since 2026-11-11
"""

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from mcp_server_rag.server import (
    rag_clear_kb,
    rag_create_kb,
    rag_delete_document,
    rag_index_file,
    rag_index_text,
    rag_list_kbs,
    rag_query_with_context,
    rag_search,
)


@pytest.fixture(autouse=True)
def test_data_dir() -> Generator[str, None, None]:
    """为每个测试用例隔离独立的临时数据目录，并在测试后清理。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_issue8_")
    old_data_dir = os.environ.get("MCP_RAG_DATA_DIR")
    os.environ["MCP_RAG_DATA_DIR"] = temp_dir
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    yield temp_dir
    if old_data_dir is not None:
        os.environ["MCP_RAG_DATA_DIR"] = old_data_dir
    else:
        os.environ.pop("MCP_RAG_DATA_DIR", None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_rag_delete_document_success(tmp_path: Path) -> None:
    """验证原子删除指定文档及其切片，SQLite 计数与向量存储同步减少。"""
    kb_name = "kb_doc_mgmt"
    rag_create_kb(name=kb_name, description="文档管理测试知识库")

    # 1. 摄取纯文本与本地文件
    res_text = rag_index_text(
        kb_name=kb_name,
        title="doc_python",
        text="Python 是一种优雅的编程语言，广泛应用于数据科学与人工智能领域。",
    )
    assert res_text["success"] is True

    test_file = tmp_path / "k8s_guide.md"
    test_file.write_text(
        "# Kubernetes 指南\n\nKubernetes 负责容器编排与集群调度。",
        encoding="utf-8",
    )
    res_file = rag_index_file(
        kb_name=kb_name,
        file_path=str(test_file),
    )
    assert res_file["success"] is True

    # 初始状态检查
    kbs = rag_list_kbs()
    target_kb = next(k for k in kbs if k["name"] == kb_name)
    assert target_kb["document_count"] == 2
    assert target_kb["chunk_count"] >= 2

    # 验证搜索可召回
    search_hits = rag_search(kb_name=kb_name, query="Python 编程语言")
    assert len(search_hits) >= 1
    assert any(h["metadata"].get("doc_id") == "doc_python" for h in search_hits)

    # 2. 执行删除纯文本文档
    del_res = rag_delete_document(kb_name=kb_name, doc_id="doc_python")
    assert del_res["success"] is True
    assert del_res["kb_name"] == kb_name
    assert del_res["deleted_chunks_count"] >= 1

    # 验证计数已同步减少
    kbs_after_text = rag_list_kbs()
    target_kb_after_text = next(k for k in kbs_after_text if k["name"] == kb_name)
    assert target_kb_after_text["document_count"] == 1

    # 验证搜索不再召回已删除的文档
    search_hits_after = rag_search(kb_name=kb_name, query="Python 编程语言")
    assert not any(h["metadata"].get("doc_id") == "doc_python" for h in search_hits_after)

    # 3. 按文件路径删除文件文档
    del_file_res = rag_delete_document(kb_name=kb_name, doc_id=str(test_file))
    assert del_file_res["success"] is True
    assert del_file_res["deleted_chunks_count"] >= 1

    # 验证知识库文档与切片已全部归零
    kbs_final = rag_list_kbs()
    target_kb_final = next(k for k in kbs_final if k["name"] == kb_name)
    assert target_kb_final["document_count"] == 0
    assert target_kb_final["chunk_count"] == 0


def test_rag_delete_document_not_found_and_validation() -> None:
    """验证删除不存在文档或非法知识库时的防御与错误提示。"""
    rag_create_kb(name="kb_empty_del")

    # 1. 知识库不存在
    res_no_kb = rag_delete_document(kb_name="non_existent_kb", doc_id="any_doc")
    assert res_no_kb["success"] is False
    assert "not found" in res_no_kb["message"].lower()

    # 2. 文档不存在
    res_no_doc = rag_delete_document(kb_name="kb_empty_del", doc_id="non_existent_doc")
    assert res_no_doc["success"] is False
    assert "not found" in res_no_doc["message"].lower()

    # 3. 参数为空校验
    res_empty_doc = rag_delete_document(kb_name="kb_empty_del", doc_id="")
    assert res_empty_doc["success"] is False
    assert "不能为空" in res_empty_doc["message"]


def test_cross_kb_search_all_kbs_wildcard() -> None:
    """验证传入 kb_name='*' 时跨所有知识库联合检索并按相似度全局排序。"""
    rag_create_kb(name="kb_ai", description="AI 知识库")
    rag_create_kb(name="kb_devops", description="DevOps 知识库")
    rag_create_kb(name="kb_history", description="历史文化知识库")

    # 分别向三个知识库摄取不同主题文档
    rag_index_text(
        kb_name="kb_ai",
        title="rag_pipeline",
        text="RAG (检索增强生成) 结合向量数据库与大模型实现精准上下文知识问答与推理。",
    )
    rag_index_text(
        kb_name="kb_devops",
        title="cicd_docker",
        text="CI/CD 持续集成流水线使用 Docker 容器构建镜像与自动化部署。",
    )
    rag_index_text(
        kb_name="kb_history",
        title="tang_dynasty",
        text="唐朝是中国历史上繁荣开放的朝代，诗歌文化极度繁荣，李白杜甫等诗人名扬千古。",
    )

    # 跨所有知识库检索关于 RAG 与 向量问答
    hits = rag_search(
        kb_name="*",
        query="RAG 向量检索增强生成模型",
        top_k=2,
    )

    assert len(hits) >= 1
    # 验证第一名必定是来自 kb_ai
    top_hit = hits[0]
    assert top_hit["kb_name"] == "kb_ai"
    assert top_hit["metadata"]["kb_name"] == "kb_ai"
    assert "检索增强生成" in top_hit["text"]

    # 验证所有返回切片都包含 kb_name 标识且分数降序
    for h in hits:
        assert "kb_name" in h
        assert "kb_name" in h["metadata"]
        assert h["kb_name"] == h["metadata"]["kb_name"]
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_cross_kb_search_specified_list() -> None:
    """验证传入知识库名称列表时，仅在指定的知识库范围内召回。"""
    rag_create_kb(name="kb_python", description="Python 库")
    rag_create_kb(name="kb_rust", description="Rust 库")
    rag_create_kb(name="kb_golang", description="Go 库")

    rag_index_text(
        kb_name="kb_python",
        title="doc_py",
        text="Python 使用 asyncio 库和 GIL 全局解释器锁管理线程与并发任务。",
    )
    rag_index_text(
        kb_name="kb_rust",
        title="doc_rs",
        text="Rust 使用所有权系统与无锁并发模型保障内存安全与高并发吞吐。",
    )
    rag_index_text(
        kb_name="kb_golang",
        title="doc_go",
        text="Go 使用 Goroutine 协程与 CSP 通道模型实现轻量级并发处理。",
    )

    # 仅在 Python 与 Rust 两个知识库中检索并发编程
    hits = rag_search(
        kb_name=["kb_python", "kb_rust"],
        query="并发编程模型与内存安全",
        top_k=5,
    )

    assert len(hits) >= 2
    recalled_kbs = {h["kb_name"] for h in hits}
    # 必须包含 Python 或 Rust，但绝不能包含 Golang
    assert "kb_golang" not in recalled_kbs
    assert recalled_kbs.issubset({"kb_python", "kb_rust"})

    # 验证元数据标识
    for h in hits:
        assert h["metadata"]["kb_name"] in {"kb_python", "kb_rust"}


def test_cross_kb_query_with_context() -> None:
    """验证跨知识库上下文注入能格式化输出来源知识库与片段引用。"""
    rag_create_kb(name="kb_sec", description="安全规范")
    rag_create_kb(name="kb_arch", description="架构规范")

    rag_index_text(
        kb_name="kb_sec",
        title="auth_policy",
        text="所有对外暴露的 API 接口必须启用 OAuth2.0 与 JWT 签名校验，严禁明文传输。",
    )
    rag_index_text(
        kb_name="kb_arch",
        title="rpc_policy",
        text="微服务间调用采用 gRPC 协议传输，超时时间必须控制在 3 秒以内。",
    )

    # 跨所有知识库生成上下文提示词
    context_str = rag_query_with_context(
        kb_name="*",
        query="接口安全与通信规范",
        top_k=2,
    )

    assert "根据以下检索到的相关知识库参考资料回答问题：" in context_str
    assert "参考片段 [1]" in context_str
    # 验证来源中包含知识库标识
    assert "kb_sec" in context_str or "kb_arch" in context_str


def test_cross_kb_search_empty_and_nonexistent_cases() -> None:
    """验证空列表、不存在的知识库集合时的空安全返回。"""
    rag_create_kb(name="kb_solo")

    # 1. 列表全为不存在的知识库
    hits_none = rag_search(
        kb_name=["non_existent_1", "non_existent_2"],
        query="测试查询",
    )
    assert hits_none == []

    # 2. 空列表
    hits_empty = rag_search(
        kb_name=[],
        query="测试查询",
    )
    assert hits_empty == []

    # 3. 空知识库通配符
    rag_clear_kb("kb_solo")
    hits_wildcard_empty = rag_search(
        kb_name="*",
        query="任意查询",
    )
    assert hits_wildcard_empty == []
