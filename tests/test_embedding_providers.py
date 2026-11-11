"""
端到端与接缝测试：通用 OpenAI 兼容 Embedding 提供方、自动维度探针与跨模型拦截 (Issue #12)。

在最高接缝（FastMCP 工具执行边界与最外层 httpx HTTP 传输替身）验证：
1. OpenAI / 通义千问 (DashScope) / DeepSeek / Ollama 兼容协议接入与默认地址回退；
2. 无须手配维度，通过探针自动探测并缓存真实向量维度；
3. 可选 MRL 降维参数透传与向量存储适配器维度对齐；
4. 跨模型写入历史知识库（文本/文件/目录/URL）时的 Fast Fail 防污染硬拦截。

@author Ateng
@since 2026-11-11
"""

import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Generator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from mcp_server_rag.server import (
    rag_create_kb,
    rag_doctor,
    rag_index_directory,
    rag_index_file,
    rag_index_text,
    rag_index_url,
    rag_search,
)


@pytest.fixture(autouse=True)
def isolated_env() -> Generator[str, None, None]:
    """为每个测试用例提供隔离的临时数据目录并恢复全部 Embedding 相关环境变量。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_embed_")
    env_keys = [
        "MCP_RAG_DATA_DIR",
        "MCP_RAG_VECTOR_STORE",
        "MCP_RAG_EMBEDDING_PROVIDER",
        "MCP_RAG_EMBEDDING_BASE_URL",
        "MCP_RAG_EMBEDDING_API_KEY",
        "MCP_RAG_EMBEDDING_MODEL",
        "MCP_RAG_EMBEDDING_DIMENSION",
    ]
    backup = {k: os.environ.get(k) for k in env_keys}
    os.environ["MCP_RAG_DATA_DIR"] = temp_dir
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    yield temp_dir

    for k, v in backup.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    shutil.rmtree(temp_dir, ignore_errors=True)


_REAL_HTTPX_CLIENT = httpx.Client


def _patch_httpx_mock(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    """在最外层 httpx 传输边界注入 MockTransport 替身，保留真实 HTTP 请求组装与解析路径。"""
    mock_transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *args, **kwargs: _REAL_HTTPX_CLIENT(transport=mock_transport, **kwargs),
    )


def _make_fixed_dim_handler(
    dim: int,
    fill_val: float = 0.01,
    captured: list[dict[str, Any]] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """构造返回固定维度嵌入向量的标准 OpenAI 协议响应处理器。"""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if captured is not None:
            captured.append(
                {
                    "url": str(request.url),
                    "auth": request.headers.get("authorization", ""),
                    "headers": dict(request.headers),
                    "body": body,
                }
            )
        inputs = body["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        data = [
            {"object": "embedding", "index": i, "embedding": [fill_val] * dim}
            for i in range(len(inputs))
        ]
        return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"]})

    return handler


def test_qwen_openai_compatible_auto_dimension_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证通义千问 (qwen) 协议接入、默认 DashScope 端点回退及探针自动识别 1024 维。"""
    captured_requests: list[dict[str, Any]] = []
    _patch_httpx_mock(monkeypatch, _make_fixed_dim_handler(1024, captured=captured_requests))

    os.environ["MCP_RAG_EMBEDDING_PROVIDER"] = "qwen"
    os.environ["MCP_RAG_EMBEDDING_API_KEY"] = "sk-qwen-mock-key"
    os.environ["MCP_RAG_EMBEDDING_MODEL"] = "text-embedding-v3"
    os.environ.pop("MCP_RAG_EMBEDDING_BASE_URL", None)
    os.environ.pop("MCP_RAG_EMBEDDING_DIMENSION", None)

    report = rag_doctor()

    assert report["embedding_provider"] == "qwen"
    assert report["embedding_model"] == "text-embedding-v3"
    assert report["embedding_dimension"] == 1024

    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    assert req["auth"] == "Bearer sk-qwen-mock-key"
    assert req["body"]["model"] == "text-embedding-v3"
    assert "dimensions" not in req["body"]


def test_openai_mrl_dimension_override_and_e2e_chroma_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证显式 MRL 维度覆写透传 dimensions 参数、跳过探针并完整写入与检索 ChromaDB。"""
    captured_bodies: list[dict[str, Any]] = []
    captured_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_urls.append(str(request.url))
        body = json.loads(request.content.decode("utf-8"))
        captured_bodies.append(body)
        inputs = body["input"]
        if isinstance(inputs, str):
            inputs = [inputs]

        data = []
        for idx, text in enumerate(inputs):
            vec = [0.0] * 256
            if "Kubernetes" in text or "容器" in text:
                vec[0] = 0.9
                vec[1] = 0.1
            else:
                vec[2] = 0.9
                vec[3] = 0.1
            data.append({"object": "embedding", "index": idx, "embedding": vec})
        return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"]})

    _patch_httpx_mock(monkeypatch, handler)

    os.environ["MCP_RAG_EMBEDDING_PROVIDER"] = "openai"
    os.environ["MCP_RAG_EMBEDDING_BASE_URL"] = "https://custom-proxy.example.com/v1/"
    os.environ["MCP_RAG_EMBEDDING_API_KEY"] = "sk-openai-mrl-test"
    os.environ["MCP_RAG_EMBEDDING_MODEL"] = "text-embedding-3-small"
    os.environ["MCP_RAG_EMBEDDING_DIMENSION"] = "256"

    res_kb = rag_create_kb(name="kb_mrl_256", description="256维降维测试库")
    assert res_kb["success"] is True

    res_idx1 = rag_index_text(
        kb_name="kb_mrl_256",
        title="k8s_doc",
        text="Kubernetes 提供容器编排与自动化部署能力。",
    )
    res_idx2 = rag_index_text(
        kb_name="kb_mrl_256",
        title="cooking_doc",
        text="西红柿炒鸡蛋是一道经典快手家常菜。",
    )
    assert res_idx1["success"] is True
    assert res_idx2["success"] is True

    hits = rag_search(kb_name="kb_mrl_256", query="容器集群", top_k=2)
    assert len(hits) == 2
    assert hits[0]["metadata"]["title"] == "k8s_doc"

    assert all(u == "https://custom-proxy.example.com/v1/embeddings" for u in captured_urls)
    # 不应触发 dimension_probe 请求（仅两次文本入库 + 一次检索查询）
    assert len(captured_bodies) == 3
    for body in captured_bodies:
        assert body["dimensions"] == 256


def test_ollama_and_deepseek_default_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 ollama 免密端点与 deepseek 默认端点自动映射及各自维度自动探测。"""
    captured_ollama: list[dict[str, Any]] = []
    _patch_httpx_mock(monkeypatch, _make_fixed_dim_handler(768, captured=captured_ollama))

    os.environ["MCP_RAG_EMBEDDING_PROVIDER"] = "ollama"
    os.environ["MCP_RAG_EMBEDDING_MODEL"] = "nomic-embed-text"
    os.environ.pop("MCP_RAG_EMBEDDING_BASE_URL", None)
    os.environ.pop("MCP_RAG_EMBEDDING_API_KEY", None)
    os.environ.pop("MCP_RAG_EMBEDDING_DIMENSION", None)

    report = rag_doctor()
    assert report["embedding_provider"] == "ollama"
    assert report["embedding_model"] == "nomic-embed-text"
    assert report["embedding_dimension"] == 768
    assert captured_ollama[0]["url"] == "http://localhost:11434/v1/embeddings"
    assert "authorization" not in {k.lower() for k in captured_ollama[0]["headers"]}

    captured_ds: list[dict[str, Any]] = []
    _patch_httpx_mock(monkeypatch, _make_fixed_dim_handler(1536, captured=captured_ds))
    os.environ["MCP_RAG_EMBEDDING_PROVIDER"] = "deepseek"
    os.environ["MCP_RAG_EMBEDDING_API_KEY"] = "sk-deepseek-key"
    os.environ["MCP_RAG_EMBEDDING_MODEL"] = "deepseek-embed-v1"

    ds_report = rag_doctor()
    assert ds_report["embedding_provider"] == "deepseek"
    assert ds_report["embedding_dimension"] == 1536
    assert captured_ds[0]["url"] == "https://api.deepseek.com/v1/embeddings"


def test_milvus_collection_inherits_auto_probed_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Milvus 在未手配维度时自动从 Embedding 探针继承 1024 维建表，而非硬编码 512 维。"""
    _patch_httpx_mock(monkeypatch, _make_fixed_dim_handler(1024, fill_val=0.02))

    mock_milvus_client = MagicMock()
    mock_milvus_client.has_collection.return_value = False
    fake_pymilvus = ModuleType("pymilvus")
    fake_pymilvus.MilvusClient = MagicMock(return_value=mock_milvus_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pymilvus", fake_pymilvus)

    os.environ["MCP_RAG_VECTOR_STORE"] = "milvus"
    os.environ["MCP_RAG_EMBEDDING_PROVIDER"] = "qwen"
    os.environ["MCP_RAG_EMBEDDING_API_KEY"] = "sk-milvus-1024"
    os.environ["MCP_RAG_EMBEDDING_MODEL"] = "text-embedding-v3"
    os.environ.pop("MCP_RAG_EMBEDDING_DIMENSION", None)

    res = rag_create_kb(name="kb_milvus_1024", description="1024维Milvus测试")
    assert res["success"] is True

    mock_milvus_client.create_collection.assert_called_once()
    call_kwargs = mock_milvus_client.create_collection.call_args.kwargs
    assert call_kwargs["dimension"] == 1024

    idx_res = rag_index_text(
        kb_name="kb_milvus_1024",
        title="milvus_dim_doc",
        text="验证 Milvus 插入向量的真实维度匹配。",
    )
    assert idx_res["success"] is True
    inserted_rows = mock_milvus_client.insert.call_args.kwargs["data"]
    assert len(inserted_rows[0]["vector"]) == 1024


def test_cross_model_write_fast_fail_guard(
    monkeypatch: pytest.MonkeyPatch,
    isolated_env: str,
) -> None:
    """验证向已有历史数据的知识库写入不同模型数据时全部写接口均 Fast Fail 拦截。"""
    embed_call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal embed_call_count
        body = json.loads(request.content.decode("utf-8"))
        inputs = body["input"]
        if isinstance(inputs, str):
            inputs = [inputs]
        if inputs != ["dimension_probe"]:
            embed_call_count += 1
        dim = 512 if body["model"] == "model-v1" else 1024
        data = [
            {"object": "embedding", "index": i, "embedding": [0.03] * dim}
            for i in range(len(inputs))
        ]
        return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"]})

    _patch_httpx_mock(monkeypatch, handler)

    # 1. 在 model-v1 (512 维) 下创建已存量数据的知识库与一个空知识库
    os.environ["MCP_RAG_EMBEDDING_PROVIDER"] = "openai"
    os.environ["MCP_RAG_EMBEDDING_API_KEY"] = "sk-guard-v1"
    os.environ["MCP_RAG_EMBEDDING_MODEL"] = "model-v1"
    os.environ.pop("MCP_RAG_EMBEDDING_DIMENSION", None)

    rag_create_kb(name="kb_locked", description="已写入历史数据的库")
    rag_create_kb(name="kb_empty", description="纯空知识库")
    idx_res = rag_index_text(
        kb_name="kb_locked",
        title="doc1",
        text="这是 model-v1 写入的初始文档。",
    )
    assert idx_res["success"] is True

    # 2. 切换至 model-v2 (1024 维)
    os.environ["MCP_RAG_EMBEDDING_PROVIDER"] = "qwen"
    os.environ["MCP_RAG_EMBEDDING_API_KEY"] = "sk-guard-v2"
    os.environ["MCP_RAG_EMBEDDING_MODEL"] = "text-embedding-v3"

    docs_dir = Path(isolated_env) / "batch_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    sample_file = docs_dir / "extra.md"
    sample_file.write_text("# 新增章节\n跨模型尝试写入的内容。", encoding="utf-8")

    count_before_block = embed_call_count

    # 3. 对已有历史数据的 kb_locked 执行全部写入口均应被 Fast Fail 硬拦截
    res_text = rag_index_text(kb_name="kb_locked", title="doc2", text="新文本")
    assert res_text["success"] is False
    assert "model-v1" in res_text["message"]
    assert "text-embedding-v3" in res_text["message"]

    res_file = rag_index_file(kb_name="kb_locked", file_path=str(sample_file))
    assert res_file["success"] is False
    assert "model-v1" in res_file["message"]

    res_dir = rag_index_directory(kb_name="kb_locked", dir_path=str(docs_dir))
    assert res_dir["success"] is False
    assert "model-v1" in res_dir["message"]

    res_url = rag_index_url(kb_name="kb_locked", url="https://example.com/cross-model")
    assert res_url["success"] is False
    assert "model-v1" in res_url["message"]

    # 确认在校验阶段立即返回，未消耗多余的远端向量计算请求
    assert embed_call_count == count_before_block

    # 4. 纯空知识库在尚未写入数据时切换模型，允许接管更新模型绑定并正常写入
    res_empty = rag_index_text(kb_name="kb_empty", title="doc_empty", text="空库切换后写入")
    assert res_empty["success"] is True


def test_embedding_error_degradation_and_dimension_mismatch_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证远端 HTTP 鉴权异常时 rag_doctor 优雅降级，以及输出维度与显式配置冲突时报错。"""

    def handler_401(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})

    _patch_httpx_mock(monkeypatch, handler_401)

    os.environ["MCP_RAG_EMBEDDING_PROVIDER"] = "openai"
    os.environ["MCP_RAG_EMBEDDING_API_KEY"] = "sk-invalid-key"
    os.environ["MCP_RAG_EMBEDDING_MODEL"] = "text-embedding-3-small"
    os.environ.pop("MCP_RAG_EMBEDDING_DIMENSION", None)

    report = rag_doctor()
    assert report["status"] == "degraded"
    assert report["embedding_status"] == "error"
    assert report["embedding_dimension"] is None
    assert "401" in report["embedding_error"]

    # 校验显式指定 256 维但模型忽略参数并返回 1536 维时主动报错拦截
    _patch_httpx_mock(monkeypatch, _make_fixed_dim_handler(1536))
    os.environ["MCP_RAG_EMBEDDING_DIMENSION"] = "256"
    os.environ["MCP_RAG_EMBEDDING_API_KEY"] = "sk-wrong-dim"

    rag_create_kb(name="kb_dim_mismatch", description="维度不符校验")
    with pytest.raises(ValueError, match="256"):
        rag_index_text(kb_name="kb_dim_mismatch", title="mismatch_doc", text="触发维度校验")


def test_fastembed_default_dimension_empty_env_and_unsupported_provider() -> None:
    """验证默认与空环境变量回退至 FastEmbed (512 维)，且非法 provider 给出明确异常提示。"""
    os.environ["MCP_RAG_EMBEDDING_PROVIDER"] = ""
    os.environ["MCP_RAG_EMBEDDING_MODEL"] = ""
    os.environ.pop("MCP_RAG_EMBEDDING_DIMENSION", None)

    report = rag_doctor()
    assert report["embedding_provider"] == "fastembed"
    assert report["embedding_model"] == "BAAI/bge-small-zh-v1.5"
    assert report["embedding_status"] == "ok"
    assert report["embedding_dimension"] == 512

    os.environ["MCP_RAG_EMBEDDING_PROVIDER"] = "unknown_vendor"
    bad_report = rag_doctor()
    assert bad_report["status"] == "degraded"
    assert bad_report["embedding_status"] == "error"
    assert "unknown_vendor" in bad_report["embedding_error"]
