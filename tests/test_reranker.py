"""
端到端与接缝测试：双模可插拔 Reranker 二次交叉重排 (FastEmbed / 云端 HTTP API) (Issue #19)。

在最高接缝（FastMCP 工具调用边界、混合检索集成点、MockTransport 协议边界）验证：
1. 配置契约支持 MCP_RAG_RERANK_ENABLED（默认 false）、MCP_RAG_RERANK_PROVIDER、
   MCP_RAG_RERANK_MODEL、MCP_RAG_RERANK_BASE_URL、MCP_RAG_RERANK_API_KEY 与
   MCP_RAG_RERANK_TIMEOUT_SECONDS；
2. 默认 MCP_RAG_RERANK_ENABLED=false 时，检索流程完全跳过 Reranker，
   零额外计算开销且不加载交叉编码模型；
3. 本地模式（fastembed）利用 FastEmbed ONNX TextCrossEncoder 对候选集二次评分，
   按相关性得分降序重排并应用 threshold 截断返回 Top-K；
4. 云端模式（api）通过 httpx 携带 Bearer 鉴权与严谨超时调用标准 /v1/rerank 接口，
   正确解析相关性分数完成重排；
5. rag_doctor 诊断报告同步包含当前 Reranker 启用状态、驱动模式与模型名称；
6. 在 FastMCP 工具最高边界通过 httpx MockTransport 与开关对比测试端到端验证重排前后排序翻转生效。

@author Ateng
@since 2026-11-11
"""

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx
import pytest

from mcp_server_rag.config import Settings
from mcp_server_rag.reranker import (
    ApiReranker,
    FastEmbedReranker,
    RerankerProvider,
    RerankResult,
    clear_reranker_cache,
    get_reranker_provider,
)
from mcp_server_rag.server import (
    rag_create_kb,
    rag_doctor,
    rag_index_text,
    rag_search,
)


@pytest.fixture(autouse=True)
def isolated_rerank_env() -> Generator[Path, None, None]:
    """隔离全部 Reranker 环境变量与临时目录。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_rerank_")
    env_keys = [
        "MCP_RAG_DATA_DIR",
        "MCP_RAG_RERANK_ENABLED",
        "MCP_RAG_RERANK_PROVIDER",
        "MCP_RAG_RERANK_MODEL",
        "MCP_RAG_RERANK_BASE_URL",
        "MCP_RAG_RERANK_API_KEY",
        "MCP_RAG_RERANK_TIMEOUT_SECONDS",
    ]
    backup = {k: os.environ.get(k) for k in env_keys}
    for k in env_keys:
        os.environ.pop(k, None)

    os.environ["MCP_RAG_DATA_DIR"] = temp_dir
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    clear_reranker_cache()

    yield Path(temp_dir)

    clear_reranker_cache()
    for k, v in backup.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_reranker_configuration_defaults_and_custom_values() -> None:
    """验证 Reranker 配置项默认值及通过环境变量自定义解析。"""
    default_settings = Settings()
    assert default_settings.rerank_enabled is False
    assert default_settings.rerank_provider == "fastembed"
    assert default_settings.rerank_model == "BAAI/bge-reranker-base"
    assert default_settings.rerank_base_url == ""
    assert default_settings.rerank_api_key == ""
    assert default_settings.rerank_timeout_seconds == 30

    os.environ["MCP_RAG_RERANK_ENABLED"] = "true"
    os.environ["MCP_RAG_RERANK_PROVIDER"] = "api"
    os.environ["MCP_RAG_RERANK_MODEL"] = "BAAI/bge-reranker-v2-m3"
    os.environ["MCP_RAG_RERANK_BASE_URL"] = "https://api.siliconflow.cn/v1"
    os.environ["MCP_RAG_RERANK_API_KEY"] = "sk-test-token"
    os.environ["MCP_RAG_RERANK_TIMEOUT_SECONDS"] = "45"

    custom_settings = Settings()
    assert custom_settings.rerank_enabled is True
    assert custom_settings.rerank_provider == "api"
    assert custom_settings.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert custom_settings.rerank_base_url == "https://api.siliconflow.cn/v1"
    assert custom_settings.rerank_api_key == "sk-test-token"
    assert custom_settings.rerank_timeout_seconds == 45


def test_reranker_factory_validation_and_provider_creation() -> None:
    """验证 Reranker 工厂函数对合法及非法驱动模式的校验行为。"""
    # 1. 默认 fastembed 工厂实例
    settings_fe = Settings(rerank_provider="fastembed")
    provider_fe = get_reranker_provider(settings_fe)
    assert isinstance(provider_fe, FastEmbedReranker)
    assert isinstance(provider_fe, RerankerProvider)
    assert provider_fe.model_name == "BAAI/bge-reranker-base"

    # 2. api 模式缺失 base_url 抛出 ValueError
    settings_api_missing = Settings(rerank_provider="api", rerank_base_url="")
    with pytest.raises(ValueError) as exc_info:
        get_reranker_provider(settings_api_missing)
    assert "MCP_RAG_RERANK_BASE_URL" in str(exc_info.value)

    # 3. api 模式合法参数创建实例
    settings_api = Settings(
        rerank_provider="api",
        rerank_base_url="https://api.siliconflow.cn/v1",
        rerank_api_key="sk-test",
        rerank_model="BAAI/bge-reranker-v2-m3",
        rerank_timeout_seconds=20,
    )
    provider_api = get_reranker_provider(settings_api)
    assert isinstance(provider_api, ApiReranker)
    assert isinstance(provider_api, RerankerProvider)
    assert provider_api.model_name == "BAAI/bge-reranker-v2-m3"

    # 4. 不支持的 Provider 抛出 ValueError
    settings_unknown = Settings(rerank_provider="invalid_provider")
    with pytest.raises(ValueError) as exc_un:
        get_reranker_provider(settings_unknown)
    assert "不支持的 Reranker Provider" in str(exc_un.value)


def test_fastembed_reranker_lazy_load_and_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证本地 FastEmbedReranker 延迟加载、空入参安全及打分排序与 Sigmoid 归一化。"""
    provider = FastEmbedReranker(model_name="BAAI/bge-reranker-base")
    assert provider._model is None  # 验证未调用 rerank 时不加载模型

    # 空文本列表快速返回空列表
    assert provider.rerank("query", []) == []

    # 模拟 FastEmbed TextCrossEncoder 的底层打分输出（包含未归一化 logits）
    class FakeTextCrossEncoder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def rerank(self, query: str, documents: list[str]) -> list[float]:
            assert query == "深度学习"
            # 为输入文档赋予不同的 logits
            score_map = {
                "Doc 0": -2.0,  # 低相关 -> sigmoid ~0.119
                "Doc 1": 4.5,  # 高相关 -> sigmoid ~0.989
                "Doc 2": 1.0,  # 中相关 -> sigmoid ~0.731
            }
            return [score_map.get(d, 0.0) for d in documents]

    monkeypatch.setattr(
        "mcp_server_rag.reranker.fastembed_reranker._load_cross_encoder_class",
        lambda: FakeTextCrossEncoder,
    )

    docs = ["Doc 0", "Doc 1", "Doc 2"]
    results = provider.rerank("深度学习", docs)
    assert provider._model is not None  # 验证首次调用后模型已加载

    assert len(results) == 3
    # 验证按相关性得分降序排列：Doc 1 > Doc 2 > Doc 0
    assert results[0].index == 1
    assert results[0].text == "Doc 1"
    assert 0.98 < results[0].score < 1.0

    assert results[1].index == 2
    assert results[1].text == "Doc 2"
    assert 0.70 < results[1].score < 0.75

    assert results[2].index == 0
    assert results[2].text == "Doc 0"
    assert 0.10 < results[2].score < 0.15

    # 验证 top_n 截断
    top_1 = provider.rerank("深度学习", docs, top_n=1)
    assert len(top_1) == 1
    assert top_1[0].index == 1


def test_api_reranker_payload_headers_mock_transport_and_error_handling() -> None:
    """验证云端 ApiReranker 严格遵循 /v1/rerank 规范、Bearer 鉴权及异常防御。"""
    captured_requests: list[httpx.Request] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        if "error" in str(request.url):
            return httpx.Response(500, text="Internal Server Error")
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.60},
                    {"index": 1, "relevance_score": 0.25},
                ]
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = httpx.Client(transport=transport)

    provider = ApiReranker(
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-secret-token",
        model_name="BAAI/bge-reranker-v2-m3",
        timeout_seconds=15,
        client=client,
    )

    docs = ["文本A", "文本B", "文本C"]
    results = provider.rerank("测试查询", docs)

    # 1. 验证 HTTP 契约
    assert len(captured_requests) == 1
    req = captured_requests[0]
    assert req.url == "https://api.siliconflow.cn/v1/rerank"
    assert req.headers["authorization"] == "Bearer sk-secret-token"
    assert req.headers["content-type"] == "application/json"

    import json

    body = json.loads(req.content.decode("utf-8"))
    assert body["model"] == "BAAI/bge-reranker-v2-m3"
    assert body["query"] == "测试查询"
    assert body["documents"] == docs

    # 2. 验证结果解析与排序
    assert len(results) == 3
    assert results[0].index == 2
    assert results[0].score == 0.95
    assert results[0].text == "文本C"

    assert results[1].index == 0
    assert results[1].score == 0.60

    assert results[2].index == 1
    assert results[2].score == 0.25

    # 3. 验证异常网络响应抛出语义化 RuntimeError
    err_provider = ApiReranker(
        base_url="https://api.siliconflow.cn/error",
        model_name="BAAI/bge-reranker-v2-m3",
        client=client,
    )
    with pytest.raises(RuntimeError) as exc_err:
        err_provider.rerank("测试", docs)
    assert "500" in str(exc_err.value)


def test_rag_doctor_reports_reranker_diagnostics() -> None:
    """验证 rag_doctor 诊断报告在关闭与开启 Reranker 时的字段完整性与状态。"""
    # 1. 默认关闭
    report_default = rag_doctor()
    assert report_default["rerank_enabled"] is False
    assert report_default["rerank_provider"] == "fastembed"
    assert report_default["rerank_model"] == "BAAI/bge-reranker-base"

    # 2. 环境变量显式开启并切换驱动
    os.environ["MCP_RAG_RERANK_ENABLED"] = "true"
    os.environ["MCP_RAG_RERANK_PROVIDER"] = "api"
    os.environ["MCP_RAG_RERANK_MODEL"] = "BAAI/bge-reranker-v2-m3"
    os.environ["MCP_RAG_RERANK_BASE_URL"] = "https://api.example.com"

    report_enabled = rag_doctor()
    assert report_enabled["rerank_enabled"] is True
    assert report_enabled["rerank_provider"] == "api"
    assert report_enabled["rerank_model"] == "BAAI/bge-reranker-v2-m3"


def test_default_rerank_disabled_skips_reranker_with_zero_overhead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 MCP_RAG_RERANK_ENABLED=false 时完全不调用 RerankerFactory 与模型。"""
    assert rag_create_kb(name="kb_disabled", description="默认禁用重排测试")["success"] is True
    rag_index_text(kb_name="kb_disabled", text="Python FastMCP 架构设计与实现", title="d1")

    called_factory = False

    def spy_get_reranker(*args: Any, **kwargs: Any) -> Any:
        nonlocal called_factory
        called_factory = True
        raise AssertionError("在 MCP_RAG_RERANK_ENABLED=false 时绝对不应调用 get_reranker_provider")

    monkeypatch.setattr("mcp_server_rag.server.get_reranker_provider", spy_get_reranker)

    hits = rag_search(kb_name="kb_disabled", query="Python FastMCP", top_k=5)
    assert len(hits) == 1
    assert called_factory is False
    assert "rerank_score" not in (hits[0].get("metadata") or {})


def test_e2e_rank_flip_and_threshold_filtering_with_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """端到端验证开启 Reranker 后对扩大候选集进行交叉重排导致结果翻转并满足 threshold 截断。"""
    assert rag_create_kb(name="kb_rerank", description="端到端重排翻转测试")["success"] is True

    # 准备两篇文档：
    # Doc A: 充满查询关键字 "VectorDatabase"，在第一阶段 BM25/FTS+稠密向量召回中得分极高排第一
    # Doc B: 深度阐述 Milvus 与 Chroma 架构机制，语义匹配度极高但在首阶段倒排评分稍低排第二
    text_a = "VectorDatabase VectorDatabase 基础介绍：向量数据库提供高维数据存储。"
    text_b = "Chroma 与 Milvus 是企业生产环境核心推荐的高性能向量检索引擎解决方案。"

    rag_index_text(kb_name="kb_rerank", text=text_a, title="doc_a")
    rag_index_text(kb_name="kb_rerank", text=text_b, title="doc_b")

    query = "VectorDatabase 基础介绍"

    # 1. 在 Reranker 未开启状态下检索：Doc A 因高频精准匹配排在 Doc B 之前
    hits_before = rag_search(kb_name="kb_rerank", query=query, top_k=2)
    assert len(hits_before) == 2
    assert "VectorDatabase" in hits_before[0]["text"]
    id_rank1_before = hits_before[0]["chunk_id"]

    # 2. 开启 Reranker (API 模式)，注入 MockTransport 模拟交叉编码器判定 Doc B 远比 Doc A 语义契合
    class CustomMockReranker:
        @property
        def model_name(self) -> str:
            return "mock-reranker-large"

        def rerank(
            self,
            q: str,
            documents: list[str],
            top_n: int | None = None,
        ) -> list[RerankResult]:
            # Doc B 文本包含 "生产环境核心推荐"，给予高分 0.95；Doc A 仅简单重复词，给予较低分 0.40
            results: list[RerankResult] = []
            for idx, doc_text in enumerate(documents):
                score = 0.95 if "Chroma 与 Milvus" in doc_text else 0.40
                results.append(RerankResult(index=idx, score=score, text=doc_text))
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:top_n] if top_n is not None else results

    mock_provider = CustomMockReranker()
    monkeypatch.setattr("mcp_server_rag.server.get_reranker_provider", lambda s: mock_provider)

    os.environ["MCP_RAG_RERANK_ENABLED"] = "true"
    os.environ["MCP_RAG_RERANK_PROVIDER"] = "api"
    os.environ["MCP_RAG_RERANK_BASE_URL"] = "https://api.example.com"

    # 执行重排检索
    hits_after = rag_search(kb_name="kb_rerank", query=query, top_k=2)
    assert len(hits_after) == 2

    # 验证发生排序翻转：Doc B 跃居第一位！
    assert hits_after[0]["chunk_id"] != id_rank1_before
    assert "Chroma 与 Milvus" in hits_after[0]["text"]
    assert hits_after[0]["score"] == 0.95
    assert hits_after[0]["metadata"]["rerank_score"] == 0.95
    assert hits_after[0]["metadata"]["rerank_model"] == "mock-reranker-large"

    assert "VectorDatabase VectorDatabase" in hits_after[1]["text"]
    assert hits_after[1]["score"] == 0.40

    # 3. 验证 threshold 阈值截断（设定 threshold=0.70，应截断仅保留 Doc B）
    hits_thresholded = rag_search(kb_name="kb_rerank", query=query, top_k=2, threshold=0.70)
    assert len(hits_thresholded) == 1
    assert "Chroma 与 Milvus" in hits_thresholded[0]["text"]
