"""
端到端与接缝测试：图片语法原子保护、Alt 语义注入与邻近上下文展开 (Issue #15)。

在最高接缝（FastMCP 工具协议边界与切片/元数据目录契约）验证：
1. Markdown 与 HTML 切分管道对长图片语法与 URL 的原子保护及 Alt 语义注入；
2. 切片写入时双向链表指针（prev_chunk_id/next_chunk_id）与结构化 images 持久化；
3. rag_search 返回结果的 metadata.images 结构化输出（无图时返回空列表 []）；
4. rag_search 与 rag_query_with_context 支持 expand_context=0/1/2 上下文缝合、
   安全上限控制、images 去重聚合及重叠命中区间防重复拼接。

@author Ateng
@since 2026-11-11
"""

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from mcp_server_rag.parsers import (
    build_embedding_text,
    dispatch_and_split,
    extract_html_content,
    extract_images_from_text,
    split_markdown,
)
from mcp_server_rag.splitter import recursive_split_text


@pytest.fixture(autouse=True)
def isolated_multimodal_env() -> Generator[Path, None, None]:
    """为每个测试用例提供隔离临时目录并在测试后恢复环境变量。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_mm_")
    env_keys = [
        "MCP_RAG_DATA_DIR",
        "MCP_RAG_CHUNK_SIZE",
        "MCP_RAG_CHUNK_OVERLAP",
        "MCP_RAG_EMBEDDING_PROVIDER",
        "MCP_RAG_EMBEDDING_MODEL",
    ]
    backup = {k: os.environ.get(k) for k in env_keys}
    os.environ["MCP_RAG_DATA_DIR"] = temp_dir
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    yield Path(temp_dir)

    for k, v in backup.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_image_atomic_protection_and_alt_semantic_injection() -> None:
    """验证 Markdown/HTML 长图片标签原子不被切碎、URL 提取与 Alt 语义注入。"""
    md_img_url = (
        "https://cdn.example.com/assets/deep/folder/v2/"
        "architecture_diagram_with_query.png?width=1920&height=1080;token=abc_123"
    )
    md_img_tag = f"![分布式链路追踪与熔断降级拓扑架构图]({md_img_url})"
    md_doc = (
        "# 架构说明\n\n"
        + ("第一段前置说明文字用于填充分片长度，介绍微服务集群入口流量控制。" * 3)
        + "\n\n"
        + md_img_tag
        + "\n\n"
        + ("后续段落说明各个节点之间的健康检查心跳机制与自动故障转移流程。" * 3)
    )

    # 即便图片标记长度大于 chunk_size，也不得从括号或空格中间截断
    md_chunks = split_markdown(md_doc, chunk_size=110, chunk_overlap=20)
    assert len(md_chunks) >= 3
    img_holders = [c for c in md_chunks if md_img_url in c]
    assert len(img_holders) >= 1
    for holder in img_holders:
        assert md_img_tag in holder

    # 直接调用递归切分器亦应保护混合空格与分号的 HTML <img> 与 Markdown 图片标签
    html_img_url = "https://static.example.org/dashboards/grafana-overview.png?orgId=1;from=now-6h"
    html_img_tag = f'<img src="{html_img_url}" alt="云原生监控告警实时大盘视图">'
    raw_mix = (
        ("前言铺垫文本用于逼近切分阈值窗口。" * 4)
        + html_img_tag
        + ("尾部补充文本说明告警通知策略。" * 4)
    )
    mix_chunks = recursive_split_text(raw_mix, chunk_size=90, chunk_overlap=15)
    assert any(html_img_tag in chunk for chunk in mix_chunks)

    # HTML 解析提取保留图片语义与 URL，并通过 dispatch_and_split(.html) 完整保留
    html_page = (
        "<html><head><title>监控方案</title><script>var bad='<img>';</script></head>"
        "<body><h1>监控指标体系</h1>"
        f"<p>核心链路可视化如下：{html_img_tag}</p>"
        "<p>当延迟超过 P99 阈值时自动触发企业微信与飞书告警通知。</p>"
        "</body></html>"
    )
    title, cleaned_html = extract_html_content(html_page)
    assert title == "监控方案"
    assert "云原生监控告警实时大盘视图" in cleaned_html
    assert html_img_url in cleaned_html
    assert "var bad=" not in cleaned_html

    html_chunks = dispatch_and_split(
        Path("monitor.html"), html_page, chunk_size=100, chunk_overlap=15
    )
    assert any(html_img_url in c and "云原生监控告警实时大盘视图" in c for c in html_chunks)

    # 验证图片提取去重与 Alt 语义向量文本构建
    dup_sample = f"{md_img_tag}\n\n{html_img_tag}\n\n![重复引用]({md_img_url})"
    extracted = extract_images_from_text(dup_sample)
    assert extracted == [md_img_url, html_img_url]

    emb_text = build_embedding_text(f"架构展示：{md_img_tag}")
    assert "分布式链路追踪与熔断降级拓扑架构图" in emb_text
    assert "width=1920" not in emb_text  # URL 噪点不参与向量语义计算
    assert build_embedding_text("无图纯文本 ![](https://example.com/blank.png)") != ""


class _DeterministicTestEmbedding:
    """确定性关键词向量桩，用于快速验证语义向量文本注入与检索排序。"""

    model_name = "test-deterministic-emb"
    dimension = 8

    def __init__(self) -> None:
        self.recorded_texts: list[str] = []

    def _encode(self, text: str) -> list[float]:
        # 为测试关键语义维度赋予正交主分量，确保语义检索结果完全可预期
        keywords = [
            "架构",
            "熔断",
            "网关",
            "部署",
            "Docker",
            "监控",
            "告警",
            "纯文本",
        ]
        vec = [1.0 if kw in text else 0.05 for kw in keywords]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.recorded_texts.extend(texts)
        return [self._encode(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._encode(query)


def test_rag_search_exposes_structured_images_metadata(
    isolated_multimodal_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证切片持久化链表指针、Alt 语义向量注入及 rag_search 结构化 metadata.images 输出。"""
    from mcp_server_rag.server import (
        _get_services,
        rag_create_kb,
        rag_index_file,
        rag_index_text,
        rag_search,
    )

    monkeypatch.setenv("MCP_RAG_CHUNK_SIZE", "120")
    monkeypatch.setenv("MCP_RAG_CHUNK_OVERLAP", "10")

    spy_emb = _DeterministicTestEmbedding()
    monkeypatch.setattr("mcp_server_rag.server.get_embedding_provider", lambda _s: spy_emb)

    create_res = rag_create_kb(name="mm_kb", description="多模态图片测试库")
    assert create_res["success"] is True

    img_url = "https://cdn.example.com/arch/gateway.png?v=1;token=secret_q_999"
    md_file = isolated_multimodal_env / "gateway_arch.md"
    md_content = (
        "# 微服务网关设计\n\n"
        "第一部分介绍网关路由分发规则与 TLS 卸载配置说明，不包含任何可视化附图。\n\n"
        "## 熔断限流架构\n\n"
        f"下图展示了详细链路：![微服务网关熔断降级架构图]({img_url})\n\n"
        "## 容器化落地\n\n"
        "第三部分说明 Docker 容器健康检查与优雅停机窗口参数。"
    )
    md_file.write_text(md_content, encoding="utf-8")

    idx_res = rag_index_file(kb_name="mm_kb", file_path=str(md_file))
    assert idx_res["success"] is True
    assert idx_res["chunk_count"] >= 3

    # 1. 验证传给嵌入模型的是去除了 URL 噪点的 Alt 语义文本
    assert any("微服务网关熔断降级架构图" in t for t in spy_emb.recorded_texts)
    assert all("token=secret_q_999" not in t for t in spy_emb.recorded_texts)

    # 2. 验证元数据目录持久化了双向链表指针与结构化 images
    catalog, _ = _get_services()
    persisted_chunks = catalog.get_chunks_by_doc("mm_kb", idx_res["doc_id"])
    assert len(persisted_chunks) == idx_res["chunk_count"]
    assert persisted_chunks[0]["prev_chunk_id"] is None
    assert persisted_chunks[-1]["next_chunk_id"] is None
    for i, ch in enumerate(persisted_chunks):
        if i > 0:
            assert ch["prev_chunk_id"] == persisted_chunks[i - 1]["id"]
        if i + 1 < len(persisted_chunks):
            assert ch["next_chunk_id"] == persisted_chunks[i + 1]["id"]
    image_chunks = [ch for ch in persisted_chunks if ch["images"]]
    assert len(image_chunks) == 1
    assert image_chunks[0]["images"] == [img_url]

    # 3. 写入无图纯文本文档并验证检索无图返回 images=[]
    plain_res = rag_index_text(
        kb_name="mm_kb",
        text="纯文本运维手顺说明：每日定时巡检无图表与纯文字记录。",
        title="纯文本手顺",
    )
    assert plain_res["success"] is True
    plain_doc_id = plain_res["doc_id"]

    # 4. 召回带图切片，验证原文完整与 metadata.images 为 list[str]
    img_hits = rag_search(kb_name="mm_kb", query="微服务网关熔断降级架构图", top_k=3)
    matched = [h for h in img_hits if img_url in h["text"]]
    assert len(matched) == 1
    assert f"![微服务网关熔断降级架构图]({img_url})" in matched[0]["text"]
    assert matched[0]["metadata"]["images"] == [img_url]

    # 5. 召回纯文本切片，验证 metadata.images 恒为空列表 []
    plain_hits = rag_search(kb_name="mm_kb", query="纯文本运维手顺", top_k=5)
    plain_matched = [h for h in plain_hits if h["metadata"].get("doc_id") == plain_doc_id]
    assert len(plain_matched) >= 1
    assert plain_matched[0]["metadata"]["images"] == []


class _LinearKeywordEmbedding:
    """支持邻近窗口多片段独立打分的确定性向量桩。"""

    model_name = "test-window-emb"
    dimension = 8

    def _encode(self, text: str) -> list[float]:
        axes = [
            "前置环境准备",
            "核心服务部署",
            "网关路由配置",
            "主从同步验证",
            "容灾备份归档",
            "独立远端审计",
            "HTML监控面板",
        ]
        matched = [1.0 if ax in text else 0.01 for ax in axes]
        vec = [*matched, 1.0 if all(x <= 0.01 for x in matched) else 0.01]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._encode(query)


def test_expand_context_linked_list_stitching_and_dedup(
    isolated_multimodal_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 expand_context=0/1/2/越界值 对双向链表缝合、images 聚合与重叠区间去重的效果。"""
    from mcp_server_rag.server import (
        MAX_EXPAND_CONTEXT,
        rag_create_kb,
        rag_index_file,
        rag_query_with_context,
        rag_search,
    )

    assert MAX_EXPAND_CONTEXT == 5

    # 控制切分为精确 6 个有序章节切片
    monkeypatch.setenv("MCP_RAG_CHUNK_SIZE", "120")
    monkeypatch.setenv("MCP_RAG_CHUNK_OVERLAP", "0")
    monkeypatch.setattr(
        "mcp_server_rag.server.get_embedding_provider",
        lambda _s: _LinearKeywordEmbedding(),
    )

    rag_create_kb(name="ctx_kb", description="上下文扩展验证库")

    pre_img = "https://cdn.example.com/pre-setup.png"
    gw_img = "https://cdn.example.com/gw-topo.png"
    handbook = isolated_multimodal_env / "ops_guide.md"
    handbook.write_text(
        "\n\n".join(
            [
                (
                    "## 一 前置环境准备\n\n"
                    f"确认内核参数与 Docker 版本，参考 ![前置环境准备示意图]({pre_img})。"
                ),
                (
                    "## 二 核心服务部署\n\n"
                    "启动订单引擎核心进程并校验健康探测端口，这里只有纯文字说明。"
                ),
                (
                    "## 三 网关路由配置\n\n"
                    f"配置入口限流策略与灰色发布路由，参考 ![网关路由配置拓扑图]({gw_img})。"
                ),
                "## 四 主从同步验证\n\n检查数据库 Binlog 位点延迟是否归零并验证只读节点。",
                "## 五 容灾备份归档\n\n定期校验冷备份快照完整性并确认恢复演练时间窗口。",
                "## 六 独立远端审计\n\n每月导出访问审计日志存档到合规加密桶。",
            ]
        ),
        encoding="utf-8",
    )

    idx_res = rag_index_file(kb_name="ctx_kb", file_path=str(handbook))
    assert idx_res["success"] is True
    assert idx_res["chunk_count"] == 6

    # 1. expand_context=0：不展开上下文，命中 核心服务部署 纯文本切片时 images 应为 []
    hits_0 = rag_search(
        kb_name="ctx_kb",
        query="核心服务部署",
        top_k=1,
        expand_context=0,
    )
    assert len(hits_0) == 1
    assert "核心服务部署" in hits_0[0]["text"]
    assert "前置环境准备" not in hits_0[0]["text"]
    assert "网关路由配置" not in hits_0[0]["text"]
    assert hits_0[0]["metadata"]["images"] == []

    # 2. expand_context=1：应向外缝合前置与后继邻接块，按阅读顺序排列且聚合去重 images
    hits_1 = rag_search(
        kb_name="ctx_kb",
        query="核心服务部署",
        top_k=1,
        expand_context=1,
    )
    assert len(hits_1) == 1
    stitched_1 = hits_1[0]["text"]
    assert "前置环境准备" in stitched_1
    assert "核心服务部署" in stitched_1
    assert "网关路由配置" in stitched_1
    assert "独立远端审计" not in stitched_1
    # 阅读先后顺序必须严格保持：前置 -> 核心 -> 网关
    assert stitched_1.index("前置环境准备") < stitched_1.index("核心服务部署")
    assert stitched_1.index("核心服务部署") < stitched_1.index("网关路由配置")
    assert hits_1[0]["metadata"]["images"] == [pre_img, gw_img]

    # 3. 同文档双重叠命中自动融合去重，而不相交命中保持独立返回
    hits_overlap = rag_search(
        kb_name="ctx_kb",
        query="核心服务部署 网关路由配置 独立远端审计",
        top_k=3,
        expand_context=1,
    )
    assert len(hits_overlap) == 2  # 重叠前段合为 1 条，远端不相交切片独立保留 1 条
    all_text_joined = "\n---\n".join(h["text"] for h in hits_overlap)
    assert all_text_joined.count("配置入口限流策略与灰色发布路由") == 1
    assert all_text_joined.count("启动订单引擎核心进程") == 1
    assert all_text_joined.count("每月导出访问审计日志") == 1

    # 4. expand_context=2 与越界保护（负数 -> 0，超大值 -> 截断至 MAX_EXPAND_CONTEXT）
    hits_2 = rag_search(kb_name="ctx_kb", query="核心服务部署", top_k=1, expand_context=2)
    assert "主从同步验证" in hits_2[0]["text"]
    assert "前置环境准备" in hits_2[0]["text"]

    hits_neg = rag_search(kb_name="ctx_kb", query="核心服务部署", top_k=1, expand_context=-5)
    assert "前置环境准备" not in hits_neg[0]["text"]

    hits_huge = rag_search(kb_name="ctx_kb", query="核心服务部署", top_k=1, expand_context=99)
    assert "核心服务部署" in hits_huge[0]["text"]
    assert hits_huge[0]["metadata"]["images"] == [pre_img, gw_img]

    # 5. rag_query_with_context 透传 expand_context 并输出包含图片与上下文的 Markdown
    prompt_ctx = rag_query_with_context(
        kb_name="ctx_kb",
        query="核心服务部署",
        top_k=1,
        expand_context=1,
    )
    assert "核心服务部署" in prompt_ctx
    assert f"![前置环境准备示意图]({pre_img})" in prompt_ctx
    assert f"![网关路由配置拓扑图]({gw_img})" in prompt_ctx


def test_html_file_and_url_end_to_end_image_rag(
    isolated_multimodal_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """端到端验证本地 HTML 文件与在线 URL 摄取的图片语义提取及 expand_context 检索。"""
    import httpx

    from mcp_server_rag.server import (
        rag_create_kb,
        rag_index_file,
        rag_index_url,
        rag_search,
    )

    monkeypatch.setenv("MCP_RAG_CHUNK_SIZE", "95")
    monkeypatch.setenv("MCP_RAG_CHUNK_OVERLAP", "0")
    monkeypatch.setattr(
        "mcp_server_rag.server.get_embedding_provider",
        lambda _s: _LinearKeywordEmbedding(),
    )

    rag_create_kb(name="html_kb", description="HTML与URL图文端到端验证")

    local_img = "https://img.example.com/local-panel.png"
    html_file = isolated_multimodal_env / "panel.html"
    html_file.write_text(
        (
            "<html><head><title>可视化指南</title>"
            "<script>var junk='<img src=bad.png>';</script></head>"
            "<body>"
            "<p>第一章介绍前置准备工作，用于测试邻近上下文展开。</p>"
            f'<p><img src="{local_img}" alt="HTML监控面板实时拓扑指标图" /></p>'
            "<p>第三章介绍后续巡检流程与告警关闭方式。</p>"
            "</body></html>"
        ),
        encoding="utf-8",
    )
    file_res = rag_index_file(kb_name="html_kb", file_path=str(html_file))
    assert file_res["success"] is True

    # 模拟在线 URL 网页抓取（返回包含图片的 HTML）
    remote_img = "https://remote.example.org/arch-diagram.webp"

    class _FakeResp:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        text = (
            "<html><head><title>远端文档</title></head><body>"
            f'<p>在线手册参考：<img data-src="{remote_img}" alt="在线 HTML监控面板架构"/></p>'
            "</body></html>"
        )

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: _FakeResp())
    url_res = rag_index_url(kb_name="html_kb", url="https://docs.example.org/guide")
    assert url_res["success"] is True

    hits = rag_search(kb_name="html_kb", query="HTML监控面板", top_k=5, expand_context=1)
    collected_imgs = [img for h in hits for img in h["metadata"]["images"]]
    assert local_img in collected_imgs
    assert remote_img in collected_imgs
    assert "bad.png" not in collected_imgs
    assert any("第一章介绍前置准备工作" in h["text"] for h in hits)


def test_fastmcp_expand_context_schema_and_cycle_safety(
    isolated_multimodal_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 FastMCP 工具 Schema 暴露 expand_context 及异常环形链表/空正文的防御韧性。"""
    import inspect
    import sqlite3
    from contextlib import closing

    from mcp_server_rag.server import (
        _get_services,
        create_server,
        rag_create_kb,
        rag_index_text,
        rag_query_with_context,
        rag_search,
    )

    # 1. 验证 FastMCP 注册成功与工具参数签名包含 expand_context
    mcp_app = create_server()
    assert mcp_app is not None
    assert "expand_context" in inspect.signature(rag_search).parameters
    assert "expand_context" in inspect.signature(rag_query_with_context).parameters

    # 2. 验证畸形环形链表不会死循环且不会重复输出同一图片或切片
    monkeypatch.setenv("MCP_RAG_CHUNK_SIZE", "45")
    monkeypatch.setenv("MCP_RAG_CHUNK_OVERLAP", "0")
    monkeypatch.setattr(
        "mcp_server_rag.server.get_embedding_provider",
        lambda _s: _LinearKeywordEmbedding(),
    )
    rag_create_kb(name="cycle_kb", description="防环测试")
    fig_url = "https://cdn.example.com/cycle.png"
    res = rag_index_text(
        kb_name="cycle_kb",
        title="环形韧性文档",
        text=(
            f"前置环境准备段落第一部分含图 ![前置环境准备图]({fig_url})\n\n"
            "核心服务部署段落第二部分作为检索命中锚点\n\n"
            "网关路由配置段落第三部分作为后续扩展窗口"
        ),
    )
    assert res["success"] is True
    catalog, _ = _get_services()
    assert catalog.get_chunks_by_ids("cycle_kb", []) == []

    doc_chunks = catalog.get_chunks_by_doc("cycle_kb", res["doc_id"])
    assert len(doc_chunks) >= 3
    first_id = doc_chunks[0]["id"]
    core_id = next(ch["id"] for ch in doc_chunks if "核心服务部署段落" in ch["content"])
    last_id = doc_chunks[-1]["id"]

    # 人工破坏 SQLite 中的指针使其成环，并清空命中节点 content 验证回退向量库正文防御
    db_path = isolated_multimodal_env / "metadata.db"
    with closing(sqlite3.connect(db_path, timeout=5.0)) as conn:
        conn.execute("UPDATE chunks SET prev_chunk_id = ? WHERE id = ?", (last_id, first_id))
        conn.execute("UPDATE chunks SET next_chunk_id = ? WHERE id = ?", (first_id, last_id))
        conn.execute("UPDATE chunks SET content = '' WHERE id = ?", (core_id,))
        conn.commit()

    hits = rag_search(kb_name="cycle_kb", query="核心服务部署", top_k=1, expand_context=5)
    assert len(hits) == 1
    assert hits[0]["text"].count("核心服务部署段落") == 1
    assert hits[0]["metadata"]["images"] == [fig_url]
