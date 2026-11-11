"""
端到端与接缝测试：团队图片多端共享网关 (MCP Resources + S3/MinIO) 与外链安全归档 (Issue #18)。

在最高接缝（FastMCP Resource 协议边界、工具调用边界、MockTransport 与 Fake S3 边界）验证：
1. FastMCP Resource Gateway（`rag://assets/{kb_name}/{doc_id}/{filename}`）列举与读取
   返回准确二进制流、MIME 类型与缺失/越权拦截；
2. 启用 MCP_RAG_READ_ONLY=true 时，写工具物理隐藏而 MCP Resource 图片读取保持可用；
3. `[s3]` 可选 extras 声明及缺失安装提示；MCP_RAG_STORAGE_TYPE=s3 自动上传相对图与内嵌图、
   转写切片与 `images` 为 HTTPS 直链、支持 S3 读回、异常回滚与级联删除；
4. 默认 MCP_RAG_ARCHIVE_REMOTE_IMAGES=false 零外网请求保证（远程 URL 原样保留）；
5. 开启 MCP_RAG_ARCHIVE_REMOTE_IMAGES=true 时强制拦截非 HTTP/HTTPS、环回/私有/链路本地 IP、
   DNS 重绑定与 302 重定向 SSRF，带超时与体积上限归档外链并转写引用。

@author Ateng
@since 2026-11-11
"""

import io
import os
import shutil
import socket
import sys
import tempfile
import zipfile
from collections.abc import Generator
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError
from mcp.types import BlobResourceContents, ReadResourceRequestParams
from PIL import Image

from mcp_server_rag.assets import make_asset_doc_key
from mcp_server_rag.config import get_settings
from mcp_server_rag.metadata import get_metadata_catalog
from mcp_server_rag.remote_images import (
    SSRFBlockedError,
    download_safe_remote_image,
    validate_remote_image_url,
)
from mcp_server_rag.s3_storage import S3AssetStorageAdapter
from mcp_server_rag.server import (
    create_server,
    rag_clear_kb,
    rag_create_kb,
    rag_delete_document,
    rag_delete_kb,
    rag_index_file,
    rag_index_text,
    rag_index_url,
    rag_search,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_png_bytes(
    width: int = 160,
    height: int = 160,
    color: tuple[int, int, int] = (20, 90, 180),
) -> bytes:
    """生成真实有效的 PNG 测试图片，使用高熵像素满足二进制文档 5KB 噪点门禁校验。"""
    _ = color
    raw = os.urandom(width * height * 3)
    img = Image.frombytes("RGB", (width, height), raw)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(width: int = 160, height: int = 160) -> bytes:
    """生成真实有效的 JPEG 测试图片以验证多 MIME 类型动态分发。"""
    raw = os.urandom(width * height * 3)
    img = Image.frombytes("RGB", (width, height), raw)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _build_synthetic_docx_with_image(
    target_path: Path,
    heading: str,
    body_text: str,
    image_bytes: bytes,
    alt_text: str = "架构图",
    image_name: str = "image1.png",
) -> None:
    """构建带内嵌插图的最小合法 .docx 压缩包用于端到端提取测试。"""
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        "<w:body>"
        f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>{heading}</w:t></w:r></w:p>'
        f"<w:p><w:r><w:t>{body_text}</w:t></w:r></w:p>"
        "<w:p><w:r><w:drawing><wp:inline>"
        f'<wp:docPr id="1" name="Picture 1" descr="{alt_text}"/>'
        "<a:graphic><a:graphicData><pic:pic>"
        '<pic:blipFill><a:blip r:embed="rIdImg1"/></pic:blipFill>'
        "</pic:pic></a:graphicData></a:graphic>"
        "</wp:inline></w:drawing></w:r></w:p>"
        "<w:sectPr/></w:body></w:document>"
    )
    img_rel_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    doc_rel_type = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rIdImg1" Type="{img_rel_type}" Target="media/{image_name}"/>'
        "</Relationships>"
    )
    rel_ct = "application/vnd.openxmlformats-package.relationships+xml"
    docx_main_ct = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        f'<Default Extension="rels" ContentType="{rel_ct}"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Default Extension="jpg" ContentType="image/jpeg"/>'
        f'<Override PartName="/word/document.xml" ContentType="{docx_main_ct}"/>'
        "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{doc_rel_type}" Target="word/document.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(target_path, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("word/document.xml", doc_xml)
        zf.writestr("word/_rels/document.xml.rels", rels_xml)
        zf.writestr(f"word/media/{image_name}", image_bytes)


class FakeS3Body:
    """模拟 boto3 StreamingBody 响应体。"""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.closed = False

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        self.closed = True


class FakeS3Client:
    """内存 Fake S3 客户端桩，记录 put/get/delete 调用以校验边界行为。"""

    def __init__(self, fail_on_key_substring: str = "") -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.put_calls: list[tuple[str, str, str]] = []
        self.get_calls: list[tuple[str, str]] = []
        self.delete_calls: list[tuple[str, str]] = []
        self._fail_on_key_substring = fail_on_key_substring

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        bucket = str(kwargs["Bucket"])
        key = str(kwargs["Key"])
        body = bytes(kwargs["Body"])
        content_type = str(kwargs.get("ContentType", "application/octet-stream"))
        if self._fail_on_key_substring and self._fail_on_key_substring in key:
            raise RuntimeError(f"Simulated S3 put_object failure for {key}")
        self.put_calls.append((bucket, key, content_type))
        self.objects[(bucket, key)] = {"Body": body, "ContentType": content_type}
        return {"ETag": '"fake-etag"'}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        bucket = str(kwargs["Bucket"])
        key = str(kwargs["Key"])
        self.get_calls.append((bucket, key))
        obj = self.objects.get((bucket, key))
        if obj is None:
            raise KeyError(f"NoSuchKey: {bucket}/{key}")
        return {"Body": FakeS3Body(obj["Body"]), "ContentType": obj["ContentType"]}

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        bucket = str(kwargs["Bucket"])
        key = str(kwargs["Key"])
        self.delete_calls.append((bucket, key))
        self.objects.pop((bucket, key), None)
        return {}


def _install_fake_boto3(monkeypatch: pytest.MonkeyPatch, fake_s3: FakeS3Client) -> list[Any]:
    """向 sys.modules 注入假 boto3 并记录 client 初始化参数。"""
    fake_boto3 = ModuleType("boto3")
    captured_client_kwargs: list[dict[str, Any]] = []

    def _fake_client(service_name: str, **kwargs: Any) -> FakeS3Client:
        assert service_name == "s3"
        captured_client_kwargs.append(kwargs)
        return fake_s3

    fake_boto3.client = _fake_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    return captured_client_kwargs


@pytest.fixture(autouse=True)
def isolated_gateway_env() -> Generator[Path, None, None]:
    """隔离全部网关/S3/只读环境变量及临时目录。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_gw_")
    env_keys = [
        "MCP_RAG_DATA_DIR",
        "MCP_RAG_READ_ONLY",
        "MCP_RAG_STORAGE_TYPE",
        "MCP_RAG_S3_BUCKET",
        "MCP_RAG_S3_ENDPOINT",
        "MCP_RAG_S3_REGION",
        "MCP_RAG_S3_ACCESS_KEY",
        "MCP_RAG_S3_SECRET_KEY",
        "MCP_RAG_S3_PREFIX",
        "MCP_RAG_S3_PUBLIC_BASE_URL",
        "MCP_RAG_S3_TIMEOUT_SECONDS",
        "MCP_RAG_ARCHIVE_REMOTE_IMAGES",
        "MCP_RAG_REMOTE_IMAGE_TIMEOUT_SECONDS",
        "MCP_RAG_REMOTE_IMAGE_MAX_BYTES",
    ]
    backup = {k: os.environ.get(k) for k in env_keys}
    for k in env_keys:
        os.environ.pop(k, None)
    os.environ["MCP_RAG_DATA_DIR"] = temp_dir
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    yield Path(temp_dir)

    for k, v in backup.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_pyproject_declares_s3_optional_dependency_and_missing_guide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 pyproject.toml 声明 [s3] extras，且缺失 boto3 时给出清晰安装提示。"""
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "s3 = [" in pyproject_text
    assert '"boto3>=' in pyproject_text or "'boto3>=" in pyproject_text

    monkeypatch.setitem(sys.modules, "boto3", None)
    os.environ["MCP_RAG_STORAGE_TYPE"] = "s3"
    os.environ["MCP_RAG_S3_BUCKET"] = "team-rag-assets"
    settings = get_settings()

    with pytest.raises(ImportError) as exc_info:
        S3AssetStorageAdapter.from_settings(settings)
    msg = str(exc_info.value)
    assert "[s3]" in msg
    assert "boto3" in msg


def test_relative_image_traversal_and_size_limits_rejected(
    isolated_gateway_env: Path,
) -> None:
    """验证相对路径图片目录穿越攻击与超大图片被安全过滤。"""
    from mcp_server_rag.asset_gateway import prepare_document_assets

    docs_dir = isolated_gateway_env / "sub"
    docs_dir.mkdir(parents=True)
    outside_secret = isolated_gateway_env / "secret.png"
    outside_secret.write_bytes(_make_png_bytes(64, 64))

    # 1. 尝试目录穿越引用外部图片
    evil_text = "![evil](../secret.png)"
    settings = get_settings()
    prepared = prepare_document_assets(
        settings,
        kb_name="test_kb",
        doc_id="evil_doc",
        text_content=evil_text,
        base_dir=docs_dir,
    )
    assert len(prepared.images) == 0
    assert prepared.processed_text == evil_text

    # 2. 尝试引用超大本地图片（超过 remote_image_max_bytes，设定阈值 2048 bytes）
    huge_img = docs_dir / "huge.png"
    huge_img.write_bytes(_make_png_bytes(64, 64))
    os.environ["MCP_RAG_REMOTE_IMAGE_MAX_BYTES"] = "2048"
    huge_settings = get_settings()
    assert huge_settings.remote_image_max_bytes == 2048
    try:
        prepared_huge = prepare_document_assets(
            huge_settings,
            kb_name="test_kb",
            doc_id="huge_doc",
            text_content="![huge](./huge.png)",
            base_dir=docs_dir,
        )
        assert len(prepared_huge.images) == 0
    finally:
        os.environ.pop("MCP_RAG_REMOTE_IMAGE_MAX_BYTES", None)


@pytest.mark.anyio
async def test_mcp_resource_gateway_list_read_mime_and_read_only_boundary(
    isolated_gateway_env: Path,
) -> None:
    """E2E: FastMCP Resource 列举、动态 MIME 读取与 read_only=true 可用性。"""
    assert rag_create_kb(name="gw_kb", description="网关测试")["success"] is True

    # 1. 准备带局部相对 PNG 与 JPEG 的 Markdown 及 .html 文件
    docs_dir = isolated_gateway_env / "docs"
    img_dir = docs_dir / "images"
    img_dir.mkdir(parents=True)
    png_bytes = _make_png_bytes(160, 160)
    jpg_bytes = _make_jpeg_bytes(140, 140)
    (img_dir / "arch.png").write_bytes(png_bytes)
    (img_dir / "topo.jpg").write_bytes(jpg_bytes)

    md_file = docs_dir / "design.md"
    md_file.write_text(
        "# 团队协作架构\n\n"
        "核心部署视图：\n\n"
        "![系统架构图](./images/arch.png)\n\n"
        '节点拓扑参考：<img src="images/topo.jpg" alt="集群拓扑大图" />\n',
        encoding="utf-8",
    )
    assert rag_index_file(kb_name="gw_kb", file_path=str(md_file))["success"] is True

    # 同时验证 HTML 文件忽略 <script> 噪声并将正文相对图片转写为 rag://assets
    html_file = docs_dir / "portal.html"
    html_file.write_text(
        "<html><head><title>前端门户</title>"
        '<script>var x="<img src=./images/topo.jpg>";</script></head>'
        "<body><h1>门户部署图</h1>"
        '<p><img src="./images/arch.png" alt="门户拓扑视图" /></p>'
        "</body></html>",
        encoding="utf-8",
    )
    assert rag_index_file(kb_name="gw_kb", file_path=str(html_file))["success"] is True

    doc_key = make_asset_doc_key(str(md_file.resolve()))
    png_uri = f"rag://assets/gw_kb/{doc_key}/arch.png"
    jpg_uri = f"rag://assets/gw_kb/{doc_key}/topo.jpg"

    search_hits = rag_search(kb_name="gw_kb", query="系统架构图", top_k=5)
    assert len(search_hits) >= 1
    all_imgs = [img for hit in search_hits for img in hit["metadata"]["images"]]
    assert png_uri in all_imgs
    assert jpg_uri in all_imgs

    # 2. 检验模板与动态列举、精确二进制及 MIME 类型
    server_rw = create_server(get_settings())
    templates = await server_rw.list_resource_templates()
    assert any(t.uri_template == "rag://assets/{kb_name}/{doc_id}/{filename}" for t in templates)

    listed = await server_rw.list_resources()
    listed_map = {r.uri: r for r in listed}
    assert png_uri in listed_map
    assert jpg_uri in listed_map
    assert listed_map[png_uri].mime_type == "image/png"
    assert listed_map[jpg_uri].mime_type == "image/jpeg"

    png_contents = list(await server_rw.read_resource(png_uri))
    assert len(png_contents) == 1
    assert png_contents[0].content == png_bytes
    assert png_contents[0].mime_type == "image/png"

    jpg_contents = list(await server_rw.read_resource(jpg_uri))
    assert len(jpg_contents) == 1
    assert jpg_contents[0].content == jpg_bytes
    assert jpg_contents[0].mime_type == "image/jpeg"

    # 测试低层 MCP JSON-RPC Resource 读取编码为 base64 BlobResourceContents
    lowlevel_result = await server_rw._handle_read_resource(
        None,  # type: ignore[arg-type]
        ReadResourceRequestParams(uri=png_uri),  # type: ignore[arg-type]
    )
    assert len(lowlevel_result.contents) == 1
    blob_item = lowlevel_result.contents[0]
    assert isinstance(blob_item, BlobResourceContents)
    assert blob_item.mime_type == "image/png"

    # 缺失或路径穿越资源须抛出 ResourceNotFoundError
    with pytest.raises(ResourceNotFoundError):
        await server_rw.read_resource(f"rag://assets/gw_kb/{doc_key}/missing.png")
    with pytest.raises(ResourceNotFoundError):
        await server_rw.read_resource("rag://assets/gw_kb/../secret.png")

    # 3. MCP_RAG_READ_ONLY=true 时写工具隐藏，但 MCP Resource 读取网关保持可用
    os.environ["MCP_RAG_READ_ONLY"] = "true"
    server_ro = create_server(get_settings())
    ro_tool_names = {t.name for t in await server_ro.list_tools()}
    assert "rag_index_file" not in ro_tool_names
    assert "rag_search" in ro_tool_names

    ro_templates = await server_ro.list_resource_templates()
    assert any(t.uri_template == "rag://assets/{kb_name}/{doc_id}/{filename}" for t in ro_templates)
    ro_listed = {r.uri: r for r in await server_ro.list_resources()}
    assert png_uri in ro_listed
    ro_read = list(await server_ro.read_resource(png_uri))
    assert ro_read[0].content == png_bytes
    assert ro_read[0].mime_type == "image/png"


@pytest.mark.anyio
async def test_s3_storage_mode_uploads_markdown_and_docx_images_rewrites_urls_and_cascades(
    isolated_gateway_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E2E: S3 模式上传相对与内嵌图、转写 HTTPS 直链、支持读回/异常回滚与级联删除。"""
    fake_s3 = FakeS3Client()
    _install_fake_boto3(monkeypatch, fake_s3)

    os.environ["MCP_RAG_STORAGE_TYPE"] = "s3"
    os.environ["MCP_RAG_S3_BUCKET"] = "corp-rag"
    os.environ["MCP_RAG_S3_ENDPOINT"] = "https://minio.internal.local:9000"
    os.environ["MCP_RAG_S3_REGION"] = "cn-hangzhou"
    os.environ["MCP_RAG_S3_ACCESS_KEY"] = "test-ak"
    os.environ["MCP_RAG_S3_SECRET_KEY"] = "test-sk"
    os.environ["MCP_RAG_S3_PREFIX"] = "team-assets"
    os.environ["MCP_RAG_S3_PUBLIC_BASE_URL"] = "https://cdn.example.com/corp-rag"

    assert rag_create_kb(name="s3_kb", description="S3 共享模式")["success"] is True

    # 1. Markdown 本地相对路径图片上传与转写
    md_dir = isolated_gateway_env / "s3_docs"
    (md_dir / "pic").mkdir(parents=True)
    rel_png = _make_png_bytes(160, 160)
    (md_dir / "pic" / "deploy.png").write_bytes(rel_png)
    md_doc = md_dir / "release.md"
    md_doc.write_text(
        "# 发布架构规范\n\n详细发布流程图如下：\n\n![灰度发布拓扑图](./pic/deploy.png)\n",
        encoding="utf-8",
    )

    md_res = rag_index_file(kb_name="s3_kb", file_path=str(md_doc))
    assert md_res["success"] is True

    md_doc_key = make_asset_doc_key(str(md_doc.resolve()))
    expected_md_key = f"team-assets/s3_kb/{md_doc_key}/deploy.png"
    expected_md_url = f"https://cdn.example.com/corp-rag/{expected_md_key}"
    assert ("corp-rag", expected_md_key) in fake_s3.objects
    assert fake_s3.objects[("corp-rag", expected_md_key)]["Body"] == rel_png
    assert fake_s3.objects[("corp-rag", expected_md_key)]["ContentType"] == "image/png"
    assert not (isolated_gateway_env / "assets" / "s3_kb" / md_doc_key).exists()

    md_hits = rag_search(kb_name="s3_kb", query="灰度发布拓扑图", top_k=3)
    assert len(md_hits) >= 1
    assert expected_md_url in md_hits[0]["text"]
    assert md_hits[0]["metadata"]["images"] == [expected_md_url]
    assert "rag://assets/" not in md_hits[0]["text"]

    # 2. .docx 内嵌图片上传 S3 并转写切片与 metadata.images 为 HTTPS 直链
    docx_doc = md_dir / "handbook.docx"
    docx_png = _make_png_bytes(180, 140, color=(90, 40, 140))
    _build_synthetic_docx_with_image(
        docx_doc,
        heading="多节点容灾方案",
        body_text="主备切换与流量摘除核心机制如下：",
        image_bytes=docx_png,
        alt_text="双活容灾架构图",
    )
    docx_res = rag_index_file(kb_name="s3_kb", file_path=str(docx_doc))
    assert docx_res["success"] is True

    docx_doc_key = make_asset_doc_key(str(docx_doc.resolve()))
    expected_docx_key = f"team-assets/s3_kb/{docx_doc_key}/image1.png"
    expected_docx_url = f"https://cdn.example.com/corp-rag/{expected_docx_key}"
    assert ("corp-rag", expected_docx_key) in fake_s3.objects

    docx_hits = rag_search(kb_name="s3_kb", query="双活容灾架构图", top_k=3)
    assert len(docx_hits) >= 1
    assert expected_docx_url in docx_hits[0]["text"]
    assert docx_hits[0]["metadata"]["images"] == [expected_docx_url]

    # 3. FastMCP Resource 端点在 S3 模式下同样可穿透读取二进制内容；缺少 [s3] 依赖返回 ResourceError
    server = create_server(get_settings())
    docx_uri = f"rag://assets/s3_kb/{docx_doc_key}/image1.png"
    docx_res_bytes = list(await server.read_resource(docx_uri))
    assert docx_res_bytes[0].content == docx_png
    assert docx_res_bytes[0].mime_type == "image/png"

    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(ResourceError) as missing_s3_err:
        await server.read_resource(docx_uri)
    assert "[s3]" in str(missing_s3_err.value)
    _install_fake_boto3(monkeypatch, fake_s3)

    # 4. 资产登记失败时回滚已上传 S3 对象与文档记录
    catalog = get_metadata_catalog(get_settings())
    orig_register = type(catalog).register_image_assets

    def _boom_register(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("Simulated DB error after S3 upload")

    rollback_doc = md_dir / "rollback.md"
    rollback_doc.write_text(
        "# 回滚测试文档\n\n![回滚校验图](./pic/deploy.png)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(type(catalog), "register_image_assets", _boom_register)
    with pytest.raises(RuntimeError, match="Simulated DB error"):
        rag_index_file(kb_name="s3_kb", file_path=str(rollback_doc))
    monkeypatch.setattr(type(catalog), "register_image_assets", orig_register)

    rb_doc_key = make_asset_doc_key(str(rollback_doc.resolve()))
    rb_s3_key = f"team-assets/s3_kb/{rb_doc_key}/deploy.png"
    assert ("corp-rag", rb_s3_key) not in fake_s3.objects
    assert catalog.get_document_by_id("s3_kb", str(rollback_doc.resolve())) is None

    # 5. 级联删除文档与清库须同步清理 S3 存储桶对象
    del_res = rag_delete_document(kb_name="s3_kb", doc_id=str(md_doc.resolve()))
    assert del_res["success"] is True
    assert ("corp-rag", expected_md_key) not in fake_s3.objects
    assert ("corp-rag", expected_docx_key) in fake_s3.objects

    clear_res = rag_clear_kb(name="s3_kb")
    assert clear_res["success"] is True
    assert ("corp-rag", expected_docx_key) not in fake_s3.objects
    assert rag_delete_kb(name="s3_kb")["success"] is True


def test_remote_images_disabled_by_default_makes_zero_network_calls(
    isolated_gateway_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认 MCP_RAG_ARCHIVE_REMOTE_IMAGES=false 时，零网络请求并保留原 URL。"""
    settings = get_settings()
    assert settings.archive_remote_images is False
    assert rag_create_kb(name="no_fetch_kb")["success"] is True

    called_urls: list[str] = []

    def _forbidden_handler(request: httpx.Request) -> httpx.Response:
        called_urls.append(str(request.url))
        raise AssertionError(f"默认关闭外链归档时禁止发起图片抓取请求: {request.url}")

    mock_transport = httpx.MockTransport(_forbidden_handler)
    orig_client = httpx.Client

    def _patched_client(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = mock_transport
        return orig_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _patched_client)

    remote_url = "https://cdn.example.com/remote-flow.png"
    res = rag_index_text(
        kb_name="no_fetch_kb",
        title="remote_ref_doc",
        text=f"微服务发布链路说明，详见 ![发布流程图]({remote_url})。",
    )
    assert res["success"] is True
    assert called_urls == []

    hits = rag_search(kb_name="no_fetch_kb", query="发布流程图", top_k=3)
    assert len(hits) >= 1
    assert hits[0]["metadata"]["images"] == [remote_url]
    assert remote_url in hits[0]["text"]
    assert get_metadata_catalog(get_settings()).list_image_assets("no_fetch_kb") == []


@pytest.mark.parametrize(
    "bad_url",
    [
        "file:///etc/passwd",
        "ftp://files.example.com/secret.png",
        "gopher://127.0.0.1:6379/_INFO",
        "data:image/png;base64,iVBORw0KGgo=",
        "http://127.0.0.1/admin.png",
        "http://localhost:8080/secret.png",
        "http://sub.localhost/a.png",
        "http://printer.local/status.png",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.8/internal.png",
        "http://172.16.5.4/internal.png",
        "http://192.168.1.100/internal.png",
        "http://100.64.0.1/cgnat.png",
        "http://0.0.0.0/zero.png",
        "http://[::1]/loopback.png",
        "http://[fe80::1]/linklocal.png",
        "http://[fc00::1]/ula.png",
        "http://[::ffff:127.0.0.1]/v4mapped.png",
    ],
)
def test_ssrf_blocks_disallowed_schemes_and_non_global_ips(bad_url: str) -> None:
    """SSRF 校验强制拒绝非 HTTP/HTTPS、环回、私有与链路本地 IP。"""
    with pytest.raises(SSRFBlockedError):
        validate_remote_image_url(
            bad_url,
            dns_resolver=lambda _host, _port: ["93.184.216.34"],
        )


def test_ssrf_blocks_dns_rebinding_and_unsafe_redirects_and_size_limit() -> None:
    """验证 DNS 私网解析、302 跳转内网/file:// 以及超大响应体均被严格拦截。"""
    with pytest.raises(SSRFBlockedError):
        validate_remote_image_url(
            "https://rebind.attacker.example/pic.png",
            dns_resolver=lambda _host, _port: ["10.20.30.40"],
        )

    visited: list[str] = []

    def _redirect_handler(request: httpx.Request) -> httpx.Response:
        visited.append(str(request.url))
        if request.url.host == "public.example.com" and request.url.path == "/to-loopback.png":
            return httpx.Response(302, headers={"Location": "http://127.0.0.1/secret.png"})
        if request.url.host == "public.example.com" and request.url.path == "/to-file.png":
            return httpx.Response(302, headers={"Location": "file:///etc/passwd"})
        if request.url.host == "public.example.com" and request.url.path == "/huge.png":
            return httpx.Response(200, headers={"Content-Type": "image/png"}, content=b"A" * 4096)
        if request.url.host == "public.example.com" and request.url.path == "/html.png":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html"},
                content=b"<html>not an image</html>",
            )
        raise AssertionError(f"不应该实际请求被 SSRF 拦截的地址: {request.url}")

    transport = httpx.MockTransport(_redirect_handler)

    with pytest.raises(SSRFBlockedError):
        download_safe_remote_image(
            "https://public.example.com/to-loopback.png",
            transport=transport,
            dns_resolver=lambda _h, _p: ["93.184.216.34"],
        )
    with pytest.raises(SSRFBlockedError):
        download_safe_remote_image(
            "https://public.example.com/to-file.png",
            transport=transport,
            dns_resolver=lambda _h, _p: ["93.184.216.34"],
        )
    assert all("127.0.0.1" not in u and "file://" not in u for u in visited)

    assert (
        download_safe_remote_image(
            "https://public.example.com/huge.png",
            max_bytes=1024,
            transport=transport,
            dns_resolver=lambda _h, _p: ["93.184.216.34"],
        )
        is None
    )
    assert (
        download_safe_remote_image(
            "https://public.example.com/html.png",
            transport=transport,
            dns_resolver=lambda _h, _p: ["93.184.216.34"],
        )
        is None
    )


@pytest.mark.anyio
async def test_archive_remote_images_enabled_filters_ssrf_and_archives_safe_urls_e2e(
    isolated_gateway_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E2E: 开启外链归档时拦截 SSRF 并将合法外链安全归档至本地网关与 S3 网关。"""
    os.environ["MCP_RAG_ARCHIVE_REMOTE_IMAGES"] = "true"
    os.environ["MCP_RAG_REMOTE_IMAGE_TIMEOUT_SECONDS"] = "5"
    os.environ["MCP_RAG_REMOTE_IMAGE_MAX_BYTES"] = str(2 * 1024 * 1024)

    safe_png = _make_png_bytes(140, 140)
    fetched_urls: list[str] = []

    orig_getaddrinfo = socket.getaddrinfo

    def _fake_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> list[Any]:
        if host in {"cdn.safe.example", "docs.safe.example"}:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        return orig_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)

    def _handler(request: httpx.Request) -> httpx.Response:
        fetched_urls.append(str(request.url))
        if str(request.url) == "https://cdn.safe.example/diagram.png":
            return httpx.Response(200, headers={"Content-Type": "image/png"}, content=safe_png)
        if str(request.url) == "https://docs.safe.example/page":
            html_body = (
                "<html><head><title>远程技术文档</title></head><body>"
                "<h1>云原生日志规范</h1>"
                '<p>安全图：<img src="https://cdn.safe.example/diagram.png" alt="安全架构" /></p>'
                '<p>探测：<img src="http://169.254.169.254/latest/meta-data/" alt="元数据" /></p>'
                "</body></html>"
            )
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html; charset=utf-8"},
                text=html_body,
            )
        raise AssertionError(f"SSRF 防护失效，非法发起外部请求: {request.url}")

    transport = httpx.MockTransport(_handler)
    orig_client = httpx.Client

    def _mock_client(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = transport
        return orig_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _mock_client)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: orig_client(transport=transport).get(url, **kw),
    )

    assert rag_create_kb(name="archive_kb")["success"] is True

    # 1. 本地模式安全归档外链，并阻断同文档内的 file:/// 与私网/环回 IP
    res_text = rag_index_text(
        kb_name="archive_kb",
        title="hybrid_remote_doc",
        text=(
            "安全图片：![公网架构图](https://cdn.safe.example/diagram.png)\n\n"
            "非法本地：![窃密](file:///etc/passwd)\n\n"
            "非法私网：![元数据](http://169.254.169.254/latest/meta-data/)\n\n"
            "非法环回：![环回](http://127.0.0.1:8080/p.png)"
        ),
    )
    assert res_text["success"] is True
    assert fetched_urls == ["https://cdn.safe.example/diagram.png"]

    doc_key = make_asset_doc_key("hybrid_remote_doc")
    expected_local_uri = f"rag://assets/archive_kb/{doc_key}/diagram.png"
    hits = rag_search(kb_name="archive_kb", query="公网架构图", top_k=3)
    assert len(hits) >= 1
    assert expected_local_uri in hits[0]["text"]
    assert expected_local_uri in hits[0]["metadata"]["images"]

    server = create_server(get_settings())
    archived_payload = list(await server.read_resource(expected_local_uri))
    assert archived_payload[0].content == safe_png
    assert archived_payload[0].mime_type == "image/png"

    # 2. 网页摄取 rag_index_url 在开启归档时也会安全抓取合法外链并过滤 SSRF 链接
    url_res = rag_index_url(kb_name="archive_kb", url="https://docs.safe.example/page")
    assert url_res["success"] is True
    url_doc_key = make_asset_doc_key("https://docs.safe.example/page")
    url_asset_uri = f"rag://assets/archive_kb/{url_doc_key}/diagram.png"
    url_read = list(await server.read_resource(url_asset_uri))
    assert url_read[0].content == safe_png
    assert all("169.254.169.254" not in u for u in fetched_urls)

    # 3. 结合 MCP_RAG_STORAGE_TYPE=s3：外链抓取后直接落桶 S3 并转为 CDN HTTPS 直链
    fake_s3 = FakeS3Client()
    _install_fake_boto3(monkeypatch, fake_s3)
    os.environ["MCP_RAG_STORAGE_TYPE"] = "s3"
    os.environ["MCP_RAG_S3_BUCKET"] = "archive-bucket"
    os.environ["MCP_RAG_S3_PREFIX"] = "remote-cache"
    os.environ["MCP_RAG_S3_PUBLIC_BASE_URL"] = "https://img-cdn.example.com"

    assert rag_create_kb(name="archive_s3_kb")["success"] is True
    res_s3 = rag_index_text(
        kb_name="archive_s3_kb",
        title="s3_remote_doc",
        text="混合模式测试：![外链上云架构图](https://cdn.safe.example/diagram.png)",
    )
    assert res_s3["success"] is True
    s3_doc_key = make_asset_doc_key("s3_remote_doc")
    s3_obj_key = f"remote-cache/archive_s3_kb/{s3_doc_key}/diagram.png"
    s3_cdn_url = f"https://img-cdn.example.com/{s3_obj_key}"
    assert ("archive-bucket", s3_obj_key) in fake_s3.objects
    s3_hits = rag_search(kb_name="archive_s3_kb", query="外链上云架构图", top_k=3)
    assert s3_hits[0]["metadata"]["images"] == [s3_cdn_url]
