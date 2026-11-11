"""
端到端与接缝测试：轻量 PDF/Word 文档解析、图片噪点门禁与本地资产打桩 (Issue #16)。

在最高接缝（FastMCP 工具协议边界与资产/解析模块契约）验证：
1. [office] 可选依赖组声明及 pypdf / python-docx 缺失时的友好安装引导；
2. Image Noise Filter 内存门禁对微型图元（< 100px 或 < 5KB）的精准过滤；
3. 真实合成 .docx 与 .pdf 夹具经 rag_index_file / rag_index_directory 摄取、
   就地打入 rag://assets/... 锚点、SHA-256 幂等跳过与原子更新；
4. rag_search 召回与 metadata.images 结构化透出，以及删文档/清库/删库物理资产级联清理。

@author Ateng
@since 2026-11-11
"""

import hashlib
import io
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from PIL import Image

from mcp_server_rag.config import get_settings
from mcp_server_rag.metadata import get_metadata_catalog
from mcp_server_rag.server import (
    rag_clear_kb,
    rag_create_kb,
    rag_delete_document,
    rag_delete_kb,
    rag_index_directory,
    rag_index_file,
    rag_search,
)


@pytest.fixture(autouse=True)
def isolated_office_env() -> Generator[Path, None, None]:
    """为每个测试用例提供隔离临时目录并在测试后恢复环境变量。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_office_")
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


def test_office_extra_declared_and_missing_dependency_errors(
    isolated_office_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 [office] 可选依赖声明，及缺失依赖时抛出友好指引且不阻断文本文件与目录其余文档。"""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    office_match = re.search(
        r"^office\s*=\s*\[(?P<deps>[^\]]+)\]",
        pyproject_text,
        re.MULTILINE,
    )
    assert office_match is not None
    office_specs = office_match.group("deps")
    assert "pypdf" in office_specs
    assert "python-docx" in office_specs

    assert rag_create_kb(name="office_dep_kb", description="可选依赖检查库")["success"] is True

    docs_dir = isolated_office_env / "mixed_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    pdf_file = docs_dir / "guide.pdf"
    docx_file = docs_dir / "spec.docx"
    md_file = docs_dir / "readme.md"
    pdf_file.write_bytes(b"%PDF-1.4\n%stub\n")
    docx_file.write_bytes(b"PK\x03\x04stub")
    md_file.write_text("# 文本说明\n\n纯文本 Markdown 文档应始终正常索引。", encoding="utf-8")

    monkeypatch.setitem(sys.modules, "pypdf", None)
    monkeypatch.setitem(sys.modules, "docx", None)

    with pytest.raises(ImportError) as pdf_exc:
        rag_index_file(kb_name="office_dep_kb", file_path=str(pdf_file))
    pdf_msg = str(pdf_exc.value)
    assert "atengk-mcp-server-rag[office]" in pdf_msg
    assert "pypdf" in pdf_msg

    with pytest.raises(ImportError) as docx_exc:
        rag_index_file(kb_name="office_dep_kb", file_path=str(docx_file))
    docx_msg = str(docx_exc.value)
    assert "atengk-mcp-server-rag[office]" in docx_msg
    assert "python-docx" in docx_msg

    # 文本格式与目录中的文本文件不受 [office] 缺失影响
    md_res = rag_index_file(kb_name="office_dep_kb", file_path=str(md_file))
    assert md_res["success"] is True
    assert md_res["status"] == "indexed"

    md_file.write_text("# 文本说明\n\n更新后纯文本内容依然正常摄取。", encoding="utf-8")
    dir_res = rag_index_directory(kb_name="office_dep_kb", dir_path=str(docs_dir))
    assert dir_res["success"] is True
    assert dir_res["total_found"] == 3
    assert dir_res["updated_count"] == 1
    assert dir_res["failed_count"] == 2
    failed_reports = [item for item in dir_res["files"] if item["status"] == "failed"]
    assert len(failed_reports) == 2
    assert all("atengk-mcp-server-rag[office]" in item["message"] for item in failed_reports)


def _make_png_bytes(width: int, height: int, noisy: bool = True) -> bytes:
    """生成指定像素尺寸的真实 PNG 图片字节；noisy=True 时使用高熵随机像素确保体积大于 5KB。"""
    if noisy:
        raw = os.urandom(width * height * 3)
        img = Image.frombytes("RGB", (width, height), raw)
    else:
        img = Image.new("RGB", (width, height), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(width: int, height: int) -> bytes:
    """生成指定尺寸的高熵 JPEG 字节流（>= 5KB）。"""
    raw = os.urandom(width * height * 3)
    img = Image.frombytes("RGB", (width, height), raw)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_exact_png_bytes(width: int, height: int, target_size: int) -> bytes:
    """构造具有精确字节体积且可通过 PIL verify 校验的有效 PNG 图片。"""
    from PIL import PngImagePlugin

    img = Image.new("RGB", (width, height), color=(120, 160, 200))
    base_buf = io.BytesIO()
    img.save(base_buf, format="PNG")
    base_len = len(base_buf.getvalue())
    if target_size <= base_len:
        return base_buf.getvalue()

    # iTXt/tEXt chunk 增加 12 字节固定头开销 + key + NUL + padding
    key = "pad"
    overhead = 12 + len(key) + 1
    pad_len = target_size - base_len - overhead
    assert pad_len >= 0
    meta = PngImagePlugin.PngInfo()
    meta.add_text(key, "x" * pad_len)
    out_buf = io.BytesIO()
    img.save(out_buf, format="PNG", pnginfo=meta)
    assert len(out_buf.getvalue()) == target_size
    return out_buf.getvalue()


def _make_ppm_bytes(width: int, height: int) -> bytes:
    """生成大于 5KB 的非 Web 常见 PPM 格式有效图像字节，验证格式归一化转码。"""
    raw = os.urandom(width * height * 3)
    img = Image.frombytes("RGB", (width, height), raw)
    buf = io.BytesIO()
    img.save(buf, format="PPM")
    return buf.getvalue()


def test_image_noise_filter_thresholds_and_local_storage(isolated_office_env: Path) -> None:
    """验证 Image Noise Filter 精确阈值、损坏流拦截、路径安全越界防御及写入回滚契约。"""
    from mcp_server_rag.assets import (
        MIN_IMAGE_BYTES,
        MIN_IMAGE_DIMENSION_PX,
        _resolve_local_storage_path,
        cleanup_document_local_assets,
        cleanup_kb_local_assets,
        deduplicate_filename,
        inspect_image_bytes,
        persist_local_image_assets,
        sanitize_asset_filename,
    )

    assert MIN_IMAGE_DIMENSION_PX == 100
    assert MIN_IMAGE_BYTES == 5120

    # 1. 精确临界值校验：100x100 + 5120B 放行；5119B / 99x100 / 100x99 / 竖向 40x600 丢弃
    assert inspect_image_bytes(_make_exact_png_bytes(100, 100, 5119), "edge_small.png") is None
    edge_ok = inspect_image_bytes(_make_exact_png_bytes(100, 100, 5120), "edge_ok.png")
    assert edge_ok is not None
    assert edge_ok.width == 100
    assert edge_ok.height == 100
    assert edge_ok.byte_size == 5120

    narrow_w = _make_png_bytes(99, 100, noisy=True)
    assert len(narrow_w) >= MIN_IMAGE_BYTES
    assert inspect_image_bytes(narrow_w, filename_hint="w99.png") is None

    narrow_h = _make_png_bytes(100, 99, noisy=True)
    assert len(narrow_h) >= MIN_IMAGE_BYTES
    assert inspect_image_bytes(narrow_h, filename_hint="h99.png") is None

    vertical_divider = _make_png_bytes(40, 600, noisy=True)
    assert len(vertical_divider) >= MIN_IMAGE_BYTES
    assert inspect_image_bytes(vertical_divider, filename_hint="vdivider.png") is None

    # 纯色大图 (< 5KB) 与带噪微图 (< 100px 但 >= 5KB) 均被拦截
    light_large_png = _make_png_bytes(160, 160, noisy=False)
    assert len(light_large_png) < MIN_IMAGE_BYTES
    assert inspect_image_bytes(light_large_png, filename_hint="blank.png") is None

    small_icon = _make_png_bytes(96, 96, noisy=True)
    assert len(small_icon) >= MIN_IMAGE_BYTES
    assert inspect_image_bytes(small_icon, filename_hint="icon.png") is None

    # 2. 损坏/截断流 PNG（头部完整但 IDAT 截断且 > 5KB）-> verify() 拦截丢弃
    valid_png = _make_png_bytes(220, 180, noisy=True)
    truncated_png = valid_png[:-200]
    assert len(truncated_png) >= MIN_IMAGE_BYTES
    assert inspect_image_bytes(truncated_png, filename_hint="truncated.png") is None
    assert inspect_image_bytes(b"not-an-image-" * 600, filename_hint="corrupt.png") is None

    # 3. 非常规格式（PPM）应被真实转码为 PNG，确保后缀/MIME 与文件魔数一致
    ppm_bytes = _make_ppm_bytes(120, 120)
    assert len(ppm_bytes) >= MIN_IMAGE_BYTES
    ppm_inspected = inspect_image_bytes(ppm_bytes, filename_hint="scan.ppm")
    assert ppm_inspected is not None
    assert ppm_inspected.mime_type == "image/png"
    assert ppm_inspected.filename.endswith(".png")
    with Image.open(io.BytesIO(ppm_inspected.data)) as decoded_img:
        assert decoded_img.format == "PNG"

    # 4. 文件名安全清洗：扩展名穿越防御、Windows 保留设备名逃逸、大小写无关去重
    assert sanitize_asset_filename("../../evil.png", "../../exe") == "evil.exe"
    reserved = sanitize_asset_filename("CON.png", ".png")
    assert reserved != "CON.png" and reserved.lower() != "con.png"
    used: set[str] = set()
    assert deduplicate_filename("Chart.png", used) == "Chart.png"
    assert deduplicate_filename("chart.png", used) == "chart_2.png"

    # 5. 路径越界防御：禁止解析或清理 metadata.db 及 assets 根目录自身
    sentinel_db = isolated_office_env / "metadata.db"
    sentinel_db.write_text("do-not-delete", encoding="utf-8")
    assert _resolve_local_storage_path(isolated_office_env, "rag://assets/../metadata.db") is None
    assert _resolve_local_storage_path(isolated_office_env, "metadata.db") is None
    assert _resolve_local_storage_path(isolated_office_env, "assets/.") is None
    cleanup_kb_local_assets(isolated_office_env, ".")
    cleanup_kb_local_assets(isolated_office_env, "")
    assert sentinel_db.read_text(encoding="utf-8") == "do-not-delete"

    # 6. 有效大图落盘、部分失败回滚及文档级清理
    inspected = inspect_image_bytes(
        valid_png, filename_hint="../dir/arch_diagram.png", alt="架构图"
    )
    assert inspected is not None
    assert inspected.width == 220
    assert inspected.height == 180
    assert inspected.byte_size == len(valid_png)
    assert inspected.mime_type == "image/png"
    assert inspected.filename == "arch_diagram.png"

    doc_id = str(isolated_office_env / "sub" / "arch.pdf")
    records = persist_local_image_assets(
        data_dir=isolated_office_env,
        kb_name="asset_unit_kb",
        doc_id=doc_id,
        images=[inspected],
    )
    assert len(records) == 1
    saved_rel = records[0]["storage_uri"]
    expected_key = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:16]
    assert saved_rel == f"assets/asset_unit_kb/{expected_key}/arch_diagram.png"
    saved_path = isolated_office_env / saved_rel
    assert saved_path.is_file()
    assert saved_path.read_bytes() == valid_png

    # 批量落盘中若后项发生大小写重名冲突，先前已写入同批次的文件须回滚清理
    dup_case = inspect_image_bytes(valid_png, filename_hint="Arch_Diagram.png", alt="重复")
    assert dup_case is not None
    rollback_doc_id = str(isolated_office_env / "sub" / "rollback.pdf")
    rollback_key = hashlib.sha256(rollback_doc_id.encode("utf-8")).hexdigest()[:16]
    with pytest.raises(ValueError, match="文件名冲突"):
        persist_local_image_assets(
            data_dir=isolated_office_env,
            kb_name="asset_unit_kb",
            doc_id=rollback_doc_id,
            images=[inspected, dup_case],
        )
    assert not (isolated_office_env / "assets" / "asset_unit_kb" / rollback_key).exists()

    cleanup_document_local_assets(
        data_dir=isolated_office_env,
        kb_name="asset_unit_kb",
        doc_id=doc_id,
        assets=records,
    )
    assert not saved_path.exists()


def _build_synthetic_docx(
    target_path: Path,
    *,
    heading: str,
    body_text: str,
    valid_image_bytes: bytes,
    valid_alt: str,
    noise_images: list[bytes],
    table_rows: list[tuple[str, str]] | None = None,
) -> None:
    """使用 python-docx 构造包含章节、表格、有效大图（含 Alt）与噪点小图的真实 .docx 夹具。"""
    docx_mod = pytest.importorskip("docx")
    doc = docx_mod.Document()
    doc.add_heading(heading, level=1)
    doc.add_paragraph(body_text)

    # 插入噪点图标，应被噪点门禁静默丢弃
    for noise_bytes in noise_images:
        doc.add_picture(io.BytesIO(noise_bytes))

    # 插入有效大图并写入 OOXML wp:docPr/@descr 作为图片 Alt 语义
    pic_para = doc.add_paragraph()
    run = pic_para.add_run()
    run.add_picture(io.BytesIO(valid_image_bytes))
    for el in pic_para._element.iter():
        if el.tag.endswith("}docPr"):
            el.set("descr", valid_alt)

    if table_rows:
        table = doc.add_table(rows=len(table_rows), cols=2)
        for r_idx, (c0, c1) in enumerate(table_rows):
            table.cell(r_idx, 0).text = c0
            table.cell(r_idx, 1).text = c1

    doc.save(str(target_path))


def test_docx_e2e_ingestion_noise_filter_anchoring_and_idempotency(
    isolated_office_env: Path,
) -> None:
    """验证 .docx 解析、噪点过滤、本地资产打桩、images 回填、幂等跳过与原子更新。"""
    from mcp_server_rag.assets import make_asset_doc_key

    assert rag_create_kb(name="docx_kb", description="Word 文档多模态测试库")["success"] is True

    docx_path = isolated_office_env / "gateway_design.docx"
    valid_diagram = _make_png_bytes(240, 200, noisy=True)
    tiny_icon = _make_png_bytes(48, 48, noisy=True)
    thin_divider = _make_png_bytes(500, 30, noisy=True)
    blank_large = _make_png_bytes(180, 180, noisy=False)

    _build_synthetic_docx(
        docx_path,
        heading="分布式网关架构设计",
        body_text="本文说明网关熔断降级机制与核心链路流量调度策略。",
        valid_image_bytes=valid_diagram,
        valid_alt="微服务网关熔断架构图",
        noise_images=[tiny_icon, thin_divider, blank_large],
        table_rows=[("组件名称", "监听端口"), ("API-Gateway", "8443")],
    )

    # 1. 首次索引：验证提取、去噪与就地打桩
    res = rag_index_file(kb_name="docx_kb", file_path=str(docx_path))
    assert res["success"] is True
    assert res["status"] == "indexed"
    assert res["chunk_count"] >= 1
    doc_id = res["doc_id"]
    doc_key = make_asset_doc_key(doc_id)

    catalog = get_metadata_catalog(get_settings())
    assets_v1 = catalog.list_image_assets("docx_kb", doc_id=doc_id)
    assert len(assets_v1) == 1
    asset_v1 = assets_v1[0]
    assert asset_v1["mime_type"] == "image/png"
    assert asset_v1["byte_size"] == len(valid_diagram)
    # 同时支持按原始 doc_id 与单段哈希 doc_key 精确反查资产记录（兼容 Issue #18 Resource 路由）
    assert catalog.get_image_asset("docx_kb", doc_id, asset_v1["filename"]) is not None
    by_key = catalog.get_image_asset("docx_kb", doc_key, asset_v1["filename"])
    assert by_key is not None
    assert by_key["doc_id"] == doc_id
    assert by_key["storage_uri"] == asset_v1["storage_uri"]

    asset_v1_path = isolated_office_env / asset_v1["storage_uri"]
    assert asset_v1_path.is_file()
    assert asset_v1_path.read_bytes() == valid_diagram
    # 确保磁盘资产目录内不存在其他噪点文件
    doc_asset_dir = isolated_office_env / "assets" / "docx_kb" / doc_key
    assert [p.name for p in doc_asset_dir.iterdir()] == [asset_v1["filename"]]

    # 2. 通过 rag_search 验证文本、表格与图片锚点及 metadata.images
    expected_uri = f"rag://assets/docx_kb/{doc_key}/{asset_v1['filename']}"
    hits = rag_search(kb_name="docx_kb", query="微服务网关熔断架构图 API-Gateway", top_k=3)
    assert len(hits) >= 1
    combined_text = "\n".join(h["text"] for h in hits)
    assert "分布式网关架构设计" in combined_text
    assert "API-Gateway" in combined_text
    assert f"![微服务网关熔断架构图]({expected_uri})" in combined_text
    assert any(expected_uri in h["metadata"]["images"] for h in hits)

    # 3. 幂等跳过验证
    skip_res = rag_index_file(kb_name="docx_kb", file_path=str(docx_path))
    assert skip_res["success"] is True
    assert skip_res["status"] == "skipped"

    # 4. 原子更新：更换新文本与新图片，验证旧物理文件被清理且新资产生效
    new_diagram = _make_jpeg_bytes(240, 220)
    _build_synthetic_docx(
        docx_path,
        heading="全新零信任网关架构",
        body_text="已升级至基于 mTLS 双向认证的零信任安全网关架构。",
        valid_image_bytes=new_diagram,
        valid_alt="零信任安全认证流程图",
        noise_images=[tiny_icon],
    )
    update_res = rag_index_file(kb_name="docx_kb", file_path=str(docx_path))
    assert update_res["success"] is True
    assert update_res["status"] == "updated"
    assert update_res["content_hash"] != res["content_hash"]

    assets_v2 = catalog.list_image_assets("docx_kb", doc_id=doc_id)
    assert len(assets_v2) == 1
    assert assets_v2[0]["mime_type"] == "image/jpeg"
    assert not asset_v1_path.exists()
    asset_v2_path = isolated_office_env / assets_v2[0]["storage_uri"]
    assert asset_v2_path.is_file()
    assert asset_v2_path.read_bytes() == new_diagram

    updated_hits = rag_search(kb_name="docx_kb", query="零信任安全认证流程图", top_k=3)
    expected_uri_v2 = f"rag://assets/docx_kb/{doc_key}/{assets_v2[0]['filename']}"
    assert any(expected_uri_v2 in h["metadata"]["images"] for h in updated_hits)


def _build_synthetic_pdf(
    target_path: Path,
    pages: list[tuple[str, list[bytes]]],
) -> None:
    """使用 pypdf 与 Pillow 构造多页含真实文本流与内嵌图片的 .pdf 夹具。"""
    pypdf_mod = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf_mod.PdfWriter()
    for page_text, image_bytes_list in pages:
        text_writer = pypdf_mod.PdfWriter()
        page = text_writer.add_blank_page(width=595, height=842)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_ref = text_writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
        )
        escaped = page_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1"))
        page[NameObject("/Contents")] = text_writer._add_object(stream)
        text_buf = io.BytesIO()
        text_writer.write(text_buf)
        text_buf.seek(0)
        base_page = pypdf_mod.PdfReader(text_buf).pages[0]

        out_page = writer.add_page(base_page)
        for img_raw in image_bytes_list:
            with Image.open(io.BytesIO(img_raw)) as pil_img:
                img_pdf_buf = io.BytesIO()
                pil_img.convert("RGB").save(img_pdf_buf, format="PDF")
                img_pdf_buf.seek(0)
                img_page = pypdf_mod.PdfReader(img_pdf_buf).pages[0]
                out_page.merge_page(img_page)

    with target_path.open("wb") as fp:
        writer.write(fp)


def test_pdf_e2e_ingestion_noise_filter_anchoring_and_idempotency(
    isolated_office_env: Path,
) -> None:
    """端到端验证 .pdf 解析、噪点过滤、本地资产落盘与页面打桩、images 召回与幂等更新。"""
    from mcp_server_rag.assets import make_asset_doc_key

    assert rag_create_kb(name="pdf_kb", description="PDF 文档多模态测试库")["success"] is True

    pdf_path = isolated_office_env / "milvus_architecture.pdf"
    valid_chart = _make_png_bytes(240, 240, noisy=True)
    tiny_icon = _make_png_bytes(48, 48, noisy=True)
    blank_logo = _make_png_bytes(160, 160, noisy=False)

    _build_synthetic_pdf(
        pdf_path,
        pages=[
            ("Milvus Vector Engine Architecture Overview", [valid_chart, tiny_icon]),
            ("Deployment Topology and Monitoring Checklist", [blank_logo]),
        ],
    )

    # 1. 首次索引：验证提取、去噪、打桩与落盘
    res = rag_index_file(kb_name="pdf_kb", file_path=str(pdf_path))
    assert res["success"] is True
    assert res["status"] == "indexed"
    doc_id = res["doc_id"]
    doc_key = make_asset_doc_key(doc_id)

    catalog = get_metadata_catalog(get_settings())
    assets_v1 = catalog.list_image_assets("pdf_kb", doc_id=doc_id)
    assert len(assets_v1) == 1
    asset_v1 = assets_v1[0]
    asset_v1_path = isolated_office_env / asset_v1["storage_uri"]
    assert asset_v1_path.is_file()
    assert asset_v1["byte_size"] >= 5120

    # 2. 检索第 1 页应命中图片锚点与 metadata.images；第 2 页仅含噪点图标故无图片
    expected_uri_v1 = f"rag://assets/pdf_kb/{doc_key}/{asset_v1['filename']}"
    p1_hits = rag_search(kb_name="pdf_kb", query="Milvus Vector Engine Architecture", top_k=2)
    assert len(p1_hits) >= 1
    assert any(expected_uri_v1 in h["metadata"]["images"] for h in p1_hits)
    assert any(f"]({expected_uri_v1})" in h["text"] for h in p1_hits)

    p2_hits = rag_search(
        kb_name="pdf_kb", query="Deployment Topology Monitoring Checklist", top_k=1
    )
    assert len(p2_hits) == 1
    assert "Deployment Topology" in p2_hits[0]["text"]
    assert p2_hits[0]["metadata"]["images"] == []

    # 3. 幂等跳过
    skip_res = rag_index_file(kb_name="pdf_kb", file_path=str(pdf_path))
    assert skip_res["success"] is True
    assert skip_res["status"] == "skipped"

    # 4. 更新 PDF 内容与图片位置（第 1 页纯文本，第 2 页含新图），验证旧文件清理及新资产登记
    new_chart = _make_png_bytes(260, 220, noisy=True)
    _build_synthetic_pdf(
        pdf_path,
        pages=[
            ("Updated Cluster Sharding Overview", []),
            ("Cluster Sharding Blueprint Diagram", [new_chart]),
        ],
    )
    update_res = rag_index_file(kb_name="pdf_kb", file_path=str(pdf_path))
    assert update_res["success"] is True
    assert update_res["status"] == "updated"

    assets_v2 = catalog.list_image_assets("pdf_kb", doc_id=doc_id)
    assert len(assets_v2) == 1
    assert assets_v2[0]["filename"].startswith("page_2_")
    assert not asset_v1_path.exists()
    asset_v2_path = isolated_office_env / assets_v2[0]["storage_uri"]
    assert asset_v2_path.is_file()


def test_directory_batch_ingestion_and_cascading_physical_asset_cleanup(
    isolated_office_env: Path,
) -> None:
    """验证目录批量摄取 .pdf/.docx/.md 及删文档/清库/删库时的物理图片级联原子清理。"""
    assert rag_create_kb(name="cascade_kb", description="级联清理测试库")["success"] is True

    batch_dir = isolated_office_env / "knowledge_bundle"
    batch_dir.mkdir(parents=True, exist_ok=True)
    pdf_file = batch_dir / "cluster.pdf"
    docx_file = batch_dir / "handbook.docx"
    md_file = batch_dir / "notes.md"

    _build_synthetic_pdf(
        pdf_file,
        pages=[("Cluster Replication Protocol", [_make_png_bytes(220, 220, noisy=True)])],
    )
    _build_synthetic_docx(
        docx_file,
        heading="运维手册",
        body_text="容器集群自动扩缩容操作手册。",
        valid_image_bytes=_make_png_bytes(220, 220, noisy=True),
        valid_alt="集群扩缩容流程图",
        noise_images=[_make_png_bytes(32, 32, noisy=True)],
    )
    md_file.write_text("# 备忘录\n\n日常巡检命令速查表。", encoding="utf-8")

    # 1. 目录批量摄取应同时识别 .pdf、.docx 与 .md
    dir_res = rag_index_directory(kb_name="cascade_kb", dir_path=str(batch_dir))
    assert dir_res["success"] is True
    assert dir_res["total_found"] == 3
    assert dir_res["indexed_count"] == 3

    catalog = get_metadata_catalog(get_settings())
    pdf_assets = catalog.list_image_assets("cascade_kb", doc_id=str(pdf_file.resolve()))
    docx_assets = catalog.list_image_assets("cascade_kb", doc_id=str(docx_file.resolve()))
    assert len(pdf_assets) == 1
    assert len(docx_assets) == 1

    pdf_img_path = isolated_office_env / pdf_assets[0]["storage_uri"]
    docx_img_path = isolated_office_env / docx_assets[0]["storage_uri"]
    assert pdf_img_path.is_file()
    assert docx_img_path.is_file()

    # 2. 再次目录扫描应全部幂等跳过
    dir_skip = rag_index_directory(kb_name="cascade_kb", dir_path=str(batch_dir))
    assert dir_skip["skipped_count"] == 3

    # 3. 删除单个 PDF 文档：同步清理其物理图片与 image_assets 记录，保留 Word 物理图片
    del_pdf = rag_delete_document(kb_name="cascade_kb", doc_id=str(pdf_file.resolve()))
    assert del_pdf["success"] is True
    assert not pdf_img_path.exists()
    assert not pdf_img_path.parent.exists()
    assert catalog.list_image_assets("cascade_kb", doc_id=str(pdf_file.resolve())) == []
    assert docx_img_path.is_file()
    assert len(catalog.list_image_assets("cascade_kb", doc_id=str(docx_file.resolve()))) == 1

    # 4. 清空知识库：同步清理所有剩余物理图片与记录，但保留知识库
    clear_res = rag_clear_kb(name="cascade_kb")
    assert clear_res["success"] is True
    assert not docx_img_path.exists()
    assert not (isolated_office_env / "assets" / "cascade_kb").exists()
    assert catalog.list_image_assets("cascade_kb") == []
    assert catalog.get_kb("cascade_kb") is not None

    # 5. 按文件名 title 删除文档同样须经 actual_doc_id 级联删除本地 doc_key 资产目录
    title_docx = isolated_office_env / "by_title_doc.docx"
    _build_synthetic_docx(
        title_docx,
        heading="按标题删文档测试",
        body_text="验证按 title 删除文档时也能清理物理资产。",
        valid_image_bytes=_make_png_bytes(160, 160, noisy=True),
        valid_alt="标题删除测试图",
        noise_images=[],
    )
    assert rag_create_kb(name="drop_assets_kb", description="删库清理")["success"] is True
    assert rag_index_file(kb_name="drop_assets_kb", file_path=str(title_docx))["success"] is True
    assert rag_index_file(kb_name="drop_assets_kb", file_path=str(docx_file))["success"] is True

    title_assets = catalog.list_image_assets("drop_assets_kb", doc_id=str(title_docx.resolve()))
    assert len(title_assets) == 1
    title_img_path = isolated_office_env / title_assets[0]["storage_uri"]
    assert title_img_path.is_file()

    del_by_title = rag_delete_document(kb_name="drop_assets_kb", doc_id="by_title_doc.docx")
    assert del_by_title["success"] is True
    assert not title_img_path.exists()
    assert not title_img_path.parent.exists()

    # 6. 删除知识库：级联清理全部剩余物理资产文件，无孤儿文件残留
    drop_assets = catalog.list_image_assets("drop_assets_kb")
    assert len(drop_assets) == 1
    drop_img_path = isolated_office_env / drop_assets[0]["storage_uri"]
    assert drop_img_path.is_file()

    drop_res = rag_delete_kb(name="drop_assets_kb")
    assert drop_res["success"] is True
    assert not drop_img_path.exists()
    assert not (isolated_office_env / "assets" / "drop_assets_kb").exists()


def test_docx_sdt_tables_alternate_content_and_error_cases(
    isolated_office_env: Path,
) -> None:
    """验证 DOCX 内容控件 <w:sdt>、表格内嵌图、mc:AlternateContent 去重与损坏/全噪点容错。"""
    docx_mod = pytest.importorskip("docx")
    from docx.oxml import parse_xml
    from docx.shared import Inches

    assert (
        rag_create_kb(name="docx_adv_kb", description="DOCX 高级 XML 解析测试")["success"] is True
    )

    doc_path = isolated_office_env / "advanced_ooxml.docx"
    doc = docx_mod.Document()

    # 1) 表格单元格内嵌有效大图 + 嵌套表格，且单元格同行不被空行截断
    tbl_img_bytes = _make_png_bytes(160, 150, noisy=True)
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "网关服务节点"
    cell_para = table.cell(0, 1).paragraphs[0]
    cell_para.add_run("核心状态图：")
    cell_para.add_run().add_picture(io.BytesIO(tbl_img_bytes), width=Inches(1.5))
    nested_tbl = table.cell(0, 0).add_table(rows=1, cols=2)
    nested_tbl.cell(0, 0).text = "嵌套键"
    nested_tbl.cell(0, 1).text = "嵌套值_X9"

    # 2) <mc:AlternateContent> 内同时包含 Choice(<w:drawing>) 与 Fallback(<w:pict>)：只能提取 1 次
    alt_img_bytes = _make_png_bytes(180, 180, noisy=True)
    alt_para = doc.add_paragraph("主备内容控件说明：")
    alt_run = alt_para.add_run()
    alt_run.add_picture(io.BytesIO(alt_img_bytes))
    drawing_el = alt_run._r.find(
        ".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing"
    )
    assert drawing_el is not None
    # 找到对应的 relationship id
    blip_el = drawing_el.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip")
    assert blip_el is not None
    alt_rid = blip_el.attrib[
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
    ]
    alt_run._r.remove(drawing_el)
    mc_xml = (
        "<mc:AlternateContent "
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<mc:Choice Requires="wps">{drawing_placeholder}</mc:Choice>'
        '<mc:Fallback><w:pict><v:shape><v:imagedata r:id="'
        + alt_rid
        + '" alt="备选大图"/></v:shape></w:pict></mc:Fallback>'
        "</mc:AlternateContent>"
    ).replace("{drawing_placeholder}", "")
    mc_el = parse_xml(mc_xml)
    choice_el = mc_el.find("{http://schemas.openxmlformats.org/markup-compatibility/2006}Choice")
    assert choice_el is not None
    choice_el.append(drawing_el)
    alt_run._r.append(mc_el)

    # 3) 内容控件 <w:sdt>/<w:sdtContent> 包裹段落文本与有效大图
    sdt_img_bytes = _make_png_bytes(170, 170, noisy=True)
    sdt_rid, _ = doc.part.get_or_add_image(io.BytesIO(sdt_img_bytes))
    # 使用独立关系写入 <w:pict><v:imagedata> 亦同时覆盖 v:imagedata 提取路径
    body_sdt = parse_xml(
        '<w:sdt xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<w:sdtContent><w:p><w:r><w:t>SDT_CONTENT_PARAGRAPH_TAG</w:t></w:r>"
        '<w:r><w:pict><v:shape><v:imagedata r:id="'
        + sdt_rid
        + '" title="内容控件架构图"/></v:shape></w:pict></w:r></w:p></w:sdtContent></w:sdt>'
    )
    doc.element.body.append(body_sdt)
    doc.save(str(doc_path))

    res = rag_index_file(kb_name="docx_adv_kb", file_path=str(doc_path))
    assert res["success"] is True
    catalog = get_metadata_catalog(get_settings())
    assets = catalog.list_image_assets("docx_adv_kb", doc_id=res["doc_id"])
    # 严格提取 3 张有效图：表格图 + Choice 单次图（Fallback 未重复）+ SDT legacy pict 图
    assert len(assets) == 3

    hits_sdt = rag_search(
        kb_name="docx_adv_kb", query="SDT_CONTENT_PARAGRAPH_TAG 内容控件架构图", top_k=3
    )
    assert any("SDT_CONTENT_PARAGRAPH_TAG" in h["text"] for h in hits_sdt)
    assert any("嵌套值_X9" in h["text"] for h in hits_sdt)
    assert any(h["metadata"]["images"] for h in hits_sdt)

    # 4) 损坏的 .docx 与仅有噪点图无正文的 .docx 均应优雅返回 success=False
    bad_docx = isolated_office_env / "corrupt.docx"
    bad_docx.write_bytes(b"NOT_A_VALID_DOCX_ZIP")
    bad_res = rag_index_file(kb_name="docx_adv_kb", file_path=str(bad_docx))
    assert bad_res["success"] is False

    noise_only_docx = isolated_office_env / "noise_only.docx"
    noise_doc = docx_mod.Document()
    noise_doc.add_picture(io.BytesIO(_make_png_bytes(40, 40, noisy=True)))
    noise_doc.save(str(noise_only_docx))
    noise_res = rag_index_file(kb_name="docx_adv_kb", file_path=str(noise_only_docx))
    assert noise_res["success"] is False


def test_pdf_fault_isolation_dedup_encryption_scan_and_rollback(
    isolated_office_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 PDF 单图异常隔离、跨页同图 SHA-256 去重、加密/损坏/纯图扫描件及失败补偿回滚。"""
    pypdf_mod = pytest.importorskip("pypdf")
    from mcp_server_rag.office import _parse_pdf_document

    assert rag_create_kb(name="pdf_adv_kb", description="PDF 边界与回滚测试")["success"] is True

    # 1. 跨页包含完全同一张有效大图 + 1 张仅正文页：仅落盘 1 份物理文件且双页均含锚点
    shared_img = _make_png_bytes(160, 160, noisy=True)
    dedup_pdf = isolated_office_env / "cross_page_dedup.pdf"
    _build_synthetic_pdf(
        dedup_pdf,
        pages=[
            ("Page One Shared Diagram Header", [shared_img]),
            ("Page Two Shared Diagram Reference", [shared_img]),
        ],
    )
    res_dedup = rag_index_file(kb_name="pdf_adv_kb", file_path=str(dedup_pdf))
    assert res_dedup["success"] is True
    catalog = get_metadata_catalog(get_settings())
    dedup_assets = catalog.list_image_assets("pdf_adv_kb", doc_id=res_dedup["doc_id"])
    assert len(dedup_assets) == 1

    # 更新为纯文本无图 PDF：旧物理图片目录及 image_assets 必须全部清空
    old_asset_path = isolated_office_env / dedup_assets[0]["storage_uri"]
    assert old_asset_path.is_file()
    _build_synthetic_pdf(dedup_pdf, pages=[("Text Only Version Two", [])])
    upd_zero = rag_index_file(kb_name="pdf_adv_kb", file_path=str(dedup_pdf))
    assert upd_zero["success"] is True
    assert upd_zero["status"] == "updated"
    assert catalog.list_image_assets("pdf_adv_kb", doc_id=res_dedup["doc_id"]) == []
    assert not old_asset_path.exists()
    assert not old_asset_path.parent.exists()

    # 2. 纯图片无文本扫描件 PDF：应依靠 PDF 第 1 页插图锚点完成索引与召回
    scan_pdf = isolated_office_env / "scanned_only.pdf"
    _build_synthetic_pdf(scan_pdf, pages=[("", [_make_png_bytes(180, 180, noisy=True)])])
    scan_res = rag_index_file(kb_name="pdf_adv_kb", file_path=str(scan_pdf))
    assert scan_res["success"] is True
    scan_hits = rag_search(kb_name="pdf_adv_kb", query="PDF 第 1 页插图", top_k=1)
    assert len(scan_hits) == 1
    assert len(scan_hits[0]["metadata"]["images"]) == 1

    # 3. 单页多图故障隔离：第 1 张图读取抛出 NotImplementedError，第 2 张有效大图仍须成功提取
    two_img_pdf = isolated_office_env / "fault_isolation.pdf"
    _build_synthetic_pdf(
        two_img_pdf,
        pages=[
            (
                "Multi Image Isolation Page",
                [_make_png_bytes(140, 140, noisy=True), _make_png_bytes(150, 150, noisy=True)],
            )
        ],
    )
    orig_reader_cls = pypdf_mod.PdfReader

    class _FaultyFirstImageList:
        def __init__(self, wrapped: object) -> None:
            self._wrapped = wrapped

        def __len__(self) -> int:
            return len(self._wrapped)  # type: ignore[arg-type]

        def __getitem__(self, idx: int) -> object:
            if idx == 0:
                raise NotImplementedError("Simulated unsupported filter on image 0")
            return self._wrapped[idx]  # type: ignore[index]

    class _FaultyReader(orig_reader_cls):  # type: ignore[valid-type,misc]
        def __init__(self, stream: object) -> None:
            super().__init__(stream)
            orig_page = self.pages[0]
            faulty_images = _FaultyFirstImageList(orig_page.images)
            monkeypatch.setattr(type(orig_page), "images", property(lambda self: faulty_images))

    monkeypatch.setattr(pypdf_mod, "PdfReader", _FaultyReader)
    parsed_isolated = _parse_pdf_document(
        two_img_pdf, kb_name="pdf_adv_kb", doc_id=str(two_img_pdf)
    )
    monkeypatch.undo()
    assert len(parsed_isolated.images) == 1

    # 4. 加密 PDF 与损坏 PDF 优雅返回 success=False
    encrypted_pdf = isolated_office_env / "protected.pdf"
    enc_writer = pypdf_mod.PdfWriter()
    enc_writer.add_blank_page(width=400, height=400)
    enc_writer.encrypt("TopSecret123")
    with encrypted_pdf.open("wb") as fp:
        enc_writer.write(fp)
    enc_res = rag_index_file(kb_name="pdf_adv_kb", file_path=str(encrypted_pdf))
    assert enc_res["success"] is False

    corrupt_pdf = isolated_office_env / "broken.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.7\nbroken_stream")
    corrupt_res = rag_index_file(kb_name="pdf_adv_kb", file_path=str(corrupt_pdf))
    assert corrupt_res["success"] is False

    # 5. 资产登记异常回滚验证：register_image_assets 失败须全部回滚，重试不误判 skipped
    rollback_pdf = isolated_office_env / "rollback_test.pdf"
    _build_synthetic_pdf(
        rollback_pdf,
        pages=[("Rollback Recovery Document", [_make_png_bytes(160, 160, noisy=True)])],
    )
    from mcp_server_rag.assets import make_asset_doc_key

    rollback_doc_id = str(rollback_pdf.resolve())
    rollback_dir = (
        isolated_office_env / "assets" / "pdf_adv_kb" / make_asset_doc_key(rollback_doc_id)
    )
    orig_reg = type(catalog).register_image_assets

    def _boom_register(
        self: object, kb_name: str, doc_id: str, assets: list[dict[str, object]]
    ) -> None:
        raise RuntimeError("Simulated catalog asset write failure")

    monkeypatch.setattr(type(catalog), "register_image_assets", _boom_register)
    with pytest.raises(RuntimeError, match="Simulated catalog asset write failure"):
        rag_index_file(kb_name="pdf_adv_kb", file_path=str(rollback_pdf))
    monkeypatch.setattr(type(catalog), "register_image_assets", orig_reg)

    assert catalog.get_document_by_id("pdf_adv_kb", rollback_doc_id) is None
    assert not rollback_dir.exists()
    retry_res = rag_index_file(kb_name="pdf_adv_kb", file_path=str(rollback_pdf))
    assert retry_res["success"] is True
    assert retry_res["status"] == "indexed"
