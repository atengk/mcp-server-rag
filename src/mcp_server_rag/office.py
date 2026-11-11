"""
轻量二进制办公文档（PDF / Word）解析与图文锚点注入模块。

基于纯 Python 可选依赖（`pypdf` 与 `python-docx`，对应 `[office]` 安装分组）提取正文文本
与内嵌二进制图片，并通过噪点门禁清洗后就地注入标准 Markdown 图片引用。

@author Ateng
@since 2026-11-11
"""

import hashlib
import importlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mcp_server_rag.assets import (
    ExtractedImage,
    build_asset_resource_uri,
    deduplicate_filename,
    inspect_image_bytes,
    make_asset_doc_key,
)

logger = logging.getLogger(__name__)

OFFICE_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx"})
DEFAULT_DOCX_IMAGE_ALT: str = "Word 文档插图"

_REL_EMBED_ATTR = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
_REL_ID_ATTR = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_DOCX_HEADING_PREFIXES: dict[str, str] = {
    "Title": "# ",
    "Subtitle": "## ",
    "Heading 1": "# ",
    "Heading 2": "## ",
    "Heading 3": "### ",
    "Heading 4": "#### ",
    "Heading 5": "##### ",
    "Heading 6": "###### ",
}


@dataclass(frozen=True)
class ParsedOfficeDocument:
    """二进制办公文档归一化解析结果（含就地打桩的 Markdown 文本与过滤后有效图片）。"""

    markdown_text: str
    images: list[ExtractedImage]


def is_office_document(file_path: Path) -> bool:
    """判断路径扩展名是否属于受支持的二进制办公文档格式（``.pdf`` / ``.docx``）。"""
    return file_path.suffix.lower() in OFFICE_EXTENSIONS


def _load_pypdf() -> Any:
    """惰性加载 pypdf 模块，缺失时返回包含 [office] 安装指引的 ImportError。"""
    try:
        return importlib.import_module("pypdf")
    except ImportError as exc:
        raise ImportError(
            "未检测到 pypdf 依赖，无法解析 PDF 文档。"
            "请运行 `pip install atengk-mcp-server-rag[office]` "
            '或 `uv pip install "pypdf>=4.0.0"` 进行安装。'
        ) from exc


def _load_python_docx() -> Any:
    """惰性加载 python-docx 模块，缺失时返回包含 [office] 安装指引的 ImportError。"""
    try:
        return importlib.import_module("docx")
    except ImportError as exc:
        raise ImportError(
            "未检测到 python-docx 依赖，无法解析 Word (.docx) 文档。"
            "请运行 `pip install atengk-mcp-server-rag[office]` "
            '或 `uv pip install "python-docx>=1.1.0"` 进行安装。'
        ) from exc


def ensure_office_dependency(file_path: Path) -> None:
    """按文件扩展名校验对应办公文档解析依赖是否已安装。

    Args:
        file_path: 待解析的文档路径。

    Raises:
        ImportError: 当 ``.pdf`` 缺少 ``pypdf`` 或 ``.docx`` 缺少 ``python-docx`` 时抛出。
    """
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        _load_pypdf()
    elif ext == ".docx":
        _load_python_docx()


def _format_markdown_anchor(kb_name: str, doc_key: str, image: ExtractedImage) -> str:
    """构造转义安全、受原子切片保护的 Markdown 图片引用标记。"""
    safe_alt = image.alt.replace("[", "(").replace("]", ")")
    uri = build_asset_resource_uri(kb_name, doc_key, image.filename)
    return f"![{safe_alt}]({uri})"


def _register_unique_image(
    raw_bytes: bytes,
    filename_hint: str,
    alt_hint: str,
    valid_images: list[ExtractedImage],
    used_lower_filenames: set[str],
    content_cache: dict[str, ExtractedImage],
) -> ExtractedImage | None:
    """按 SHA-256 内容去重、执行噪点门禁清洗并分配大小写无冲突文件名。"""
    if not raw_bytes:
        return None
    digest = hashlib.sha256(raw_bytes).hexdigest()
    cached = content_cache.get(digest)
    if cached is not None:
        return cached

    inspected = inspect_image_bytes(raw_bytes, filename_hint=filename_hint, alt=alt_hint)
    if inspected is None:
        return None

    unique_name = deduplicate_filename(inspected.filename, used_lower_filenames)
    final_image = (
        replace(inspected, filename=unique_name) if unique_name != inspected.filename else inspected
    )
    content_cache[digest] = final_image
    valid_images.append(final_image)
    return final_image


def _walk_docx_elem(
    elem: Any,
    text_parts: list[str],
    anchors: list[str],
    seen_xml_ids: set[int],
    kb_name: str,
    doc_key: str,
    extract_rid: Callable[[str, str], ExtractedImage | None],
) -> None:
    """递归遍历 OOXML 节点，优先保留 AlternateContent 的 Choice 并跳过修订删除分支。"""
    local_name = elem.tag.rsplit("}", 1)[-1]
    if local_name in {"del", "delText", "instrText"}:
        return

    if local_name == "AlternateContent":
        children = list(elem)
        choices = [ch for ch in children if ch.tag.rsplit("}", 1)[-1] == "Choice"]
        if choices:
            prev_text_len = len(text_parts)
            prev_anchor_len = len(anchors)
            _walk_docx_elem(
                choices[0],
                text_parts,
                anchors,
                seen_xml_ids,
                kb_name,
                doc_key,
                extract_rid,
            )
            if len(text_parts) > prev_text_len or len(anchors) > prev_anchor_len:
                return
        for fb in [ch for ch in children if ch.tag.rsplit("}", 1)[-1] == "Fallback"]:
            _walk_docx_elem(fb, text_parts, anchors, seen_xml_ids, kb_name, doc_key, extract_rid)
        return

    if local_name == "t" and elem.text:
        text_parts.append(elem.text)
    elif local_name == "tab":
        text_parts.append("\t")
    elif local_name in {"br", "cr"}:
        text_parts.append("\n")
    elif local_name == "drawing":
        alt_hint = ""
        for sub in elem.iter():
            if sub.tag.rsplit("}", 1)[-1] == "docPr":
                alt_hint = (
                    sub.attrib.get("descr")
                    or sub.attrib.get("title")
                    or sub.attrib.get("name")
                    or ""
                )
                break
        for sub in elem.iter():
            if sub.tag.rsplit("}", 1)[-1] == "blip" and id(sub) not in seen_xml_ids:
                seen_xml_ids.add(id(sub))
                r_id = sub.attrib.get(_REL_EMBED_ATTR) or ""
                if r_id:
                    image = extract_rid(r_id, alt_hint or DEFAULT_DOCX_IMAGE_ALT)
                    if image is not None:
                        anchors.append(_format_markdown_anchor(kb_name, doc_key, image))
        return
    elif local_name == "pict":
        for sub in elem.iter():
            if sub.tag.rsplit("}", 1)[-1] == "imagedata" and id(sub) not in seen_xml_ids:
                seen_xml_ids.add(id(sub))
                r_id = sub.attrib.get(_REL_ID_ATTR) or ""
                alt_hint = (
                    sub.attrib.get("alt") or sub.attrib.get("title") or DEFAULT_DOCX_IMAGE_ALT
                )
                if r_id:
                    image = extract_rid(r_id, alt_hint)
                    if image is not None:
                        anchors.append(_format_markdown_anchor(kb_name, doc_key, image))
        return

    for child in elem:
        _walk_docx_elem(child, text_parts, anchors, seen_xml_ids, kb_name, doc_key, extract_rid)


def _extract_docx_paragraph(
    paragraph: Any,
    kb_name: str,
    doc_key: str,
    extract_rid: Callable[[str, str], ExtractedImage | None],
    *,
    multiline: bool = True,
) -> str:
    """按 OOXML 阅读流抽取 Word 段落正文、标题层级与内嵌图片锚点。"""
    style_name = str(getattr(getattr(paragraph, "style", None), "name", "") or "")
    heading_prefix = _DOCX_HEADING_PREFIXES.get(style_name, "") if multiline else ""

    text_parts: list[str] = []
    anchors: list[str] = []
    seen_xml_ids: set[int] = set()
    _walk_docx_elem(
        paragraph._element,
        text_parts,
        anchors,
        seen_xml_ids,
        kb_name,
        doc_key,
        extract_rid,
    )

    raw_text = re.sub(r"[ \t]+", " ", "".join(text_parts)).strip()
    if not multiline:
        raw_text = re.sub(r"\s+", " ", raw_text).strip()
        cell_pieces = [piece for piece in (raw_text, *anchors) if piece]
        return " ".join(cell_pieces)

    pieces: list[str] = []
    if raw_text:
        pieces.append(f"{heading_prefix}{raw_text}" if heading_prefix else raw_text)
    if anchors:
        pieces.extend(anchors)
    return "\n\n".join(pieces)


def _extract_docx_cell_content(
    tc_elem: Any,
    parent: Any,
    kb_name: str,
    doc_key: str,
    extract_rid: Callable[[str, str], ExtractedImage | None],
    paragraph_cls: Any,
    table_cls: Any,
) -> str:
    """递归抽取单元格内段落、嵌套内容控件与嵌套表格文本及图片锚点（保持单行安全）。"""
    parts: list[str] = []
    for child in tc_elem.iterchildren():
        local_name = child.tag.rsplit("}", 1)[-1]
        if local_name == "p":
            text = _extract_docx_paragraph(
                paragraph_cls(child, parent),
                kb_name=kb_name,
                doc_key=doc_key,
                extract_rid=extract_rid,
                multiline=False,
            )
            if text:
                parts.append(text)
        elif local_name == "tbl":
            nested = _extract_docx_table(
                table_cls(child, parent),
                parent,
                kb_name=kb_name,
                doc_key=doc_key,
                extract_rid=extract_rid,
                paragraph_cls=paragraph_cls,
                table_cls=table_cls,
            )
            if nested:
                parts.append(nested.replace("\n", " ; "))
        elif local_name == "sdt":
            sdt_content = next(
                (el for el in child.iterchildren() if el.tag.rsplit("}", 1)[-1] == "sdtContent"),
                None,
            )
            if sdt_content is not None:
                nested_text = _extract_docx_cell_content(
                    sdt_content,
                    parent,
                    kb_name=kb_name,
                    doc_key=doc_key,
                    extract_rid=extract_rid,
                    paragraph_cls=paragraph_cls,
                    table_cls=table_cls,
                )
                if nested_text:
                    parts.append(nested_text)
    return " ".join(parts).strip()


def _extract_docx_table(
    table: Any,
    parent: Any,
    kb_name: str,
    doc_key: str,
    extract_rid: Callable[[str, str], ExtractedImage | None],
    paragraph_cls: Any,
    table_cls: Any,
) -> str:
    """抽取 Word 表格按行拼接文本与单元格内嵌图片，合并单元格仅提取一次。"""
    row_lines: list[str] = []
    for row in table.rows:
        seen_tcs: set[int] = set()
        cell_texts: list[str] = []
        for cell in row.cells:
            tc_id = id(cell._tc)
            if tc_id in seen_tcs:
                continue
            seen_tcs.add(tc_id)
            cell_str = _extract_docx_cell_content(
                cell._tc,
                parent,
                kb_name=kb_name,
                doc_key=doc_key,
                extract_rid=extract_rid,
                paragraph_cls=paragraph_cls,
                table_cls=table_cls,
            )
            if cell_str:
                cell_texts.append(cell_str)
        if cell_texts:
            row_lines.append(" | ".join(cell_texts))
    return "\n".join(row_lines)


def _iter_docx_container_blocks(
    container_elem: Any,
    document: Any,
    kb_name: str,
    doc_key: str,
    extract_rid: Callable[[str, str], ExtractedImage | None],
    paragraph_cls: Any,
    table_cls: Any,
) -> list[str]:
    """递归展开 Word 容器（body 或 <w:sdtContent>）中的段落与表格块。"""
    blocks: list[str] = []
    for child in container_elem.iterchildren():
        local_name = child.tag.rsplit("}", 1)[-1]
        if local_name == "p":
            para_md = _extract_docx_paragraph(
                paragraph_cls(child, document),
                kb_name=kb_name,
                doc_key=doc_key,
                extract_rid=extract_rid,
                multiline=True,
            )
            if para_md:
                blocks.append(para_md)
        elif local_name == "tbl":
            tbl_md = _extract_docx_table(
                table_cls(child, document),
                document,
                kb_name=kb_name,
                doc_key=doc_key,
                extract_rid=extract_rid,
                paragraph_cls=paragraph_cls,
                table_cls=table_cls,
            )
            if tbl_md:
                blocks.append(tbl_md)
        elif local_name == "sdt":
            sdt_content = next(
                (el for el in child.iterchildren() if el.tag.rsplit("}", 1)[-1] == "sdtContent"),
                None,
            )
            if sdt_content is not None:
                blocks.extend(
                    _iter_docx_container_blocks(
                        sdt_content,
                        document,
                        kb_name=kb_name,
                        doc_key=doc_key,
                        extract_rid=extract_rid,
                        paragraph_cls=paragraph_cls,
                        table_cls=table_cls,
                    )
                )
    return blocks


def _parse_docx_document(
    file_path: Path,
    kb_name: str,
    doc_id: str,
) -> ParsedOfficeDocument:
    """解析 Word (.docx) 文档：按阅读流抽取段落、内容控件、表格与内嵌图片并打桩。"""
    docx_mod = _load_python_docx()
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc_key = make_asset_doc_key(doc_id)
    valid_images: list[ExtractedImage] = []
    used_lower_filenames: set[str] = set()
    rid_cache: dict[str, ExtractedImage | None] = {}
    content_cache: dict[str, ExtractedImage] = {}

    with file_path.open("rb") as fp:
        document = docx_mod.Document(fp)

        def _extract_rid(r_id: str, alt_hint: str) -> ExtractedImage | None:
            if r_id in rid_cache:
                return rid_cache[r_id]
            try:
                part = document.part.related_parts.get(r_id)
                blob = getattr(part, "blob", None) if part is not None else None
            except Exception as exc:
                logger.warning(
                    "读取 Word 图片关系流失败 (%s, rId=%s): %s", file_path.name, r_id, exc
                )
                rid_cache[r_id] = None
                return None

            if not isinstance(blob, (bytes, bytearray)):
                rid_cache[r_id] = None
                return None
            partname = str(getattr(part, "partname", "") or "")
            final_image = _register_unique_image(
                bytes(blob),
                filename_hint=partname,
                alt_hint=alt_hint,
                valid_images=valid_images,
                used_lower_filenames=used_lower_filenames,
                content_cache=content_cache,
            )
            rid_cache[r_id] = final_image
            return final_image

        blocks = _iter_docx_container_blocks(
            document.element.body,
            document,
            kb_name=kb_name,
            doc_key=doc_key,
            extract_rid=_extract_rid,
            paragraph_cls=Paragraph,
            table_cls=Table,
        )

    return ParsedOfficeDocument(
        markdown_text="\n\n".join(blocks).strip(),
        images=valid_images,
    )


def _parse_pdf_page(
    page: Any,
    page_idx: int,
    file_name: str,
    kb_name: str,
    doc_key: str,
    valid_images: list[ExtractedImage],
    used_lower_filenames: set[str],
    content_cache: dict[str, ExtractedImage],
) -> str | None:
    """提取单个 PDF 页面的正文文本与逐张隔离的合格图片锚点。"""
    try:
        raw_page_text = (page.extract_text() or "").replace("\r\n", "\n").strip()
    except Exception as exc:
        logger.warning("提取 PDF 第 %d 页文本失败 (%s): %s", page_idx, file_name, exc)
        raw_page_text = ""

    page_anchors: list[str] = []
    try:
        images_proxy = page.images
        image_count = len(images_proxy)
    except Exception as exc:
        logger.warning("统计 PDF 第 %d 页图片数量失败 (%s): %s", page_idx, file_name, exc)
        images_proxy = None
        image_count = 0

    for img_idx in range(image_count):
        try:
            img_obj = images_proxy[img_idx]
            raw_bytes = bytes(getattr(img_obj, "data", b"") or b"")
            raw_name = str(getattr(img_obj, "name", "") or f"img_{img_idx + 1}")
        except Exception as exc:
            logger.warning(
                "读取 PDF 第 %d 页第 %d 张图片失败 (%s): %s",
                page_idx,
                img_idx + 1,
                file_name,
                exc,
            )
            continue

        stem_hint = Path(raw_name.replace("\\", "/")).stem or f"img_{img_idx + 1}"
        hint = f"page_{page_idx}_{stem_hint}"
        final_image = _register_unique_image(
            raw_bytes,
            filename_hint=hint,
            alt_hint=f"PDF 第 {page_idx} 页插图",
            valid_images=valid_images,
            used_lower_filenames=used_lower_filenames,
            content_cache=content_cache,
        )
        if final_image is not None:
            page_anchors.append(_format_markdown_anchor(kb_name, doc_key, final_image))

    if not raw_page_text and not page_anchors:
        return None
    page_parts: list[str] = [f"## 第 {page_idx} 页"]
    if raw_page_text:
        page_parts.append(raw_page_text)
    if page_anchors:
        page_parts.extend(page_anchors)
    return "\n\n".join(page_parts)


def _parse_pdf_document(
    file_path: Path,
    kb_name: str,
    doc_id: str,
) -> ParsedOfficeDocument:
    """解析 PDF 文档：逐页提取文本与内嵌图片，经过噪点门禁后按页打入 Markdown 锚点。"""
    pypdf_mod = _load_pypdf()
    doc_key = make_asset_doc_key(doc_id)
    valid_images: list[ExtractedImage] = []
    used_lower_filenames: set[str] = set()
    content_cache: dict[str, ExtractedImage] = {}
    blocks: list[str] = []

    with file_path.open("rb") as fp:
        reader = pypdf_mod.PdfReader(fp)
        if getattr(reader, "is_encrypted", False):
            dep_err_type = getattr(getattr(pypdf_mod, "errors", None), "DependencyError", ())
            try:
                decrypt_result = reader.decrypt("")
            except dep_err_type as exc:
                raise ValueError(
                    "PDF 文档使用了强加密算法且缺少 cryptography 解密依赖，无法解析"
                ) from exc
            except Exception as exc:
                raise ValueError("PDF 文档受密码加密保护，无法解析") from exc
            if not decrypt_result:
                raise ValueError("PDF 文档受密码加密保护，无法解析")

        for page_idx, page in enumerate(reader.pages, start=1):
            page_block = _parse_pdf_page(
                page,
                page_idx=page_idx,
                file_name=file_path.name,
                kb_name=kb_name,
                doc_key=doc_key,
                valid_images=valid_images,
                used_lower_filenames=used_lower_filenames,
                content_cache=content_cache,
            )
            if page_block:
                blocks.append(page_block)

    return ParsedOfficeDocument(
        markdown_text="\n\n".join(blocks).strip(),
        images=valid_images,
    )


def parse_office_document(
    file_path: Path,
    kb_name: str,
    doc_id: str,
) -> ParsedOfficeDocument:
    """根据扩展名解析 ``.pdf`` 或 ``.docx`` 文档并返回打桩后的 Markdown 与有效图片资产。

    Args:
        file_path: 待解析的本地二进制文档路径。
        kb_name: 目标知识库名称，用于生成 ``rag://assets/{kb_name}/{doc_key}/{filename}`` 锚点。
        doc_id: 文档唯一标识符，用于生成跨平台安全的资产目录键。

    Returns:
        包含归一化 Markdown 正文与通过噪点门禁的有效图片列表的 ``ParsedOfficeDocument``。

    Raises:
        ImportError: 当所需 ``[office]`` 可选依赖未安装时抛出。
        OSError: 当目标文件不可读或不存在时抛出。
        ValueError: 当扩展名不受支持、文件损坏或受密码保护无法解密时抛出。
    """
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return _parse_pdf_document(file_path, kb_name=kb_name, doc_id=doc_id)
    if ext == ".docx":
        return _parse_docx_document(file_path, kb_name=kb_name, doc_id=doc_id)
    raise ValueError(f"不支持的二进制文档格式: {file_path.suffix}")
