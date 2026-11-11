"""
多模态图片资产网关协调层（本地存储、S3/MinIO 对象存储与 FastMCP Resource 分发）。

统一调度以下核心链路：
1. 文档摄取期资产解析与转写：提取 Markdown/HTML 相对路径图片、按开关安全归档外链图片、
   协调 ``pdf``/``docx`` 内嵌图片，并按 ``storage_type`` 转为 ``rag://assets/...`` 或 HTTPS 直链；
2. 事务级持久化与补偿回滚：屏蔽 ``local``/``s3`` 存储差异并维护 ``image_assets`` 元数据；
3. FastMCP Resource Gateway 读取与动态列举：依据数据库登记清单返回精确图片二进制流与 MIME 类型。

@author Ateng
@since 2026-11-11
"""

import html
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

from mcp_server_rag.assets import (
    ExtractedImage,
    build_asset_resource_uri,
    cleanup_document_local_assets,
    cleanup_kb_local_assets,
    cleanup_stale_document_assets,
    deduplicate_filename,
    inspect_explicit_image_bytes,
    is_within_dir,
    make_asset_doc_key,
    persist_local_image_assets,
    resolve_local_storage_path,
    validate_asset_segment,
)
from mcp_server_rag.config import Settings
from mcp_server_rag.metadata.base import MetadataCatalogAdapter
from mcp_server_rag.parsers import _FENCED_CODE_BLOCK_RE, _HTML_ATTR_RE, _parse_image_tag
from mcp_server_rag.remote_images import SSRFBlockedError, download_safe_remote_image
from mcp_server_rag.s3_storage import S3AssetStorageAdapter, get_s3_storage_adapter
from mcp_server_rag.splitter import ATOMIC_IMAGE_PATTERN

logger = logging.getLogger(__name__)

_LOCAL_ASSET_SUFFIXES: frozenset[str] = frozenset({".md", ".mdx", ".txt", ".html", ".htm", ""})


@dataclass(frozen=True)
class PreparedDocumentAssets:
    """文档摄取前置处理的图片资产计划与转写后正文。"""

    processed_text: str
    images: list[ExtractedImage]
    s3_adapter: S3AssetStorageAdapter | None
    blocked_remote_urls: list[str]
    archived_remote_count: int


def is_text_markup_document(file_path: Path) -> bool:
    """判断文件是否为可含相对/外链图片标签的富文本或 Markdown/HTML 文档。"""
    return file_path.suffix.lower() in _LOCAL_ASSET_SUFFIXES


def _validate_storage_type(storage_type: str) -> str:
    """校验当前运行时存储模式是否受支持。"""
    normalized = storage_type.strip().lower() or "local"
    if normalized not in {"local", "s3"}:
        raise ValueError(f"不支持的图片存储网关模式 '{storage_type}' (合法值: 'local', 's3')")
    return normalized


def _is_local_relative_image_ref(raw_url: str) -> bool:
    """判断图片 URL 是否为安全的相对路径（排除协议方案、根目录与盘符绝对路径）。"""
    clean = unquote(raw_url.strip())
    if not clean or clean.startswith(("#", "/", "\\")):
        return False
    parsed = urlparse(clean)
    if parsed.scheme:
        return False
    # 拦截 Windows 盘符绝对路径（如 C:\a.png 或 D:/a.png）
    if len(clean) >= 2 and clean[1] == ":":
        return False
    return True


def _resolve_relative_image_file(base_dir: Path, raw_url: str) -> Path | None:
    """按文档所在目录解析相对路径本地图片文件，严格限制在 base_dir 内部防止路径穿越。"""
    parsed = urlparse(raw_url.strip())
    rel_part = unquote(parsed.path).replace("\\", "/").lstrip()
    if not rel_part or "\x00" in rel_part:
        return None
    candidate = (base_dir / Path(rel_part)).resolve()
    if not is_within_dir(candidate, base_dir.resolve()):
        return None
    if not candidate.is_file():
        return None
    return candidate


def _rewrite_matched_image_tag(match: re.Match[str], new_url: str) -> str:
    """在保留原始 alt/title 与 HTML 属性前提下将图片标签 URL 替换为目标地址。"""
    raw_tag = match.group(0)
    if raw_tag.startswith("!"):
        alt = match.group("alt") or ""
        dq_title = match.group("dq_title")
        sq_title = match.group("sq_title")
        title_part = ""
        if dq_title is not None:
            title_part = f' "{dq_title}"'
        elif sq_title is not None:
            title_part = f" '{sq_title}'"
        formatted_url = f"<{new_url}>" if " " in new_url else new_url
        return f"![{alt}]({formatted_url}{title_part})"

    # HTML <img ...> 标签：精确改写 src（若仅含 data-src 则改写 data-src）
    escaped_url = html.escape(new_url, quote=True)
    has_src = bool(re.search(r"\bsrc\s*=", raw_tag, re.IGNORECASE))

    def _replace_attr(attr_match: re.Match[str]) -> str:
        attr_name = attr_match.group(1)
        attr_lower = attr_name.lower()
        target_attr = "src" if has_src else "data-src"
        if attr_lower == target_attr:
            return f'{attr_name}="{escaped_url}"'
        return attr_match.group(0)

    return _HTML_ATTR_RE.sub(_replace_attr, raw_tag)


def _transform_outside_fenced_blocks(
    text: str,
    transform_fn: Callable[[re.Match[str]], str],
) -> str:
    """仅对围栏代码块之外的图片标签应用改写逻辑，保持示例代码原样不变。"""
    if not text or ("!" not in text and "<img" not in text.lower()):
        return text

    parts: list[str] = []
    last_end = 0
    for block_match in _FENCED_CODE_BLOCK_RE.finditer(text):
        start, end = block_match.span()
        if start > last_end:
            parts.append(ATOMIC_IMAGE_PATTERN.sub(transform_fn, text[last_end:start]))
        parts.append(block_match.group(0))
        last_end = end

    if last_end < len(text):
        parts.append(ATOMIC_IMAGE_PATTERN.sub(transform_fn, text[last_end:]))
    return "".join(parts)


def prepare_document_assets(
    settings: Settings,
    kb_name: str,
    doc_id: str,
    text_content: str,
    *,
    base_dir: Path | None = None,
    source_url: str | None = None,
    pre_extracted_images: list[ExtractedImage] | None = None,
    s3_client: Any | None = None,
) -> PreparedDocumentAssets:
    """收集并归一化文档内的图片资产（内嵌图/局部相对路径图/安全归档外链），完成正文引用转写。

    Args:
        settings: 当前运行时配置实例。
        kb_name: 目标知识库名称。
        doc_id: 文档唯一标识符。
        text_content: 归一化后的 Markdown/纯文本正文内容。
        base_dir: 本地文件摄取时的文件所在目录（用于解析相对路径本地图片）。
        source_url: 在线网页摄取时的来源页面 URL（用于解析网页相对图链）。
        pre_extracted_images: 已由二进制文档解析器提取并打桩的图片资产列表。
        s3_client: 可选自定义 S3 客户端（主要用于测试）。

    Returns:
        包含转写后正文、待落盘图片清单与已校验 S3 适配器的 ``PreparedDocumentAssets``。
    """
    storage_type = _validate_storage_type(settings.storage_type)
    doc_key = make_asset_doc_key(doc_id)

    images: list[ExtractedImage] = list(pre_extracted_images or [])
    used_lower_filenames: set[str] = {img.filename.lower() for img in images}
    pre_extracted_uri_to_filename: dict[str, str] = {
        build_asset_resource_uri(kb_name, doc_key, img.filename): img.filename for img in images
    }

    local_path_cache: dict[Path, str | None] = {}
    remote_url_cache: dict[str, str | None] = {}
    blocked_remote_urls: list[str] = []
    archived_remote_count = 0

    def _collect_and_map(match: re.Match[str]) -> str | None:
        nonlocal archived_remote_count
        alt, raw_url = _parse_image_tag(match)
        if not raw_url or raw_url.startswith("#") or raw_url.lower().startswith("javascript:"):
            return None

        # 1. 已提取打桩的内部 rag://assets 引用（来自 PDF/Word 解析）
        if raw_url in pre_extracted_uri_to_filename:
            return pre_extracted_uri_to_filename[raw_url]

        # 2. 本地相对路径图片（当存在本地文档基准目录时提取）
        if base_dir is not None and _is_local_relative_image_ref(raw_url):
            img_path = _resolve_relative_image_file(base_dir, raw_url)
            if img_path is None:
                return None
            if img_path in local_path_cache:
                return local_path_cache[img_path]
            try:
                if img_path.stat().st_size > settings.remote_image_max_bytes:
                    logger.warning(
                        "本地相对图片体积超过上限 (%d bytes): %s",
                        settings.remote_image_max_bytes,
                        img_path,
                    )
                    local_path_cache[img_path] = None
                    return None
                raw_bytes = img_path.read_bytes()
            except OSError as exc:
                logger.warning("读取本地相对图片失败 (%s): %s", img_path, exc)
                local_path_cache[img_path] = None
                return None
            inspected = inspect_explicit_image_bytes(
                raw_bytes,
                filename_hint=img_path.name,
                alt=alt,
            )
            if inspected is None:
                local_path_cache[img_path] = None
                return None
            unique_name = deduplicate_filename(inspected.filename, used_lower_filenames)
            normalized_img = ExtractedImage(
                filename=unique_name,
                data=inspected.data,
                mime_type=inspected.mime_type,
                alt=inspected.alt,
                width=inspected.width,
                height=inspected.height,
                byte_size=inspected.byte_size,
            )
            images.append(normalized_img)
            local_path_cache[img_path] = unique_name
            return unique_name

        # 3. 远程外链（仅在显式开启 MCP_RAG_ARCHIVE_REMOTE_IMAGES=true 时抓取归档）
        if not settings.archive_remote_images:
            return None

        target_url = raw_url.strip()
        if source_url and not urlparse(target_url).scheme:
            target_url = urljoin(source_url, target_url)

        if target_url in remote_url_cache:
            return remote_url_cache[target_url]

        try:
            remote_img = download_safe_remote_image(
                target_url,
                alt=alt,
                timeout_seconds=settings.remote_image_timeout_seconds,
                max_bytes=settings.remote_image_max_bytes,
            )
        except SSRFBlockedError as exc:
            logger.warning("SSRF 防护拦截远程图片外链 (%s): %s", target_url, exc)
            blocked_remote_urls.append(target_url)
            remote_url_cache[target_url] = None
            return None

        if remote_img is None:
            remote_url_cache[target_url] = None
            return None

        unique_name = deduplicate_filename(remote_img.filename, used_lower_filenames)
        normalized_remote = ExtractedImage(
            filename=unique_name,
            data=remote_img.data,
            mime_type=remote_img.mime_type,
            alt=remote_img.alt,
            width=remote_img.width,
            height=remote_img.height,
            byte_size=remote_img.byte_size,
        )
        images.append(normalized_remote)
        remote_url_cache[target_url] = unique_name
        archived_remote_count += 1
        return unique_name

    # 第一遍：收集有效图片并生成待替换 URL 映射
    s3_adapter: S3AssetStorageAdapter | None = None
    url_filename_pairs: list[tuple[re.Match[str], str]] = []

    def _collect_pass(match: re.Match[str]) -> str:
        target_filename = _collect_and_map(match)
        if target_filename is not None:
            url_filename_pairs.append((match, target_filename))
        return match.group(0)

    _transform_outside_fenced_blocks(text_content, _collect_pass)

    if storage_type == "s3" and images:
        s3_adapter = get_s3_storage_adapter(settings, client=s3_client)

    def _rewrite_pass(match: re.Match[str]) -> str:
        target_filename = _collect_and_map_cached(match)
        if target_filename is None:
            return match.group(0)
        if s3_adapter is not None:
            object_key = s3_adapter.build_object_key(kb_name, doc_key, target_filename)
            new_url = s3_adapter.build_public_url(object_key)
        else:
            new_url = build_asset_resource_uri(kb_name, doc_key, target_filename)
        return _rewrite_matched_image_tag(match, new_url)

    def _collect_and_map_cached(match: re.Match[str]) -> str | None:
        _, raw_url = _parse_image_tag(match)
        if not raw_url:
            return None
        if raw_url in pre_extracted_uri_to_filename:
            return pre_extracted_uri_to_filename[raw_url]
        if base_dir is not None and _is_local_relative_image_ref(raw_url):
            img_path = _resolve_relative_image_file(base_dir, raw_url)
            if img_path is not None:
                return local_path_cache.get(img_path)
        if settings.archive_remote_images:
            target_url = raw_url.strip()
            if source_url and not urlparse(target_url).scheme:
                target_url = urljoin(source_url, target_url)
            return remote_url_cache.get(target_url)
        return None

    rewritten_text = (
        _transform_outside_fenced_blocks(text_content, _rewrite_pass)
        if url_filename_pairs
        else text_content
    )

    return PreparedDocumentAssets(
        processed_text=rewritten_text,
        images=images,
        s3_adapter=s3_adapter,
        blocked_remote_urls=blocked_remote_urls,
        archived_remote_count=archived_remote_count,
    )


def persist_and_register_document_assets(
    settings: Settings,
    catalog: MetadataCatalogAdapter,
    kb_name: str,
    doc_id: str,
    prepared: PreparedDocumentAssets,
) -> list[dict[str, Any]]:
    """按配置存储后端落盘或上传图片资产并写入 Metadata Catalog（异常自动回滚文件/对象）。"""
    if not prepared.images:
        return []

    if prepared.s3_adapter is not None:
        asset_records, _ = prepared.s3_adapter.upload_image_assets(
            kb_name,
            doc_id,
            prepared.images,
        )
        try:
            catalog.register_image_assets(kb_name, doc_id, asset_records)
        except Exception:
            prepared.s3_adapter.delete_assets(asset_records)
            raise
        return asset_records

    written_assets = persist_local_image_assets(
        settings.data_dir,
        kb_name,
        doc_id,
        prepared.images,
    )
    try:
        catalog.register_image_assets(kb_name, doc_id, written_assets)
    except Exception:
        cleanup_document_local_assets(settings.data_dir, kb_name, doc_id, written_assets)
        raise
    return written_assets


def _try_get_s3_adapter_for_cleanup(
    settings: Settings,
    assets: list[dict[str, Any]] | None,
) -> S3AssetStorageAdapter | None:
    """仅当资产清单存在 ``s3://`` 对象 URI 时懒加载 S3 适配器执行远端清理。"""
    has_s3 = any(
        str(asset.get("storage_uri") or "").strip().startswith("s3://") for asset in (assets or [])
    )
    if not has_s3:
        return None
    try:
        return get_s3_storage_adapter(settings)
    except Exception as exc:
        logger.warning("初始化 S3 清理适配器失败，跳过远端对象删除: %s", exc)
        return None


def cleanup_stale_gateway_assets(
    settings: Settings,
    old_assets: list[dict[str, Any]],
    new_assets: list[dict[str, Any]],
) -> None:
    """文档更新后统一清理陈旧的本地物理图片与 S3 远端对象。"""
    if not old_assets:
        return
    cleanup_stale_document_assets(settings.data_dir, old_assets, new_assets)
    s3_adapter = _try_get_s3_adapter_for_cleanup(settings, old_assets)
    if s3_adapter is not None:
        s3_adapter.delete_stale_assets(old_assets, new_assets)


def cleanup_document_gateway_assets(
    settings: Settings,
    kb_name: str,
    doc_id: str,
    assets: list[dict[str, Any]] | None = None,
) -> None:
    """级联清理单篇文档的本地与 S3 图片资产。"""
    cleanup_document_local_assets(settings.data_dir, kb_name, doc_id, assets)
    s3_adapter = _try_get_s3_adapter_for_cleanup(settings, assets)
    if s3_adapter is not None:
        s3_adapter.delete_assets(assets)


def cleanup_kb_gateway_assets(
    settings: Settings,
    kb_name: str,
    assets: list[dict[str, Any]] | None = None,
) -> None:
    """级联清理整座知识库的本地与 S3 图片资产。"""
    cleanup_kb_local_assets(settings.data_dir, kb_name, assets)
    s3_adapter = _try_get_s3_adapter_for_cleanup(settings, assets)
    if s3_adapter is not None:
        s3_adapter.delete_assets(assets)


def read_gateway_image_asset(
    settings: Settings,
    catalog: MetadataCatalogAdapter,
    kb_name: str,
    doc_id: str,
    filename: str,
    *,
    s3_client: Any | None = None,
) -> tuple[bytes, str]:
    """按 FastMCP Resource 参数查询 Metadata Catalog 并读取底层图片二进制流与 MIME 类型。

    Raises:
        ValueError: 当路径参数含目录穿越或非法段时抛出。
        FileNotFoundError: 当登记记录或底层物理文件/对象不存在时抛出。
    """
    safe_kb = validate_asset_segment(kb_name, "kb_name")
    safe_doc = validate_asset_segment(doc_id, "doc_id")
    safe_filename = validate_asset_segment(filename, "filename")

    asset = catalog.get_image_asset(safe_kb, safe_doc, safe_filename)
    if asset is None:
        raise FileNotFoundError(
            f"未找到知识库 '{safe_kb}' 文档 '{safe_doc}' 的图片资产 '{safe_filename}'"
        )

    storage_uri = str(asset.get("storage_uri") or "").strip()
    mime_type = str(asset.get("mime_type") or "application/octet-stream")

    if storage_uri.startswith("s3://"):
        adapter = get_s3_storage_adapter(settings, client=s3_client)
        try:
            payload = adapter.read_bytes(
                storage_uri,
                kb_name=safe_kb,
                doc_id=safe_doc,
                filename=safe_filename,
            )
        except (KeyError, FileNotFoundError) as exc:
            raise FileNotFoundError(f"S3 图片资产不存在: {storage_uri}") from exc
        return payload, mime_type

    local_path = resolve_local_storage_path(settings.data_dir, storage_uri)
    if local_path is None or not local_path.is_file():
        raise FileNotFoundError(f"本地图片资产文件不存在: {storage_uri}")
    return local_path.read_bytes(), mime_type


def list_gateway_asset_descriptors(catalog: MetadataCatalogAdapter) -> list[dict[str, Any]]:
    """列举全部知识库下已登记的标准化 MCP Resource 图片资产描述条目。"""
    descriptors: list[dict[str, Any]] = []
    for kb in catalog.list_kbs():
        kb_name = str(kb["name"])
        for asset in catalog.list_image_assets(kb_name):
            doc_id = str(asset["doc_id"])
            filename = str(asset["filename"])
            doc_key = make_asset_doc_key(doc_id)
            descriptors.append(
                {
                    "uri": build_asset_resource_uri(kb_name, doc_key, filename),
                    "name": filename,
                    "title": f"{kb_name}/{filename}",
                    "description": f"知识库 '{kb_name}' 文档 '{doc_id}' 的图片资产 ({filename})",
                    "mime_type": str(asset.get("mime_type") or "application/octet-stream"),
                }
            )
    return descriptors
