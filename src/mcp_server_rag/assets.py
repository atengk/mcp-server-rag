"""
多模态图片资产噪点门禁与本地持久化存储管理模块。

提供内存级图片尺寸与体积噪点过滤（Image Noise Filter）、安全的跨平台资产键与
`rag://assets/...` 引用 URI 构建，以及本地物理图片落盘与级联清理能力。

@author Ateng
@since 2026-11-11
"""

import hashlib
import io
import logging
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Image Noise Filter 物理门禁阈值：过滤装饰图标、页眉 Logo 与细长分割线
MIN_IMAGE_DIMENSION_PX: int = 100
MIN_IMAGE_BYTES: int = 5 * 1024
ASSET_DOC_KEY_LENGTH: int = 16

_ASSET_UUID_NAMESPACE: uuid.UUID = uuid.UUID("7f3d5a60-9b4a-4e82-9a1c-4d7c1b8e2f10")
_WINDOWS_RESERVED_NAMES: frozenset[str] = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
)

_FORMAT_METADATA: dict[str, tuple[str, str]] = {
    "PNG": (".png", "image/png"),
    "JPEG": (".jpg", "image/jpeg"),
    "JPG": (".jpg", "image/jpeg"),
    "WEBP": (".webp", "image/webp"),
    "GIF": (".gif", "image/gif"),
    "BMP": (".bmp", "image/bmp"),
    "TIFF": (".tiff", "image/tiff"),
    "JPEG2000": (".jp2", "image/jp2"),
    "J2K": (".jp2", "image/jp2"),
}


@dataclass(frozen=True)
class ExtractedImage:
    """通过噪点门禁校验的有效内嵌图片实体载荷。"""

    filename: str
    data: bytes
    mime_type: str
    alt: str
    width: int
    height: int
    byte_size: int


def _load_pil_image_module() -> Any:
    """惰性加载 Pillow Image 模块，避免核心包导入期强制耦合。"""
    try:
        from PIL import Image

        return Image
    except ImportError as exc:
        raise ImportError(
            "未检测到 Pillow 依赖，无法解码并校验图片资产。"
            "请运行 `pip install pillow>=10.0.0` 或 `pip install atengk-mcp-server-rag[office]`。"
        ) from exc


def validate_asset_segment(value: str, label: str) -> str:
    """校验用于拼接目录或对象键的单级路径段，拦截空值、路径分隔符与目录穿越段。"""
    clean = value.strip()
    if (
        not clean
        or clean in {".", ".."}
        or "/" in clean
        or "\\" in clean
        or "\x00" in clean
        or ":" in clean
    ):
        raise ValueError(f"非法的单级路径参数 {label}: {value!r}")
    return clean


def _validate_path_segment(value: str, label: str) -> str:
    """兼容内部调用的单级路径校验别名。"""
    return validate_asset_segment(value, label)


def make_asset_doc_key(doc_id: str) -> str:
    """将任意文档标识映射为无路径分隔符、跨平台安全且确定的 16 位十六进制目录键。

    Args:
        doc_id: 文档唯一标识符（如本地解析后的绝对路径或 URL）。

    Returns:
        长度为 16 个字符的小写十六进制 SHA-256 前缀。
    """
    return hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:ASSET_DOC_KEY_LENGTH]


def make_asset_id(kb_name: str, doc_key: str, filename: str) -> str:
    """生成基于 ``(kb_name, doc_key, filename)`` 的确定性 UUID5 资产主键。

    Args:
        kb_name: 所属知识库名称。
        doc_key: 经 ``make_asset_doc_key`` 生成的单级文档目录键。
        filename: 安全清洗后的资产文件名。

    Returns:
        标准 UUID5 字符串，同时用于 DB 主键去重与 Resource 单段路由反查。
    """
    seed = f"{kb_name}\n{doc_key}\n{filename}"
    return str(uuid.uuid5(_ASSET_UUID_NAMESPACE, seed))


def build_asset_resource_uri(kb_name: str, doc_key: str, filename: str) -> str:
    """构建标准 MCP Resource 图片引用 URI（``rag://assets/{kb_name}/{doc_key}/{filename}``）。

    Args:
        kb_name: 所属知识库名称。
        doc_key: 单级文档键。
        filename: 图片文件名。

    Returns:
        标准化 ``rag://assets/...`` 引用链接。
    """
    return f"rag://assets/{kb_name}/{doc_key}/{filename}"


def build_asset_storage_uri(kb_name: str, doc_key: str, filename: str) -> str:
    """构建相对 ``data_dir`` 的跨平台 POSIX 物理存储路径（``assets/{kb}/{doc_key}/{file}``）。

    Args:
        kb_name: 所属知识库名称。
        doc_key: 单级文档键。
        filename: 图片文件名。

    Returns:
        基于正斜杠 ``/`` 拼接的相对存储路径。
    """
    return f"assets/{kb_name}/{doc_key}/{filename}"


def sanitize_asset_filename(filename_hint: str, ext: str) -> str:
    """将来源文件名与扩展名清洗为安全、跨平台无冲突的单级图片文件名。

    Args:
        filename_hint: 原始图片文件名提示。
        ext: 期望文件扩展名（如 ``".png"``）。

    Returns:
        过滤了路径穿越字符、特殊符号及 Windows 保留设备名的安全文件名。
    """
    base_name = Path(filename_hint.replace("\\", "/")).name if filename_hint else ""
    stem = Path(base_name).stem if base_name else "image"
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("._-") or "image"
    if safe_stem.lower() in _WINDOWS_RESERVED_NAMES:
        safe_stem = f"img_{safe_stem}"
    clean_ext = re.sub(r"[^a-z0-9]+", "", ext.lower()) or "png"
    return f"{safe_stem}.{clean_ext}"


def deduplicate_filename(candidate: str, used_lower_filenames: set[str]) -> str:
    """为同一文档内多张有效图片分配大小写不敏感的唯一文件名，防止 Windows/macOS 相互覆写。

    Args:
        candidate: 期望分配的文件名。
        used_lower_filenames: 已占用的小写文件名集合（函数会就地更新该集合）。

    Returns:
        不与既有集合发生大小写冲突的唯一文件名。
    """
    if candidate.lower() not in used_lower_filenames:
        used_lower_filenames.add(candidate.lower())
        return candidate

    stem = Path(candidate).stem
    suffix = Path(candidate).suffix
    seq = 2
    while True:
        numbered = f"{stem}_{seq}{suffix}"
        if numbered.lower() not in used_lower_filenames:
            used_lower_filenames.add(numbered.lower())
            return numbered
        seq += 1


def _transcode_image_to_png(pil_image_mod: Any, raw_data: bytes) -> bytes | None:
    """将非主流栅格格式图片在内存中转码为标准 PNG 字节，保证后缀与魔数一致。"""
    try:
        with pil_image_mod.open(io.BytesIO(raw_data)) as src_img:
            if src_img.mode in {"RGBA", "LA"} or (
                src_img.mode == "P" and "transparency" in src_img.info
            ):
                converted = src_img.convert("RGBA")
            elif src_img.mode != "RGB":
                converted = src_img.convert("RGB")
            else:
                converted = src_img
            out_buf = io.BytesIO()
            converted.save(out_buf, format="PNG")
            return out_buf.getvalue()
    except Exception as exc:
        logger.debug("非常规格式图片转码 PNG 失败，已丢弃: %s", exc)
        return None


def _decode_and_normalize_image_bytes(
    data: bytes,
    filename_hint: str = "",
    alt: str = "",
    *,
    min_dimension: int = 1,
    min_bytes: int = 1,
) -> ExtractedImage | None:
    """校验图片字节完整性并在尺寸与体积达到门禁时返回标准化 ``ExtractedImage``。"""
    if not data or len(data) < min_bytes:
        return None

    pil_image_mod = _load_pil_image_module()
    try:
        with pil_image_mod.open(io.BytesIO(data)) as img:
            width, height = img.size
            fmt = (img.format or "").upper()
            img.verify()
    except Exception as exc:
        logger.debug("跳过无法解码或已损坏的图片流 (%s): %s", filename_hint or "<unnamed>", exc)
        return None

    if width < min_dimension or height < min_dimension:
        return None

    final_data = data
    if fmt in _FORMAT_METADATA:
        ext, mime_type = _FORMAT_METADATA[fmt]
    else:
        transcoded = _transcode_image_to_png(pil_image_mod, data)
        if not transcoded or len(transcoded) < min_bytes:
            return None
        final_data = transcoded
        ext, mime_type = ".png", "image/png"

    safe_filename = sanitize_asset_filename(filename_hint, ext)
    clean_alt = re.sub(r"[\r\n]+", " ", alt).strip() or "文档插图"

    return ExtractedImage(
        filename=safe_filename,
        data=final_data,
        mime_type=mime_type,
        alt=clean_alt,
        width=width,
        height=height,
        byte_size=len(final_data),
    )


def inspect_image_bytes(
    data: bytes,
    filename_hint: str = "",
    alt: str = "",
) -> ExtractedImage | None:
    """在内存中对图片二进制字节执行完整性校验、尺寸与体积噪点门禁过滤。

    任意宽度 < 100px、高度 < 100px、字节数 < 5KB 或损坏截断的图元均会被丢弃；
    对非常规栅格格式会自动在内存中转码为标准 PNG，确保 MIME 与实际魔数一致。

    Args:
        data: 原始图片二进制字节流。
        filename_hint: 原始内嵌文件名提示（仅用于提取安全基础名称）。
        alt: 图片语义替代描述文本。

    Returns:
        校验通过返回 ``ExtractedImage`` 实例；未通过噪点门禁或解码失败返回 ``None``。
    """
    return _decode_and_normalize_image_bytes(
        data,
        filename_hint=filename_hint,
        alt=alt,
        min_dimension=MIN_IMAGE_DIMENSION_PX,
        min_bytes=MIN_IMAGE_BYTES,
    )


def inspect_explicit_image_bytes(
    data: bytes,
    filename_hint: str = "",
    alt: str = "",
) -> ExtractedImage | None:
    """校验文档显式引用图片（如 Markdown 本地相对图片与远程图片）的合法性与 MIME 类型。

    仅校验图片完整性与可解码性（>= 1x1 像素），不应用二进制文档噪点尺寸过滤阈值。

    Args:
        data: 图片二进制字节流。
        filename_hint: 原始图片文件名提示。
        alt: 图片替代文本描述。

    Returns:
        有效栅格图片返回 ``ExtractedImage``，无法解码返回 ``None``。
    """
    return _decode_and_normalize_image_bytes(
        data,
        filename_hint=filename_hint,
        alt=alt,
        min_dimension=1,
        min_bytes=1,
    )


def is_within_dir(target: Path, root: Path) -> bool:
    """校验目标路径解析后为根目录的严格真子路径（不含根目录自身），防御路径穿越误删。"""
    try:
        resolved_target = target.resolve()
        resolved_root = root.resolve()
        return resolved_target != resolved_root and resolved_target.is_relative_to(resolved_root)
    except OSError:
        return False


def _is_within_dir(target: Path, root: Path) -> bool:
    """兼容内部调用的路径范围判定别名。"""
    return is_within_dir(target, root)


def resolve_local_storage_path(data_dir: Path, storage_uri: str) -> Path | None:
    """将登记的相对路径或 ``rag://assets/...`` URI 解析为 ``data_dir/assets`` 子路径。"""
    clean_uri = storage_uri.strip().replace("\\", "/")
    if not clean_uri or "\x00" in clean_uri:
        return None

    assets_root = data_dir / "assets"
    if clean_uri.startswith("rag://assets/"):
        rel_part = clean_uri.removeprefix("rag://assets/").lstrip("/")
        if not rel_part or ":" in rel_part:
            return None
        rel_path = Path(rel_part)
        if ".." in rel_path.parts or rel_path.is_absolute():
            return None
        candidate = assets_root / rel_path
    elif "://" in clean_uri or ":" in clean_uri:
        return None
    else:
        rel_path = Path(clean_uri)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            return None
        candidate = data_dir / rel_path

    if not _is_within_dir(candidate, assets_root):
        return None
    return candidate


_resolve_local_storage_path = resolve_local_storage_path


def _prune_empty_dirs(start_dir: Path, stop_root: Path) -> None:
    """自底向上清理空目录，直至并保留 ``stop_root``（如 ``data_dir/assets``）。"""
    curr = start_dir.resolve()
    root_resolved = stop_root.resolve()
    while curr != root_resolved and _is_within_dir(curr, root_resolved):
        try:
            if curr.is_dir() and not any(curr.iterdir()):
                curr.rmdir()
            else:
                break
        except OSError:
            break
        curr = curr.parent


def _cleanup_registered_assets(
    data_dir: Path,
    assets_root: Path,
    assets: list[dict[str, Any]] | None,
) -> None:
    """按登记清单安全删除本地资产物理文件并修剪空父目录。"""
    for asset in assets or []:
        storage_uri = str(asset.get("storage_uri") or "")
        local_path = resolve_local_storage_path(data_dir, storage_uri)
        if local_path is not None and local_path.is_file():
            try:
                local_path.unlink(missing_ok=True)
                _prune_empty_dirs(local_path.parent, assets_root)
            except OSError as exc:
                logger.warning("删除本地图片资产文件失败 (%s): %s", local_path, exc)


def persist_local_image_assets(
    data_dir: Path,
    kb_name: str,
    doc_id: str,
    images: list[ExtractedImage],
) -> list[dict[str, Any]]:
    """将已通过门禁的图片资产写入本地资产目录并生成元数据登记载荷（失败自动回滚已写文件）。

    Args:
        data_dir: 服务数据根目录。
        kb_name: 目标知识库名称。
        doc_id: 所属文档唯一标识。
        images: 已通过噪点门禁且完成去重命名的有效图片列表。

    Returns:
        可直接传入 ``MetadataCatalogAdapter.register_image_assets`` 的资产字典列表。

    Raises:
        ValueError: ``kb_name`` 或文件名不合法、或批次内存在大小写重名冲突时抛出。
    """
    if not images:
        return []

    safe_kb = _validate_path_segment(kb_name, "kb_name")
    doc_key = make_asset_doc_key(doc_id)
    assets_root = (data_dir / "assets").resolve()
    doc_dir = (assets_root / safe_kb / doc_key).resolve()
    if not _is_within_dir(doc_dir, assets_root):
        raise ValueError(f"非法的本地资产存储目录: {doc_dir}")
    doc_dir.mkdir(parents=True, exist_ok=True)

    asset_records: list[dict[str, Any]] = []
    written_paths: list[Path] = []
    seen_lower_names: set[str] = set()

    try:
        for image in images:
            safe_filename = _validate_path_segment(image.filename, "filename")
            lower_name = safe_filename.lower()
            if lower_name in seen_lower_names:
                raise ValueError(f"图片资产存在大小写或名称文件名冲突: {safe_filename}")
            seen_lower_names.add(lower_name)

            target_path = (doc_dir / safe_filename).resolve()
            if not _is_within_dir(target_path, doc_dir):
                raise ValueError(f"非法的图片资产文件名: {safe_filename}")
            target_path.write_bytes(image.data)
            written_paths.append(target_path)

            asset_records.append(
                {
                    "asset_id": make_asset_id(safe_kb, doc_key, safe_filename),
                    "filename": safe_filename,
                    "storage_uri": build_asset_storage_uri(safe_kb, doc_key, safe_filename),
                    "mime_type": image.mime_type,
                    "byte_size": image.byte_size,
                }
            )
    except Exception:
        for partial_path in written_paths:
            try:
                partial_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("回滚部分写入的本地图片文件失败 (%s): %s", partial_path, exc)
        _prune_empty_dirs(doc_dir, assets_root)
        raise

    return asset_records


def cleanup_stale_document_assets(
    data_dir: Path,
    old_assets: list[dict[str, Any]],
    new_assets: list[dict[str, Any]],
) -> None:
    """更新文档后清理新版本不再引用的陈旧物理图片资产文件。"""
    kept_uris = {str(asset.get("storage_uri") or "") for asset in new_assets}
    stale_assets = [
        asset for asset in old_assets if str(asset.get("storage_uri") or "") not in kept_uris
    ]
    if stale_assets:
        _cleanup_registered_assets(data_dir, data_dir / "assets", stale_assets)


def cleanup_document_local_assets(
    data_dir: Path,
    kb_name: str,
    doc_id: str,
    assets: list[dict[str, Any]] | None = None,
) -> None:
    """级联清理指定文档关联的所有本地物理图片文件及空目录。

    Args:
        data_dir: 服务数据根目录。
        kb_name: 目标知识库名称。
        doc_id: 文档唯一标识。
        assets: 可选的已登记资产列表。
    """
    assets_root = data_dir / "assets"
    _cleanup_registered_assets(data_dir, assets_root, assets)

    try:
        safe_kb = _validate_path_segment(kb_name, "kb_name")
    except ValueError as exc:
        logger.warning("跳过无效知识库目录资产清理: %s", exc)
        return

    doc_key = make_asset_doc_key(doc_id)
    doc_dir = assets_root / safe_kb / doc_key
    if doc_dir.exists() and _is_within_dir(doc_dir, assets_root):
        try:
            shutil.rmtree(doc_dir, ignore_errors=False)
            _prune_empty_dirs(doc_dir.parent, assets_root)
        except OSError as exc:
            logger.warning("删除文档本地图片目录失败 (%s): %s", doc_dir, exc)


def cleanup_kb_local_assets(
    data_dir: Path,
    kb_name: str,
    assets: list[dict[str, Any]] | None = None,
) -> None:
    """级联清理指定知识库关联的所有本地物理图片文件与资产子目录。

    Args:
        data_dir: 服务数据根目录。
        kb_name: 目标知识库名称。
        assets: 可选的已登记资产列表。
    """
    assets_root = data_dir / "assets"
    _cleanup_registered_assets(data_dir, assets_root, assets)

    try:
        safe_kb = _validate_path_segment(kb_name, "kb_name")
    except ValueError as exc:
        logger.warning("跳过无效知识库目录资产清理: %s", exc)
        return

    kb_dir = assets_root / safe_kb
    if kb_dir.exists() and _is_within_dir(kb_dir, assets_root):
        try:
            shutil.rmtree(kb_dir, ignore_errors=False)
        except OSError as exc:
            logger.warning("删除知识库本地图片目录失败 (%s): %s", kb_dir, exc)
