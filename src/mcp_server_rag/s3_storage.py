"""
S3 / MinIO / 阿里云 OSS 兼容对象存储资产网关适配器。

支持按需惰性加载可选依赖 ``[s3]``（``boto3``），在 ``MCP_RAG_STORAGE_TYPE=s3`` 模式下提供
多模态图片资产的上传、HTTPS/公共直链生成、二进制读取与原子级联清理能力。

@author Ateng
@since 2026-11-11
"""

import importlib
import logging
from typing import Any

from mcp_server_rag.assets import (
    ExtractedImage,
    make_asset_doc_key,
    make_asset_id,
    validate_asset_segment,
)
from mcp_server_rag.config import Settings

logger = logging.getLogger(__name__)


def _load_boto3() -> Any:
    """惰性加载 ``boto3`` 模块；缺失 ``[s3]`` 可选依赖时抛出清晰的中文安装提示。"""
    try:
        return importlib.import_module("boto3")
    except ImportError as exc:
        raise ImportError(
            "使用 S3/MinIO/OSS 对象存储网关 (MCP_RAG_STORAGE_TYPE=s3) 需要可选依赖 `[s3]`。"
            '请运行 `pip install atengk-mcp-server-rag[s3]` 或 `uv pip install "boto3>=1.34.0"`。'
        ) from exc


class S3AssetStorageAdapter:
    """面向 S3 / MinIO / OSS 协议兼容对象存储的图片资产网关适配器。"""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str = "",
        region_name: str = "us-east-1",
        access_key: str = "",
        secret_key: str = "",
        prefix: str = "assets",
        public_base_url: str = "",
        timeout_seconds: int = 30,
        client: Any | None = None,
    ) -> None:
        """初始化对象存储适配器并构造或绑定 S3 客户端。

        Args:
            bucket: 目标 S3 存储桶名称（必填非空）。
            endpoint_url: 自定义兼容服务端点 URL（如 MinIO 或 OSS 地址）。
            region_name: 存储桶区域标识，默认 ``us-east-1``。
            access_key: 访问凭证 Access Key ID（可选，省略时回退标准 IAM/环境变量链路）。
            secret_key: 访问凭证 Secret Access Key（可选）。
            prefix: 存储桶内部路径前缀，默认 ``assets``。
            public_base_url: 对外暴露的公共/CDN/HTTPS 基础直链前缀（可选）。
            timeout_seconds: 网络连接与读取超时秒数，必须 >= 1。
            client: 可选注入的兼容 boto3 S3 client 实例（主要用于测试或自定义装配）。

        Raises:
            ValueError: ``bucket`` 为空或 ``timeout_seconds < 1`` 时抛出。
            ImportError: 未注入 ``client`` 且缺少可选依赖 ``[s3]``（``boto3``）时抛出。
        """
        clean_bucket = bucket.strip()
        if not clean_bucket:
            raise ValueError("配置 MCP_RAG_STORAGE_TYPE=s3 时必须设置非空的 MCP_RAG_S3_BUCKET")
        if timeout_seconds < 1:
            raise ValueError(f"S3 超时时间必须 >= 1 秒，当前值: {timeout_seconds}")

        self.bucket = clean_bucket
        self.endpoint_url = endpoint_url.strip().rstrip("/")
        self.region_name = region_name.strip() or "us-east-1"
        self.access_key = access_key.strip()
        self.secret_key = secret_key.strip()
        self.prefix = prefix.strip().strip("/")
        self.public_base_url = public_base_url.strip().rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._client = client if client is not None else self._create_boto3_client()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        client: Any | None = None,
    ) -> "S3AssetStorageAdapter":
        """基于全局 ``Settings`` 配置构建 ``S3AssetStorageAdapter`` 实例。"""
        return cls(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint,
            region_name=settings.s3_region,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            prefix=settings.s3_prefix,
            public_base_url=settings.s3_public_base_url,
            timeout_seconds=settings.s3_timeout_seconds,
            client=client,
        )

    def _create_boto3_client(self) -> Any:
        """延迟装配带超时策略的 boto3 S3 客户端。"""
        boto3_mod = _load_boto3()
        client_kwargs: dict[str, Any] = {}
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url
        if self.region_name:
            client_kwargs["region_name"] = self.region_name
        if self.access_key:
            client_kwargs["aws_access_key_id"] = self.access_key
        if self.secret_key:
            client_kwargs["aws_secret_access_key"] = self.secret_key

        try:
            botocore_config = importlib.import_module("botocore.config")
            client_kwargs["config"] = botocore_config.Config(
                connect_timeout=self.timeout_seconds,
                read_timeout=self.timeout_seconds,
                retries={"max_attempts": 2},
            )
        except ImportError:
            # 允许测试替身仅注入 Fake boto3 模块而不携带 botocore
            pass

        return boto3_mod.client("s3", **client_kwargs)

    def build_object_key(self, kb_name: str, doc_key: str, filename: str) -> str:
        """构造跨平台一致的 S3 对象键（``{prefix}/{kb_name}/{doc_key}/{filename}``）。"""
        safe_kb = validate_asset_segment(kb_name, "kb_name")
        safe_doc_key = validate_asset_segment(doc_key, "doc_key")
        safe_filename = validate_asset_segment(filename, "filename")
        parts = [p for p in (self.prefix, safe_kb, safe_doc_key, safe_filename) if p]
        return "/".join(parts)

    def build_storage_uri(self, object_key: str) -> str:
        """构造无损标准内部对象定位 URI（``s3://{bucket}/{object_key}``）。"""
        return f"s3://{self.bucket}/{object_key.lstrip('/')}"

    def build_public_url(self, object_key: str) -> str:
        """根据配置生成可供团队客户端与多端代理访问的公网/内网 HTTPS 直链。"""
        clean_key = object_key.lstrip("/")
        if self.public_base_url:
            return f"{self.public_base_url}/{clean_key}"
        if self.endpoint_url:
            return f"{self.endpoint_url}/{self.bucket}/{clean_key}"
        return f"https://{self.bucket}.s3.{self.region_name}.amazonaws.com/{clean_key}"

    @staticmethod
    def parse_storage_uri(storage_uri: str) -> tuple[str, str] | None:
        """解析 ``s3://{bucket}/{key}`` 格式的内部定位 URI；非 S3 URI 返回 ``None``。"""
        clean = storage_uri.strip()
        if not clean.startswith("s3://"):
            return None
        rest = clean.removeprefix("s3://")
        if "/" not in rest:
            return None
        bucket_name, object_key = rest.split("/", 1)
        if not bucket_name or not object_key:
            return None
        return bucket_name, object_key

    def upload_bytes(
        self,
        object_key: str,
        data: bytes,
        content_type: str,
    ) -> tuple[str, str]:
        """上传单个二进制图片对象至 S3 并返回 ``(storage_uri, public_url)``。"""
        clean_key = object_key.lstrip("/")
        self._client.put_object(
            Bucket=self.bucket,
            Key=clean_key,
            Body=data,
            ContentType=content_type,
        )
        return self.build_storage_uri(clean_key), self.build_public_url(clean_key)

    def read_bytes(
        self,
        storage_uri: str = "",
        *,
        kb_name: str = "",
        doc_id: str = "",
        filename: str = "",
    ) -> bytes:
        """按 ``storage_uri``（或 ``kb_name/doc_id/filename`` 回退键）从 S3 读取二进制图片字节。"""
        parsed = self.parse_storage_uri(storage_uri)
        if parsed is not None:
            target_bucket, target_key = parsed
        else:
            if not (kb_name and doc_id and filename):
                raise FileNotFoundError(f"无法定位 S3 对象资产路径: {storage_uri!r}")
            doc_key = make_asset_doc_key(doc_id)
            target_bucket = self.bucket
            target_key = self.build_object_key(kb_name, doc_key, filename)

        try:
            response = self._client.get_object(Bucket=target_bucket, Key=target_key)
        except Exception as exc:
            error_code = ""
            if hasattr(exc, "response") and isinstance(getattr(exc, "response"), dict):
                error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code in {"NoSuchKey", "404", "NotFound"}:
                raise FileNotFoundError(
                    f"S3 图片资产对象不存在 (s3://{target_bucket}/{target_key})"
                ) from exc
            raise

        body = response["Body"]
        try:
            payload = body if isinstance(body, bytes) else bytes(body.read())
        finally:
            close_fn = getattr(body, "close", None)
            if callable(close_fn):
                close_fn()
        return payload

    def delete_object(self, object_key: str, *, bucket: str | None = None) -> None:
        """删除单个 S3 对象，异常时记录告警不阻断调用方级联清理。"""
        target_bucket = bucket or self.bucket
        clean_key = object_key.lstrip("/")
        try:
            self._client.delete_object(Bucket=target_bucket, Key=clean_key)
        except Exception as exc:
            logger.warning(
                "删除 S3 图片资产对象失败 (s3://%s/%s): %s",
                target_bucket,
                clean_key,
                exc,
            )

    def delete_assets(self, assets: list[dict[str, Any]] | None) -> None:
        """批量按资产登记记录清理 S3 对象存储中的数据。"""
        for asset in assets or []:
            storage_uri = str(asset.get("storage_uri") or "")
            parsed = self.parse_storage_uri(storage_uri)
            if parsed is None:
                continue
            target_bucket, object_key = parsed
            self.delete_object(object_key, bucket=target_bucket)

    def delete_stale_assets(
        self,
        old_assets: list[dict[str, Any]],
        new_assets: list[dict[str, Any]],
    ) -> None:
        """更新文档后清理新版本不再保留的陈旧 S3 图片对象。"""
        kept_uris = {str(asset.get("storage_uri") or "") for asset in new_assets}
        stale = [
            asset for asset in old_assets if str(asset.get("storage_uri") or "") not in kept_uris
        ]
        if stale:
            self.delete_assets(stale)

    def upload_image_assets(
        self,
        kb_name: str,
        doc_id: str,
        images: list[ExtractedImage],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """批量上传文档有效图片资产至 S3（失败时自动回滚本批已上传对象）。

        Args:
            kb_name: 所属知识库名称。
            doc_id: 文档唯一标识符。
            images: 待上传的有效图片列表。

        Returns:
            ``(asset_records, filename_to_public_url)`` 元组：
            - ``asset_records`` 可直接传入 ``MetadataCatalogAdapter.register_image_assets``；
            - ``filename_to_public_url`` 提供 ``filename -> https://...`` 映射以供正文转写。
        """
        if not images:
            return [], {}

        safe_kb = validate_asset_segment(kb_name, "kb_name")
        doc_key = make_asset_doc_key(doc_id)
        asset_records: list[dict[str, Any]] = []
        public_urls: dict[str, str] = {}
        uploaded_keys: list[str] = []
        seen_lower_names: set[str] = set()

        try:
            for image in images:
                safe_filename = validate_asset_segment(image.filename, "filename")
                lower_name = safe_filename.lower()
                if lower_name in seen_lower_names:
                    raise ValueError(f"S3 图片资产批次存在大小写重名冲突: {safe_filename}")
                seen_lower_names.add(lower_name)

                object_key = self.build_object_key(safe_kb, doc_key, safe_filename)
                storage_uri, public_url = self.upload_bytes(object_key, image.data, image.mime_type)
                uploaded_keys.append(object_key)
                public_urls[safe_filename] = public_url

                asset_records.append(
                    {
                        "asset_id": make_asset_id(safe_kb, doc_key, safe_filename),
                        "filename": safe_filename,
                        "storage_uri": storage_uri,
                        "mime_type": image.mime_type,
                        "byte_size": image.byte_size,
                    }
                )
        except Exception:
            for partial_key in uploaded_keys:
                self.delete_object(partial_key)
            raise

        return asset_records, public_urls


def get_s3_storage_adapter(
    settings: Settings,
    *,
    client: Any | None = None,
) -> S3AssetStorageAdapter:
    """按配置构建 S3 对象存储网关适配器。"""
    return S3AssetStorageAdapter.from_settings(settings, client=client)
