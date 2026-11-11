"""
远程图片安全外链归档与防 SSRF (Server-Side Request Forgery) 防护模块。

在显式开启 ``MCP_RAG_ARCHIVE_REMOTE_IMAGES=true`` 时执行严格的多重安全检定：
1. 协议白名单限定 ``http://`` 与 ``https://``（严禁 ``file://``、``ftp://``、``gopher://`` 等）；
2. URL 字面量 IP 与 DNS 双重校验，拦截环回、私有网段、链路本地、CG-NAT 与组播/保留地址；
3. 手工受控 30x 重定向跳数遍历，确保每一跳重定向目标同样经过完整 SSRF 检查；
4. 显式网络超时控制与流式响应字节上限截断，并经 Pillow 解码校验后返回标准化图片资产。

@author Ateng
@since 2026-11-11
"""

import ipaddress
import logging
import socket
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx

from mcp_server_rag.assets import ExtractedImage, inspect_explicit_image_bytes

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})
_REDIRECT_STATUS_CODES: frozenset[int] = frozenset({301, 302, 303, 307, 308})
_FORBIDDEN_HOSTNAMES: frozenset[str] = frozenset({"localhost", "localhost.localdomain"})
_ALLOWED_CONTENT_TYPE_PREFIXES: tuple[str, ...] = (
    "image/",
    "application/octet-stream",
    "binary/octet-stream",
)

DnsResolver = Callable[[str, int], list[str]]


class SSRFBlockedError(ValueError):
    """当远程 URL 违反安全策略或指向非公网内部地址时抛出的明确拦截异常。"""


def _is_forbidden_ip(ip_str: str) -> bool:
    """判断给定 IPv4/IPv6 地址是否属于非全局可路由或潜在内网探测范围。"""
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return True

    # 解包 IPv6 映射的 IPv4 地址（如 ::ffff:127.0.0.1 / ::ffff:169.254.169.254）
    mapped_v4 = getattr(ip_obj, "ipv4_mapped", None)
    candidates = (ip_obj, mapped_v4) if mapped_v4 is not None else (ip_obj,)

    for addr in candidates:
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
            or getattr(addr, "is_site_local", False)
            or not addr.is_global
        ):
            return True
    return False


def _resolve_host_ips(hostname: str, port: int) -> list[str]:
    """解析目标主机名的全部 TCP IP 地址，失败或空结果时抛错阻断。"""
    try:
        addr_infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise SSRFBlockedError(f"无法解析目标主机 DNS ({hostname}): {exc}") from exc

    ips: list[str] = []
    for info in addr_infos:
        sockaddr = info[4]
        if sockaddr:
            ips.append(str(sockaddr[0]))

    if not ips:
        raise SSRFBlockedError(f"目标主机 DNS 未返回任何有效 IP 地址: {hostname}")
    return ips


def validate_remote_image_url(
    url: str,
    *,
    dns_resolver: DnsResolver | None = None,
) -> str:
    """验证远程图片 URL 是否符合安全协议与公网地址要求。

    Args:
        url: 待校验的原始 URL 字符串。
        dns_resolver: 可选 DNS 解析注入函数 ``(hostname, port) -> list[ip_str]``。

    Returns:
        校验通过的规范化 HTTP(S) URL。

    Raises:
        SSRFBlockedError: URL 格式非法、协议不在白名单或解析目标属于非公网 IP 时抛出。
    """
    clean_url = url.strip()
    if not clean_url or any(ch in clean_url for ch in ("\r", "\n", "\x00")):
        raise SSRFBlockedError("远程图片 URL 包含非法空白或控制字符")

    parsed = urlparse(clean_url)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise SSRFBlockedError(f"禁止抓取非 HTTP/HTTPS 协议的图片链接: {url}")

    if parsed.username is not None or parsed.password is not None:
        raise SSRFBlockedError(f"出于解析器混淆防护考虑，禁止携带认证信息的外链 URL: {url}")

    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise SSRFBlockedError(f"远程图片 URL 缺少合法主机名: {url}")

    if hostname in _FORBIDDEN_HOSTNAMES or hostname.endswith((".localhost", ".local", ".internal")):
        raise SSRFBlockedError(f"禁止抓取本机或内网域名图片: {hostname}")

    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise SSRFBlockedError(f"远程图片 URL 端口非法: {url}") from exc

    # 1. 若 host 为字面量 IP，直接拦截非公网网段而不触发 DNS
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if _is_forbidden_ip(hostname):
            raise SSRFBlockedError(f"禁止抓取非公网 IP 地址的图片资源: {hostname}")
        return clean_url

    # 2. 解析域名对应全部 IP 并逐一检验，防止 DNS 重绑定指向内网
    resolver = dns_resolver or _resolve_host_ips
    resolved_ips = resolver(hostname, port)
    if not resolved_ips:
        raise SSRFBlockedError(f"目标主机未能解析出可用公网 IP: {hostname}")
    for ip_str in resolved_ips:
        if _is_forbidden_ip(ip_str):
            raise SSRFBlockedError(f"目标域名 {hostname} 解析到受限非公网 IP ({ip_str})，已拦截")

    return clean_url


def _derive_remote_filename_hint(url: str, fallback: str = "remote_image") -> str:
    """从远程图片 URL 路径提取安全文件名提示。"""
    path = unquote(urlparse(url).path)
    basename = Path(path).name.strip()
    return basename or fallback


def _is_acceptable_image_content_type(content_type: str) -> bool:
    """校验 HTTP Content-Type 响应头是否可能为有效二进制图片内容。"""
    clean_ct = content_type.split(";", 1)[0].strip().lower()
    if not clean_ct:
        return True
    return clean_ct.startswith(_ALLOWED_CONTENT_TYPE_PREFIXES)


def download_safe_remote_image(
    url: str,
    *,
    filename_hint: str = "",
    alt: str = "",
    timeout_seconds: int = 15,
    max_bytes: int = 10 * 1024 * 1024,
    max_redirects: int = 5,
    transport: httpx.BaseTransport | None = None,
    dns_resolver: DnsResolver | None = None,
) -> ExtractedImage | None:
    """带全链路 SSRF 防护、超时与体积截断限制地安全下载并校验远程图片。

    Args:
        url: 远程图片 URL。
        filename_hint: 可选图片文件名提示，省略时从最终 URL 路径提取。
        alt: 图片替代描述文本。
        timeout_seconds: 单跳 HTTP 连接/读取超时秒数，默认 ``15``。
        max_bytes: 允许下载的最大响应体字节数，默认 ``10MB``。
        max_redirects: 最大允许 HTTP 重定向跳转次数，默认 ``5``。
        transport: 可选注入的 ``httpx.BaseTransport``（如测试用 ``MockTransport``）。
        dns_resolver: 可选 DNS 解析注入函数。

    Returns:
        下载并经 Pillow 解码校验通过时返回 ``ExtractedImage``；
        若因 HTTP 状态错误、非图片响应或体积超限失败则返回 ``None``。

    Raises:
        SSRFBlockedError: 当初始 URL 或任意重定向跳转目标违反 SSRF 安全策略时直接抛出。
    """
    current_url = validate_remote_image_url(url, dns_resolver=dns_resolver)
    timeout = httpx.Timeout(
        connect=float(timeout_seconds),
        read=float(timeout_seconds),
        write=float(timeout_seconds),
        pool=float(timeout_seconds),
    )
    client_kwargs: dict[str, object] = {
        "timeout": timeout,
        "follow_redirects": False,
        "trust_env": False,
    }
    if transport is not None:
        client_kwargs["transport"] = transport

    headers = {
        "User-Agent": "Antigravity-RAG/0.1.0 (+image-archive)",
        "Accept": "image/*,*/*;q=0.8",
    }

    with httpx.Client(**client_kwargs) as client:  # type: ignore[arg-type]
        for _hop in range(max_redirects + 1):
            validated_url = validate_remote_image_url(current_url, dns_resolver=dns_resolver)
            try:
                with client.stream("GET", validated_url, headers=headers) as resp:
                    if resp.status_code in _REDIRECT_STATUS_CODES:
                        location = resp.headers.get("location")
                        if not location:
                            return None
                        current_url = validate_remote_image_url(
                            urljoin(validated_url, location),
                            dns_resolver=dns_resolver,
                        )
                        continue

                    resp.raise_for_status()

                    content_type = resp.headers.get("content-type", "")
                    if not _is_acceptable_image_content_type(content_type):
                        logger.warning(
                            "跳过非图片 Content-Type (%s) 的远程资源: %s",
                            content_type,
                            validated_url,
                        )
                        return None

                    raw_len = resp.headers.get("content-length")
                    if raw_len:
                        try:
                            if int(raw_len) > max_bytes:
                                logger.warning(
                                    "远程图片声明大小超过上限 (%s > %d): %s",
                                    raw_len,
                                    max_bytes,
                                    validated_url,
                                )
                                return None
                        except ValueError:
                            logger.debug(
                                "远程图片 Content-Length 头非整数 (%s): %s",
                                raw_len,
                                validated_url,
                            )

                    buf = bytearray()
                    for chunk in resp.iter_bytes():
                        if not chunk:
                            continue
                        if len(buf) + len(chunk) > max_bytes:
                            logger.warning(
                                "远程图片流式读取超过大小上限 (%d bytes): %s",
                                max_bytes,
                                validated_url,
                            )
                            return None
                        buf.extend(chunk)

                    resolved_hint = filename_hint or _derive_remote_filename_hint(validated_url)
                    return inspect_explicit_image_bytes(
                        bytes(buf),
                        filename_hint=resolved_hint,
                        alt=alt,
                    )
            except SSRFBlockedError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("抓取远程图片失败 (%s): %s", validated_url, exc)
                return None

    logger.warning("远程图片重定向次数超过上限 (%d): %s", max_redirects, url)
    return None
