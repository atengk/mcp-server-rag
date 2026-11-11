"""
单元测试：验证 MCP Server RAG 基础服务健康检查与注册。

@author Ateng
@since 2026-11-11
"""

from mcp_server_rag import __version__
from mcp_server_rag.server import health_check, mcp


def test_version() -> None:
    """验证版本号导出正常。"""
    assert __version__ == "0.1.0"


def test_mcp_instance_initialized() -> None:
    """验证 FastMCP 实例正确初始化。"""
    assert mcp.name == "mcp-server-rag"


def test_health_check() -> None:
    """验证健康检查工具调用返回预期结构。"""
    result = health_check()
    assert result["status"] == "healthy"
    assert result["service"] == "mcp-server-rag"
    assert result["version"] == "0.1.0"
