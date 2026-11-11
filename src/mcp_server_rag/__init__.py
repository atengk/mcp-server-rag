"""
MCP Server for RAG (Retrieval-Augmented Generation).

基于 Python 与 FastMCP 构建的高性能检索增强生成 (RAG) 模型上下文协议 (MCP) 服务。

@author Ateng
@since 2026-11-11
"""

from .server import main, mcp

__version__ = "0.1.0"
__all__ = ["__version__", "main", "mcp"]
