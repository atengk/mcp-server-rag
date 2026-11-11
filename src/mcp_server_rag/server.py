"""
FastMCP 服务主实现模块。

提供检索增强生成 (RAG) 相关的基础健康检查与核心工具接口。

@author Ateng
@since 2026-11-11
"""

from mcp.server import MCPServer

# 兼容 FastMCP 类型别名
FastMCP = MCPServer

# 初始化 MCP 服务实例
mcp = FastMCP(
    name="mcp-server-rag",
    instructions="Model Context Protocol Server for Retrieval-Augmented Generation (RAG).",
)


@mcp.tool()
def health_check() -> dict[str, str]:
    """
    检查 RAG MCP 服务运行状态与健康度。

    Returns:
        dict[str, str]: 包含服务状态、名称与版本的键值对字典。
    """
    return {
        "status": "healthy",
        "service": "mcp-server-rag",
        "version": "0.1.0",
    }


def main() -> None:
    """运行 MCP 服务主入口。"""
    mcp.run()


if __name__ == "__main__":
    main()
