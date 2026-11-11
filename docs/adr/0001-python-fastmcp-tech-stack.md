# 0001: 采用 Python + FastMCP + uv 作为核心工程技术栈与包管理基座

在实现检索增强生成（RAG）领域模型上下文协议（MCP）服务时，决定采用 Python 3.10+ 作为主力语言，基于 FastMCP 构建 MCP 服务，并使用 uv 作为极速现代依赖管理工具。Python 在 AI、嵌入向量模型与向量检索生态中拥有最丰富的原生支持，FastMCP 能极大简化 Tool/Resource 的声明与类型校验，uv 提供了秒级依赖解析与确定性环境构建。
