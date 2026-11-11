# 0005: 统一 MCP_RAG_* 环境变量契约与目录持久化规范

为确保工程配置在各种宿主（Claude Desktop、Antigravity、Docker 等）中保持绝对命名空间隔离与可预测性，决定所有系统环境变量统一遵循 `MCP_RAG_` 前缀规范（如 `MCP_RAG_DATA_DIR`、`MCP_RAG_EMBEDDING_PROVIDER`、`MCP_RAG_OPENAI_API_KEY` 等）。数据存储优先读取 `MCP_RAG_DATA_DIR`，缺省时自动落盘至当前工作区 `.rag_data/` 目录，回退至用户家目录，保障多工程数据天然物理隔离。
