# 0003: 采用分层可插拔双模向量存储架构（ChromaDB + Milvus）

为了同时满足个人单机轻量开箱即用与企业级大规模分布式向量检索的双重诉求，决定采用分层可插拔向量存储适配器架构。默认使用纯 Python 内嵌式 ChromaDB 零外部依赖运行，所有数据落盘于本地目录；同时提供 Milvus 适配器，通过配置 `MCP_RAG_VECTOR_STORE=milvus` 与 `MCP_RAG_MILVUS_URI` 可无缝接入分布式 Milvus 集群，与团队已有的向量基础设施生态深度打通。
