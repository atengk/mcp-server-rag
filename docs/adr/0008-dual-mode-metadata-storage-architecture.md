# 0008: 个人与团队双模元数据存储架构（SQLite + 可插拔 PostgreSQL/MySQL）

## 背景与问题
原本设计以单机嵌入式 SQLite 作为元数据目录存储，能提供极简的开箱即用体验。但在企业团队协作与云原生容器化部署场景下，多个 MCP 实例或团队成员需要并发读写同一套知识库元数据，单文件 SQLite 面临并发写锁瓶颈与无法跨节点共享的限制。

## 架构决策
决定将元数据目录层（Metadata Catalog）重构为可插拔抽象架构，支持“个人/团队双模平滑切换”：
1. **个人模式（默认）**：保留 Python 内置的轻量嵌入式 SQLite（`metadata.db`），零外部中间件，秒级启动；
2. **团队模式**：支持通过 `MCP_RAG_METADATA_STORE=postgresql|mysql` 切换至企业级关系型数据库；
3. **细粒度与全连接串双轨配置**：全面支持拆分式环境变量（`MCP_RAG_DB_HOST`、`MCP_RAG_DB_PORT`、`MCP_RAG_DB_USER`、`MCP_RAG_DB_PASSWORD`、`MCP_RAG_DB_NAME`）与整串 `MCP_RAG_DATABASE_URL`，兼顾云原生分发与单行配置；
4. **统一表结构契约**：知识库表扩展 `owner`（所有者）与 `visibility`（可见性，默认 `public`），为团队共享与后续权限过滤打下坚实基础。
