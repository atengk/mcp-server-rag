# 0002: 基于 oss-template 工业级规范落地二元工作流与多通道发版体系

为了保障开源项目交付质量并与同级 MCP 兄弟项目（mcp-server-redis、mcp-server-rdbms）规范对齐，从开源模版 `oss-template` 提取 Python 专属二元流水线（门禁测试 ci.yml 与自动化发版 release.yml），配置原生 .githooks 提交守门，并支持 GitHub Release、Docker (GHCR) 与 PyPI 三轨综合分发。
