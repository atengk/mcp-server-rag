# MCP Server RAG (Retrieval-Augmented Generation)

<p align="center">
  <strong>基于 Python 与 FastMCP 构建的高性能检索增强生成 (RAG) 模型上下文协议 (MCP) 服务</strong>
</p>

<p align="center">
  <a href="https://github.com/atengk/mcp-server-rag/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/atengk/mcp-server-rag/ci.yml?branch=main&label=CI&style=flat-square" alt="CI Status" />
  </a>
  <a href="https://github.com/atengk/mcp-server-rag/releases">
    <img src="https://img.shields.io/github/v/release/atengk/mcp-server-rag?style=flat-square" alt="Release" />
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square" alt="License" />
  </a>
  <a href="./CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome" />
  </a>
</p>

---

## 📖 项目简介

`mcp-server-rag` 是一个专为大语言模型与 AI 智能体（Claude Desktop、Antigravity、Cursor、Cline 等）设计的检索增强生成（RAG）MCP 服务端。通过标准化的 Model Context Protocol (MCP) 接口，为 Agent 提供知识库索引、文档切分、向量嵌入、混合检索与上下文注入能力。

---

## ✨ 核心特性

- 🚀 **极速轻量**：基于 FastMCP 与现代 Python 3.10+，启动秒级响应，资源占用低；
- 🛡️ **工程规范标准**：完全继承 [oss-template](https://github.com/atengk/oss-template) 工业级开源底座，集成 GitHub Actions CI/CD、代码质量门禁 (Ruff) 与 Git 原生提交守护钩子；
- 📦 **全场景分发**：支持 PyPI 官方包发布、多架构 Docker 镜像 (GHCR) 以及一键 `uvx` 免安装即用；
- 🧩 **领域模型对齐**：遵循严谨的统一领域语言规范（详见 [CONTEXT.md](./CONTEXT.md)）。

---

## 🛠️ 快速开始

### 方式一：使用 `uvx` 免安装直接运行 (推荐首选)

```bash
uvx atengk-mcp-server-rag
```

### 方式二：使用 Docker 容器运行

```bash
docker run -i --rm ghcr.io/atengk/mcp-server-rag:latest
```

或使用内置 `docker-compose.yml`：
```bash
docker compose up -d
```

### 方式三：本地开发与源码安装

```bash
# 1. 克隆仓库
git clone https://github.com/atengk/mcp-server-rag.git
cd mcp-server-rag

# 2. 激活本地 Git 规范提交守护钩子
git config core.hooksPath .githooks

# 3. 使用 uv 安装依赖并启动环境
uv sync

# 4. 运行服务
uv run mcp-server-rag
```

---

## 💻 MCP 客户端接入配置

在你的 MCP 客户端配置文件（如 Claude Desktop 的 `claude_desktop_config.json` 或 Antigravity MCP 配置文件）中追加如下配置：

```json
{
  "mcpServers": {
    "rag": {
      "command": "uvx",
      "args": ["atengk-mcp-server-rag"]
    }
  }
}
```

如需使用源码或本地虚拟环境启动：

```json
{
  "mcpServers": {
    "rag": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-server-rag", "mcp-server-rag"]
    }
  }
}
```

---

## 🧪 测试与质量验证

```bash
# 代码风格与静态分析检查
uv run ruff check .
uv run ruff format --check .

# 单元测试执行
uv run pytest
```

---

## 📄 开源许可

本项目遵循 [Apache-2.0 许可证](./LICENSE)。欢迎提交 Issue 与 Pull Request！
