# MCP Server RAG (Retrieval-Augmented Generation)

<p align="center">
  <strong>基于 Python 与 FastMCP 构建的工业级检索增强生成 (RAG) 模型上下文协议 (MCP) 服务端</strong>
</p>

<p align="center">
  <a href="https://github.com/atengk/mcp-server-rag/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/atengk/mcp-server-rag/ci.yml?branch=main&label=CI&style=flat-square" alt="CI Status" />
  </a>
  <a href="https://github.com/atengk/mcp-server-rag/releases">
    <img src="https://img.shields.io/github/v/release/atengk/mcp-server-rag?style=flat-square" alt="Release" />
  </a>
  <a href="https://pypi.org/project/atengk-mcp-server-rag/">
    <img src="https://img.shields.io/pypi/v/atengk-mcp-server-rag?style=flat-square&color=blue" alt="PyPI" />
  </a>
  <a href="https://pypi.org/project/atengk-mcp-server-rag/">
    <img src="https://img.shields.io/pypi/pyversions/atengk-mcp-server-rag?style=flat-square" alt="Python Versions" />
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

`mcp-server-rag` 是一个专为大语言模型与 AI 智能体（Claude Desktop、Google Antigravity、Cursor、Windsurf、Cline 等）量身打造的开箱即用检索增强生成（RAG）服务端。

项目基于现代 Python 3.10+ 与 FastMCP 框架构建，严格遵循标准化的 **Model Context Protocol (MCP)** 规范。它彻底告别了传统 RAG 系统的重型捆绑负担，具备**单机零中间件开箱即用**与**企业级分布式水平扩展**双模能力：
- **灵活的存储底座**：支持本地内嵌 SQLite + ChromaDB，亦可平滑无缝切换至分布式 Milvus 与企业级关系型数据库（PostgreSQL / MySQL）；
- **全方位检索管线**：实现高维向量语义搜索与数据库原生全文检索（FTS5 / GIN tsvector / FULLTEXT）的 **RRF 倒数排名融合**，并按需接入 FastEmbed ONNX 与云端 HTTP API **双模交叉重排器（Reranker）**；
- **多模态与富文档**：覆盖 Markdown、代码、结构化数据、在线网页以及 Office/PDF（`.docx`, `.pdf`），支持图片原子保护、图像降噪过滤、FastMCP Resource 图片分发与 S3/MinIO 对象存储外链安全归档；
- **滑动上下文增强**：基于双向链表拓扑支持邻近切片窗口扩展（`expand_context`），精准还原长篇上下文连贯性。

---

## 🏛️ 系统架构 (Architecture)

本项目采用**分层解耦、双模切换与多源混合检索**设计，系统整体运行拓扑如下：

```mermaid
flowchart TD
    subgraph ClientLayer["AI 智能体客户端 (MCP Client)"]
        Client["Claude Desktop / Antigravity / Cursor / Windsurf / CLI"]
    end

    subgraph ProtocolLayer["MCP 通信、安全与资源网关层"]
        Protocol["FastMCP 协议接口 (stdio / SSE)"]
        Guard{"只读守卫门禁\n(MCP_RAG_READ_ONLY)"}
        ResourceGW["FastMCP Resource 图片网关\n(rag://assets/{kb}/{doc}/{filename})"]
    end

    subgraph PipelineLayer["结构感知摄取管线 (Ingestion Pipeline)"]
        Deduper["SHA-256 内容指纹比对\n(幂等去重 / 原子替换)"]
        MDParser["Markdown 标题树保护切片器\n(图片原子块保护)"]
        OfficeParser["Office/PDF 解析器 (.docx/.pdf)\n(图像降噪过滤 / 跨页锚定)"]
        CodeParser["源码语法边界切片器 (.py/.ts/.java/.go)"]
        DataParser["结构化数据切分器 (JSON/CSV/YAML)"]
        URLParser["原生 HTML 视觉降噪与正文清洗"]
    end

    subgraph RetrievalLayer["双路召回与二次重排流水线 (Hybrid Retrieval & Rerank)"]
        DenseSearch["稠密向量语义检索\n(ChromaDB / Milvus)"]
        NativeFTS["数据库原生 FTS 倒排检索\n(SQLite FTS5 / PG tsvector / MySQL FULLTEXT)"]
        RRF["Reciprocal Rank Fusion (RRF k=60)\n排名融合与单调校准"]
        Reranker{"可插拔双模 Reranker\n(MCP_RAG_RERANK_ENABLED)"}
        FastEmbedRerank["FastEmbed ONNX 交叉重排\n(BAAI/bge-reranker-base / CPU)"]
        ApiRerank["云端 HTTP API 重排\n(/v1/rerank: SiliconFlow / Cohere / 百炼)"]
    end

    subgraph StorageLayer["双模存储与资产底座 (Storage Adapters)"]
        MetaStore[("元数据与 FTS 存储\n(SQLite / PostgreSQL / MySQL)")]
        VecStore[("向量存储引擎\n(ChromaDB / Milvus)")]
        AssetStore[("图片资产存储\n(本地磁盘 / S3 / MinIO)")]
    end

    Client <--> Protocol
    Protocol --> Guard
    Guard -->|"写操作/索引工具 (只读时物理隐藏)"| Deduper
    Guard -->|"混合检索/诊断工具"| RetrievalLayer
    Protocol <--> ResourceGW
    ResourceGW <--> AssetStore

    Deduper --> MDParser & OfficeParser & CodeParser & DataParser & URLParser
    MDParser & OfficeParser & CodeParser & DataParser & URLParser --> StorageLayer

    DenseSearch & NativeFTS --> RRF
    RRF --> Reranker
    Reranker -->|"fastembed"| FastEmbedRerank
    Reranker -->|"api"| ApiRerank
    Reranker -->|"disabled"| Client
    FastEmbedRerank & ApiRerank --> Client
```

---

## ✨ 核心特性

- 🪶 **单机零中间件自足 (Zero-Middleware Invariant)**：
  坚决不强制引入外部中间件；内建 SQLite 管理元数据与 FTS5 倒排索引，默认内嵌 ChromaDB 持久化向量索引，单机冷启动秒级就绪。
- ⚡ **本地轻量级 CPU 推理 (FastEmbed ONNX)**：
  默认集成 FastEmbed 运行时与 `BAAI/bge-small-zh-v1.5`（512 维，~90MB 模型），纯 CPU 毫秒级生成稠密向量，内置镜像加速，无 GPU 依赖。
- 🌐 **多模型向量嵌入生态 (Embedding Providers)**：
  支持通过环境变量无缝切换至 OpenAI、通义千问 (Qwen)、Ollama、DeepSeek 等 OpenAI 兼容嵌入接口；支持 MRL 向量维度动态裁剪、自动网络维度探测与跨模型写入防错熔断。
- 🔍 **原生 FTS + 向量混合检索与 RRF 融合 (Hybrid Search)**：
  拒绝引入庞大的外部 Elasticsearch；利用 SQLite FTS5、PostgreSQL GIN tsvector 或 MySQL FULLTEXT 原生倒排检索，结合标准 RRF（Reciprocal Rank Fusion, $k=60$）算法与单调置信度校准，彻底解决特定代码、专有名词与精确术语漏召回痛点。
- 🎯 **双模可插拔二次重排 (Reranker)**：
  支持通过 `MCP_RAG_RERANK_ENABLED=true` 按需开启精排；提供基于 FastEmbed ONNX 的纯 CPU 本地交叉编码（默认 `BAAI/bge-reranker-base`）与兼容云端 `/v1/rerank`（SiliconFlow、Cohere、百炼）的 HTTP 驱动，对扩大候选初筛池进行交叉注意力重排并按阈值截断。
- 🖼️ **多模态图片资产网关与 S3/MinIO 共享 (Multimodal & MCP Resources)**：
  - **原子切片保护**：确保 Markdown 与 HTML 图片标签不被切片切断；
  - **FastMCP Resource**：提供标准化 `rag://assets/{kb_name}/{doc_id}/{filename}` 协议，供多模态模型按需拉取图像二进制流；
  - **多端共享与安全归档**：支持将图片上传至 S3/MinIO 对象存储并自动转写为公网直链；内置严谨的 SSRF 防御沙箱（拦截环回、私有 IP、链路本地、DNS 重绑定与危险重定向），安全归档远程外链。
- 📑 **Office / PDF 富文档智能摄取 (Zero-LangChain)**：
  原生集成 `.docx` 与 `.pdf` 解析引擎，支持段落与表格（SDT / AlternateContent）语法边界提取、智能图像降噪过滤（过滤宽高小于 64px 或体积小于 5KB 的装饰噪点图）与跨页图片语义锚定。
- 🔗 **滑动上下文窗口与邻近切片拼接 (Context Window Stitching)**：
  基于双向链表存储拓扑，在检索时支持通过 `expand_context`（1 ~ 5 步）自动向前、向后扩展并去重缝合邻近切片，有效弥补固定切片截断的上下文裂痕。
- 🏛️ **企业级双模存储平滑升级**：
  - **向量底座**：支持从本地内嵌 ChromaDB 无感切换至企业级分布式 Milvus 集群；
  - **元数据库**：支持从 SQLite 升级为高并发 PostgreSQL 或 MySQL。
- 🛡️ **工业级只读安全防灾门禁 (Read-Only Guard)**：
  通过 `MCP_RAG_READ_ONLY=true` 在 FastMCP 握手层物理隐藏全部写操作工具，提供双层纵深防御，杜绝智能体误操作污染生产数据。

---

## 🛠️ 快速开始

### 方式一：使用 `uvx` 免安装一键运行 (推荐首选)

无需预先安装 Python 依赖，直接通过 `uvx` 即开即用：

```bash
uvx atengk-mcp-server-rag
```

若需包含可选企业级扩展（如 Milvus、Office/PDF、Postgres、MySQL、S3 支持）：

```bash
# 包含所有可选驱动扩展
uvx "atengk-mcp-server-rag[all]"

# 或仅引入指定驱动扩展（如 office 与 s3）
uvx "atengk-mcp-server-rag[office,s3]"
```

### 方式二：使用 Docker 容器运行

```bash
# 拉取并运行预构建多架构容器镜像
docker run -i --rm -v $(pwd)/.rag_data:/app/.rag_data ghcr.io/atengk/mcp-server-rag:latest
```

或使用仓库自带的 `docker-compose.yml` 后台守护运行：

```bash
docker compose up -d
```

### 方式三：源码本地开发安装与运行

```bash
# 1. 克隆代码仓库
git clone https://github.com/atengk/mcp-server-rag.git
cd mcp-server-rag

# 2. 激活本地 Git 规范提交守护钩子
git config core.hooksPath .githooks

# 3. 基于 uv 快速同步项目全量依赖与开发工具
uv sync --all-extras --dev

# 4. 启动 MCP 服务端
uv run mcp-server-rag
```

---

## 💻 通用 MCP 客户端接入配置 (Client Configuration)

将以下配置片段追加至你的 MCP 宿主配置文件中（如 Claude Desktop 的 `claude_desktop_config.json`、Antigravity 的 `mcp_settings.json` 或 Cursor / Windsurf 的 MCP 扩展配置）：

### 1. 标准默认接入配置 (`uvx` 免安装，纯本地零配置)

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

### 2. 开启云端 Embedding + 二次 Reranker 重排模式

```json
{
  "mcpServers": {
    "rag": {
      "command": "uvx",
      "args": ["atengk-mcp-server-rag"],
      "env": {
        "MCP_RAG_EMBEDDING_PROVIDER": "qwen",
        "MCP_RAG_EMBEDDING_MODEL": "text-embedding-v3",
        "MCP_RAG_EMBEDDING_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "MCP_RAG_EMBEDDING_API_KEY": "sk-your-dashscope-api-key",
        "MCP_RAG_RERANK_ENABLED": "true",
        "MCP_RAG_RERANK_PROVIDER": "api",
        "MCP_RAG_RERANK_MODEL": "BAAI/bge-reranker-v2-m3",
        "MCP_RAG_RERANK_BASE_URL": "https://api.siliconflow.cn/v1",
        "MCP_RAG_RERANK_API_KEY": "sk-your-siliconflow-api-key"
      }
    }
  }
}
```

### 3. 企业级全栈模式 (PostgreSQL + Milvus + S3 图片网关)

```json
{
  "mcpServers": {
    "rag": {
      "command": "uvx",
      "args": ["atengk-mcp-server-rag[all]"],
      "env": {
        "MCP_RAG_VECTOR_STORE": "milvus",
        "MCP_RAG_MILVUS_URI": "http://10.0.0.10:19530",
        "MCP_RAG_MILVUS_TOKEN": "root:Milvus2026",
        "MCP_RAG_METADATA_STORE": "postgresql",
        "MCP_RAG_DATABASE_URL": "postgresql://rag_user:Secret2026@10.0.0.11:5432/rag_metadata",
        "MCP_RAG_STORAGE_TYPE": "s3",
        "MCP_RAG_S3_BUCKET": "team-rag-assets",
        "MCP_RAG_S3_ENDPOINT": "https://minio.corp.internal:9000",
        "MCP_RAG_S3_ACCESS_KEY": "minioadmin",
        "MCP_RAG_S3_SECRET_KEY": "miniopassword",
        "MCP_RAG_ARCHIVE_REMOTE_IMAGES": "true"
      }
    }
  }
}
```

### 4. 只读安全模式 (生产查阅与防误删场景)

在握手层物理隐藏所有建库、索引与清理工具，仅开放检索与问答探针：

```json
{
  "mcpServers": {
    "rag": {
      "command": "uvx",
      "args": ["atengk-mcp-server-rag"],
      "env": {
        "MCP_RAG_READ_ONLY": "true"
      }
    }
  }
}
```

---

## 🧰 FastMCP 工具与资源契约规范 (Tools & Resources)

### 一、 FastMCP 核心工具矩阵 (13 Tools)

#### 1. 知识库生命周期管理 (Knowledge Base Lifecycle)

| 工具名称 | 读写属性 | 参数与类型 | 核心功能说明 |
| :--- | :--- | :--- | :--- |
| **`rag_create_kb`** | 写操作 | `name: str`<br>`description: str = ""` | 创建新的知识库逻辑命名空间并在底层向量库与元数据库初始化。 |
| **`rag_list_kbs`** | 只读 | _(无入参)_ | 枚举所有知识库，包含各自的文档总数、切片统计、创建与更新时间。 |
| **`rag_clear_kb`** | 写操作 | `name: str` | 原子清空指定知识库下的所有文档、切片与关联图片资产，保留基础配置。 |
| **`rag_delete_kb`** | 写操作 | `name: str` | 物理销毁指定知识库及其关联的全部元数据、倒排索引、向量数据与物理资产。 |

#### 2. 多源数据摄取与细粒度管理 (Ingestion & Document Management)

| 工具名称 | 读写属性 | 参数与类型 | 核心功能说明 |
| :--- | :--- | :--- | :--- |
| **`rag_index_text`** | 写操作 | `kb_name: str`<br>`text: str`<br>`title: str`<br>`metadata: dict = None` | 对纯文本内容进行语法切片、向量嵌入与入库；支持基于内容指纹幂等去重。 |
| **`rag_index_file`** | 写操作 | `kb_name: str`<br>`file_path: str`<br>`metadata: dict = None` | 自动识别格式（Markdown、源码、数据、Office、PDF），执行结构感知切片与增量索引。 |
| **`rag_index_directory`** | 写操作 | `kb_name: str`<br>`dir_path: str`<br>`glob_pattern: str = "**/*"`<br>`max_files: int = 200`<br>`recursive: bool = True` | 批量扫描并摄取本地目录文档；内置格式过滤、忽略列表与 `max_files` 防超时截断。 |
| **`rag_index_url`** | 写操作 | `kb_name: str`<br>`url: str`<br>`metadata: dict = None` | 抓取网页并清洗 HTML 视觉噪点，提取正文、处理相对外链图片并完成向量化。 |
| **`rag_delete_document`** | 写操作 | `kb_name: str`<br>`doc_id: str` | 按文档 ID 精确级联删除单个文档、所有切片以及登记的物理图片资产。 |

#### 3. 混合检索与上下文增强 (Hybrid Search & Context Augmentation)

| 工具名称 | 读写属性 | 参数与类型 | 核心功能说明 |
| :--- | :--- | :--- | :--- |
| **`rag_search`** | 只读 | `kb_name: str \| list[str]`<br>`query: str`<br>`top_k: int = 5`<br>`threshold: float = 0.0`<br>`keyword: str = None`<br>`metadata_filter: dict = None`<br>`expand_context: int = 0` | 跨库向量检索与原生 FTS 倒排 RRF 融合混合检索：<br>1. 支持单库名、知识库名称列表或 `"*"` 跨全库联合检索；<br>2. 支持相似度阈值截断（0.0 ~ 1.0）；<br>3. 精准关键词必须包含过滤与元数据标量过滤；<br>4. 支持 Reranker 二次交叉重排打分；<br>5. 支持 `expand_context` 邻近上下文滑动扩展；<br>6. 返回结果包含 `images` 结构化图片元数据。 |
| **`rag_query_with_context`** | 只读 | `kb_name: str \| list[str]`<br>`query: str`<br>`top_k: int = 5`<br>`threshold: float = 0.0`<br>`keyword: str = None`<br>`metadata_filter: dict = None`<br>`expand_context: int = 0` | 检索相关切片，并自动格式化组装为包含来源标题、知识库归属、评分打分与 Markdown 引用块的增强提示词，供大模型直接回答问题。 |

#### 4. 运维诊断与健康监测 (Operations & Diagnostics)

| 工具名称 | 读写属性 | 参数与类型 | 核心功能说明 |
| :--- | :--- | :--- | :--- |
| **`rag_doctor`** | 只读 | _(无入参)_ | 全维系统健康自检：检查 SQLite/PG/MySQL 连通性、向量库连通性、磁盘占用、模型维度、Reranker 状态、存储模式与只读状态。 |
| **`health_check`** | 只读 | _(无入参)_ | 检查 RAG MCP 服务的在线状态、服务标识与版本号。 |

---

### 二、 FastMCP 标准资源网关 (MCP Resources)

服务端动态注册并暴露符合 MCP 规范的标准图片资源模板：

```
rag://assets/{kb_name}/{doc_id}/{filename}
```

- **协议能力**：多模态大模型在检索切片中获取图片地址后，可直接通过 MCP `read_resource` 协议读取对应图片的精确二进制流（`BlobResourceContents`）与 MIME 类型（`image/png`, `image/jpeg` 等）；
- **只读兼容**：在 `MCP_RAG_READ_ONLY=true` 模式下，写工具虽被物理隐藏，但 Resource 网关保持正常开放，保障智能体多模态图文查阅畅通无阻。

---

## 💡 典型 Agent 交互工作流示范

```markdown
# 1. 摄取本地设计文档（含本地相对图片）
Agent 调用 -> rag_index_file(kb_name="project_docs", file_path="./docs/architecture.docx")
返回结果: {"success": true, "status": "indexed", "chunk_count": 12, "images_count": 2}

# 2. 混合检索（带滑动邻近切片扩展）
Agent 调用 -> rag_search(kb_name="project_docs", query="图片资产网关与高可用架构", top_k=1, expand_context=1)
返回命中切片:
{
  "chunk_id": "doc_1_5",
  "text": "...前序切片内容...\n[当前切片] 团队图片网关采用 FastMCP Resource 协议暴露资产...\n...后序切片内容...",
  "score": 0.9520,
  "metadata": {
    "retrieval_channels": ["vector", "fts"],
    "rrf_score": 0.032787,
    "rerank_score": 0.9520,
    "rerank_model": "BAAI/bge-reranker-base",
    "images": [
      {
        "filename": "arch_diagram.png",
        "resource_uri": "rag://assets/project_docs/doc_1/arch_diagram.png",
        "mime_type": "image/png",
        "alt": "高可用架构拓扑图"
      }
    ]
  }
}
```

---

## ⚙️ 环境变量配置契约 (Configuration Contract)

本项目所有配置项统一遵循 `MCP_RAG_` 命名空间前缀，具备严格的类型校验与异常非法输入安全回退：

| 环境变量名 | 数据类型 | 默认缺省值 | 功能说明与容错机制 |
| :--- | :--- | :--- | :--- |
| **`MCP_RAG_DATA_DIR`** | Path | `.rag_data` | 本地持久化数据根目录（存放 SQLite `metadata.db`、Chroma 索引与本地图片资产）。 |
| **`MCP_RAG_READ_ONLY`** | bool | `false` | 只读安全守卫：设为 `true`/`1`/`yes`/`on` 时，在 MCP 握手层物理隐藏全部写操作工具。 |
| **`MCP_RAG_CHUNK_SIZE`** | int | `600` | 默认切片长度（字符数），若传入非法非数字字符串自动安全回退为 `600`。 |
| **`MCP_RAG_CHUNK_OVERLAP`** | int | `80` | 切片重叠窗口长度（字符数），若传入负数或非法格式自动安全回退为 `80`。 |
| **`MCP_RAG_MAX_FILES_BATCH`** | int | `200` | 单次目录批量扫描摄取文件上限，防止超大目录导致 MCP 调用超时。 |
| **向量存储配置** | | | |
| **`MCP_RAG_VECTOR_STORE`** | string | `chromadb` | 向量存储引擎类型，支持 `chromadb`（默认本地）与 `milvus`（远程分布式集群）。 |
| **`MCP_RAG_MILVUS_URI`** | string | `http://localhost:19530` | 外部 Milvus 集群接入服务地址（仅在 `VECTOR_STORE=milvus` 时生效）。 |
| **`MCP_RAG_MILVUS_TOKEN`** | string | `""` | 外部 Milvus 集群认证鉴权 Token 或 `username:password`。 |
| **元数据存储配置** | | | |
| **`MCP_RAG_METADATA_STORE`** | string | `sqlite` | 元数据与原生 FTS 存储驱动，支持 `sqlite`、`postgresql`、`mysql`。 |
| **`MCP_RAG_DATABASE_URL`** | string | `""` | 关系型数据库连接 DSN（如 `postgresql://user:pass@host:5432/db`）；优先级高于拆分变量。 |
| **`MCP_RAG_DB_HOST`** | string | `""` | 数据库主机名/IP（拆分配置项）。 |
| **`MCP_RAG_DB_PORT`** | int | `None` | 数据库端口（缺省自动适配各数据库默认端口）。 |
| **`MCP_RAG_DB_USER`** | string | `""` | 数据库认证用户名。 |
| **`MCP_RAG_DB_PASSWORD`** | string | `""` | 数据库认证密码（自动执行 URL 编码）。 |
| **`MCP_RAG_DB_NAME`** | string | `""` | 目标数据库名。 |
| **向量嵌入配置** | | | |
| **`MCP_RAG_EMBEDDING_PROVIDER`** | string | `fastembed` | 嵌入引擎类型，支持 `fastembed`、`openai`、`qwen`、`ollama`、`deepseek`。 |
| **`MCP_RAG_EMBEDDING_MODEL`** | string | `BAAI/bge-small-zh-v1.5` | 嵌入模型标识，各 Provider 具备专属默认推荐模型。 |
| **`MCP_RAG_EMBEDDING_BASE_URL`** | string | `""` | 外部 Embedding 服务基准 URL（Ollama / Qwen / OpenAI 等各具备预设默认值）。 |
| **`MCP_RAG_EMBEDDING_API_KEY`** | string | `""` | 外部 Embedding 服务鉴权 API Key（Bearer 鉴权）。 |
| **`MCP_RAG_EMBEDDING_DIMENSION`** | int | `None` | 显式指定向量维度（支持 OpenAI / Qwen 等 MRL 维度裁剪；缺省时自动发起探针探测）。 |
| **`MCP_RAG_EMBEDDING_TIMEOUT_SECONDS`** | int | `30` | 外部嵌入服务调用超时秒数（下限 `min_val=1`）。 |
| **二次重排 (Reranker) 配置** | | | |
| **`MCP_RAG_RERANK_ENABLED`** | bool | `false` | 二次重排全局开关；设为 `false` 时完全跳过 Reranker，零额外计算开销与零模型加载。 |
| **`MCP_RAG_RERANK_PROVIDER`** | string | `fastembed` | 重排驱动模式，支持 `fastembed`（本地 CPU ONNX）与 `api`（云端 HTTP API）。 |
| **`MCP_RAG_RERANK_MODEL`** | string | `BAAI/bge-reranker-base` | 重排模型名称标识（云端如 `BAAI/bge-reranker-v2-m3`）。 |
| **`MCP_RAG_RERANK_BASE_URL`** | string | `""` | 云端重排服务基准 URL（如 `https://api.siliconflow.cn/v1`，自动适配 `/v1/rerank`）。 |
| **`MCP_RAG_RERANK_API_KEY`** | string | `""` | 云端重排服务鉴权 API Key。 |
| **`MCP_RAG_RERANK_TIMEOUT_SECONDS`** | int | `30` | 云端重排 HTTP 请求超时秒数。 |
| **图片资产与对象存储配置** | | | |
| **`MCP_RAG_STORAGE_TYPE`** | string | `local` | 图片资产存储类型，支持 `local`（本地磁盘）与 `s3`（AWS S3 / MinIO）。 |
| **`MCP_RAG_S3_BUCKET`** | string | `""` | S3 存储桶名称（`storage_type=s3` 时必填）。 |
| **`MCP_RAG_S3_ENDPOINT`** | string | `""` | S3 兼容服务端点（MinIO 如 `https://minio.example.com:9000`）。 |
| **`MCP_RAG_S3_REGION`** | string | `us-east-1` | S3 存储区域。 |
| **`MCP_RAG_S3_ACCESS_KEY`** | string | `""` | S3 Access Key ID。 |
| **`MCP_RAG_S3_SECRET_KEY`** | string | `""` | S3 Secret Access Key。 |
| **`MCP_RAG_S3_PREFIX`** | string | `assets` | S3 对象存储根前缀目录。 |
| **`MCP_RAG_S3_PUBLIC_BASE_URL`** | string | `""` | S3 资产对外公共 HTTPS 直链基准域名（配置后切片内图片自动转写为此直链）。 |
| **`MCP_RAG_S3_TIMEOUT_SECONDS`** | int | `30` | S3 连接与传输超时秒数。 |
| **`MCP_RAG_ARCHIVE_REMOTE_IMAGES`** | bool | `false` | 远程图片外链自动归档开关（设为 `true` 时通过 SSRF 防御沙箱抓取外链并归档至存储桶）。 |
| **`MCP_RAG_REMOTE_IMAGE_TIMEOUT_SECONDS`** | int | `15` | 抓取外链单跳 HTTP 超时秒数。 |
| **`MCP_RAG_REMOTE_IMAGE_MAX_BYTES`** | int | `10485760` (10MB) | 抓取或提取图片最大允许字节大小（下限 `min_val=1024`）。 |

---

## 📁 支持的文件格式与切片策略

| 文件大类 | 包含后缀扩展名 | 专用切片策略与边界保护 |
| :--- | :--- | :--- |
| **Markdown 文档** | `.md`, `.markdown` | 原生感知 `#` 到 `######` 标题树结构，完整保护代码块与 `![alt](url)` 图片原子块。 |
| **Office & PDF 文档** | `.docx`, `.pdf` | 提取段落、表格与样式结构；内置图片降噪过滤（剔除噪点小图），维持上下文图文锚定。 |
| **编程源码** | `.py`, `.ts`, `.tsx`, `.js`, `.jsx`, `.java`, `.go`, `.rs`, `.sql`, `.c`, `.cpp`, `.cs`, `.sh` | 按 Class、Def、Function 及语句块自然边界切分，保留上下文可读性。 |
| **结构化数据** | `.json`, `.csv`, `.yaml`, `.yml`, `.toml` | 保留完整键值对象与行数据元组，避免字段跨切片错位断裂。 |
| **普通文本** | `.txt`, `.log`, `.rst` | 基于段落双换行符与标点符号自适应递归分块。 |
| **在线网页** | `http://` 或 `https://` 页面 | 过滤 `<script>`、`<style>`、导航与页脚等视觉噪点，提取正文并安全解析相对图片。 |

---

## 🧪 本地研发与自动化门禁验证

项目秉持严格的质量标准，要求本地验证 100% 通过后方可交付：

```bash
# 1. 静态代码风格与 Lint 检查 (0 警告)
uv run ruff check .

# 2. 代码自动排版格式校验 (100% 格式合规)
uv run ruff format --check .

# 3. 严格类型检查 (Python 3.10+ Strict Mypy)
uv run mypy src/mcp_server_rag

# 4. 全量端到端测试套件执行 (110 / 110 项用例 100% 全绿)
uv run pytest -v
```

---

## 📄 开源许可与社区参与

本项目遵循 [Apache-2.0 许可证](./LICENSE)。

- **统一领域规范与术语表**：详见 [CONTEXT.md](./CONTEXT.md)；
- **贡献者指南与规范流程**：详见 [CONTRIBUTING.md](./CONTRIBUTING.md)；
- **版本历史与发行注记**：详见 [CHANGELOG.md](./CHANGELOG.md)。

欢迎提交 [Issue](https://github.com/atengk/mcp-server-rag/issues) 提出建议或发起 [Pull Request](https://github.com/atengk/mcp-server-rag/pulls)！
