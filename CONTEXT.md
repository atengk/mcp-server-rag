# RAG MCP 领域模型与统一语言 (RAG MCP Context)

本领域上下文定义了通过 Model Context Protocol (MCP) 驱动与管理检索增强生成 (RAG) 知识库、切片索引、多引擎向量化与混合检索的核心概念与通用语言，确保所有架构设计、代码符号、环境变量与 Agent 交互使用一致的术语。

## 统一领域语言 (Language)

### 知识库与数据模型 (Knowledge Base & Data Model)

**Knowledge Base (知识库)**:
用于组织特定领域或业务范畴文档集合与索引配置的逻辑命名空间。
_Avoid_: Database, Collection, Folder, 数据库, 目录

**Document (原始文档)**:
由外部输入（本地文件、纯文本、在线 URL 网页）导入的非结构化/半结构化完整数据实体（如 Markdown、PDF、HTML、代码文件）。
_Avoid_: File, Blob, Text Data, 散落文件

**Chunk (切片 / 文本块)**:
Document 经过文本切分器按照特定长度、语法边界与重叠窗口处理后的独立语义片段单元。
_Avoid_: Segment, Paragraph, Section, Snippet, 段落

**Document Fingerprint (文档内容指纹)**:
基于文档正文计算的 SHA-256 唯一摘要，用于在重新索引时实施幂等更新（Idempotent Upsert）与防重复去重。
_Avoid_: File Hash, Checksum, MD5

**Chunk Metadata (切片元数据)**:
附加在 Chunk 上的结构化描述信息（包含 `doc_id`、来源文件路径、标题、切片序号、文件类型与自定义标签）。
_Avoid_: Tag, Property, Attribute, 附加信息

### 向量模型与计算 (Embedding & Reranker)

**Embedding Provider (向量嵌入提供方)**:
负责将 Chunk 文本转换为高维浮点数稠密向量的计算引擎（支持 FastEmbed 本地推理、Ollama 或 OpenAI 兼容 API）。
_Avoid_: Vectorizer, Encoder, 模型接口

**Reranker (重排器)**:
对初步检索召回的候选 Chunk 集合与查询进行细粒度交叉注意力评分，重新计算相关度排名的模型。
_Avoid_: Secondary Ranker, Sorter, 二次排序器

**Dense Vector (稠密向量)**:
由嵌入模型输出的定长连续浮点数数组（通常为 384、512 或 1536 维）。
_Avoid_: Float Array, Tensor, 特征值

### 向量存储与混合检索 (Vector Store & Hybrid Retrieval)

**Vector Store Adapter (向量存储适配器)**:
封装底层向量数据库读写交互的抽象驱动接口（如内置 ChromaDB 适配器与外部 Milvus 适配器）。
_Avoid_: DB Client, Connector, 存储引擎

**Hybrid Search (混合检索)**:
结合稠密向量相似度检索（Dense Retrieval）与精准关键词/BM25 过滤的复合检索机制。
_Avoid_: Multi-search, Dual-mode, Mixed Query, 双重搜索

**Cross-KB Search (跨库联合检索)**:
当 `kb_name="*"` 或指定知识库列表时，跨多个知识库并行召回候选切片并按置信度统一重排归并的机制。
_Avoid_: Federated Search, Global Search, 全局查

**Scoped Filter (范围过滤)**:
在检索阶段通过元数据字典限制搜索边界的机制（如指定来源文档、文件后缀或目录前缀）。
_Avoid_: Tag Filter, Where Clause, 属性过滤

**Top-K (检索切片数)**:
根据相似度得分从 Vector Store 中检索返回的最优 Chunk 数量上限。
_Avoid_: Limit, Page Size, Count, 返回数量

**Similarity Threshold (相似度阈值)**:
过滤与查询语义距离过大、相关度不足的 Chunk 的最低评分截断阈值（取值范围 0.0 ~ 1.0）。
_Avoid_: Cutoff, Min Score, Filter Level, 最低分

**Context Injection (上下文注入)**:
将检索出的关联 Chunk 按带来源引用的 Markdown 结构化模板组装并注入大模型上下文的增强过程。
_Avoid_: Prompt Stuffing, Text Appending, Context Merging, 提示词拼接

### 存储架构与中间件约束 (Storage & Zero-Middleware Invariant)

**Zero-Middleware Architecture (零中间件自闭环架构)**:
坚决杜绝强制引入 MySQL、Redis 等外部重型中间件，元数据与向量索引全部收敛于本地嵌入式存储（或可选远程 Milvus），确保 MCP 工具秒级自足运行。
_Avoid_: MySQL Backend, Redis Cache, 外部中间件强绑定

**Metadata Catalog (元数据目录库)**:
用于持久化记录知识库列表、文档指纹、切片统计与时间戳的轻量关系型存储（默认基于 Python 内建 SQLite，零外部依赖）。
_Avoid_: JSON State, Flat File, MySQL

### 运维与环境安全 (Operations & Security)

**MCP_RAG 环境变量契约 (Configuration Contract)**:
本项目所有运行时系统配置统一遵循 `MCP_RAG_` 命名空间前缀（如 `MCP_RAG_DATA_DIR`、`MCP_RAG_VECTOR_STORE`、`MCP_RAG_READ_ONLY` 等）。
_Avoid_: RAG_*, APP_*, CONFIG_*

**Read-Only Guard (只读安全门禁)**:
当环境变量 `MCP_RAG_READ_ONLY=true` 时，在 MCP 协议握手层物理隐藏所有写操作、索引、建库与删除类工具，防止 Agent 越权修改知识库。
_Avoid_: Permission Interceptor, Write Lock, 安全锁

**Idempotent Ingestion (幂等摄取)**:
对相同路径或内容的文档反复执行索引时，系统保证状态一致、旧切片原子替换且不发生重复堆积的特性。
_Avoid_: Re-index, Overwrite, 重复写入

**Doctor Probe (诊断探针)**:
对向量存储引擎连通性、本地模型缓存、磁盘占用与环境健康度进行全维自检的运维工具。
_Avoid_: Healthcheck, Status Checker, 检查器
