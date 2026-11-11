# RAG MCP 领域模型与统一语言 (RAG MCP Context)

本领域上下文定义了通过 Model Context Protocol (MCP) 驱动与管理检索增强生成 (RAG) 知识库、切片索引与混合检索的核心概念与通用语言，确保所有设计、代码符号与 Agent 交互使用一致的术语。

## 统一领域语言 (Language)

### 知识库与数据模型 (Knowledge Base & Data Model)

**Knowledge Base (知识库)**:
用于组织特定领域或业务范畴文档集合与索引配置的逻辑命名空间。
_Avoid_: Database, Collection, Folder, 数据库

**Document (原始文档)**:
由外部输入或知识源导入的非结构化/半结构化完整数据实体（如 Markdown、PDF、HTML 或文本）。
_Avoid_: File, Blob, Text Data, 散落文件

**Chunk (切片 / 文本块)**:
Document 经过文本切分器按照特定长度与重叠窗口处理后的独立语义片段单元。
_Avoid_: Segment, Paragraph, Section, Snippet, 段落

**Embedding (向量嵌入)**:
通过稠密向量模型将 Chunk 文本语义映射为高维浮点数稠密向量的操作与输出结果。
_Avoid_: Vectorization, Encoding, Representation, 向量化

**Vector Store (向量存储)**:
持久化并索引 Embedding 向量及 Chunk 元数据以支持近邻相似度搜索（ANN）的存储引擎。
_Avoid_: Database, Search Engine, Cache

### 检索与增强机制 (Retrieval & Augmentation)

**Hybrid Search (混合检索)**:
结合稠密向量相似度检索（Dense Retrieval）与基于关键词/BM25 的稀疏检索（Sparse Retrieval）的复合检索机制。
_Avoid_: Multi-search, Dual-mode, Mixed Query, 双重搜索

**Top-K (检索切片数)**:
根据相似度得分从 Vector Store 中检索返回的最优 Chunk 数量上限。
_Avoid_: Limit, Page Size, Count, 返回数量

**Similarity Threshold (相似度阈值)**:
过滤与查询语义距离过大、相关度不足的 Chunk 的最低评分截断阈值。
_Avoid_: Cutoff, Min Score, Filter Level, 最低分

**Context Injection (上下文注入)**:
将检索出的关联 Chunk 按设定模板组装并注入到大模型对话或提示词中的增强过程。
_Avoid_: Prompt Stuffing, Text Appending, Context Merging, 提示词拼接
