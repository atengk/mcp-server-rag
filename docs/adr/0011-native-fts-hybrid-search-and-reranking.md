# 0011: 原生 FTS + 向量混合检索与双模 Reranker 二次重排规范

## 背景与问题
纯向量余弦检索在处理精确匹配（错误代码、专有名词、特定 API 函数名）时存在语义漂移与盲区；同时，单纯的第一阶段检索得分缺乏细粒度的交叉注意力评分，制约了给大模型注入上下文的精准命中率。

## 架构决策
决定引入原生全文本检索与双模二次重排流水线：
1. **零外部中间件的原生 FTS 倒排索引**：
   - 充分复用已有的关系型元数据库：SQLite 原生 `FTS5`、PostgreSQL 原生 `tsvector/tsquery`、MySQL 原生 `FULLTEXT`；
   - 切片入库时自动同步维护倒排索引表，杜绝引入 Elasticsearch 等庞大外部中间件；
2. **混合检索评分融合**：
   - 支持高维稠密向量语义相似度与 BM25 全文检索并发召回；
   - 采用标准 RRF（Reciprocal Rank Fusion）倒数排名融合算法计算综合排序得分；
3. **双模可插拔 Reranker**：
   - 默认按需开启，对初筛候选切片执行二次精排；
   - **本地模式**：基于 FastEmbed 本地运行 `BAAI/bge-reranker-base`（ONNX 纯 CPU 毫秒级打分）；
   - **云端模式**：支持调用兼容 SiliconFlow、Cohere、百炼等标准的 Rerank HTTP 接口。
