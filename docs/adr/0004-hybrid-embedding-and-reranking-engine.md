# 0004: 采用 FastEmbed + OpenAI-Compatible 多引擎向量化方案

在向量嵌入与重排技术选型中，为了杜绝直接依赖动辄数 GB 的 PyTorch / HuggingFace Transformers 带来的包体积膨胀与构建困难，决定采用 FastEmbed（基于 ONNX Runtime）作为本地轻量 CPU 推理核心，默认搭载紧凑高效的中英文多语言模型；同时提供 OpenAI 兼容协议适配器，支持按需接入 OpenAI、Ollama、通义千问或 SiliconFlow 等云端/本地服务。
