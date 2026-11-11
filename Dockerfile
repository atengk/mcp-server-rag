# ==============================================================================
# 多阶段轻量生产镜像构建 (Python 3.12 + uv)
# ==============================================================================

# 构建阶段
FROM python:3.15-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 优先安装依赖层以最大化 Docker 缓存利用
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache "."

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache --no-deps "."

# 运行时阶段
FROM python:3.15-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# 从构建阶段复制已安装的包与二进制命令
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/mcp-server-rag /usr/local/bin/mcp-server-rag
COPY --from=builder /usr/local/bin/atengk-mcp-server-rag /usr/local/bin/atengk-mcp-server-rag

# 创建非 root 安全运行用户
RUN useradd -u 10001 -m appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["mcp-server-rag"]
