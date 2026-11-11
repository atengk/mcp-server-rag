"""
端到端测试：多格式文件与目录批量摄取管道 (Issue #6)。

在单一最高接缝（FastMCP Tool 协议执行层边界）测试 rag_index_file 与 rag_index_directory
的多格式语法感知切分、代码块保护、目录扫描与 max_files 截断安全防护。

@author Ateng
@since 2026-11-11
"""

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from mcp_server_rag.server import (
    rag_create_kb,
    rag_index_directory,
    rag_index_file,
    rag_search,
)


@pytest.fixture(autouse=True)
def test_data_dir() -> Generator[Path, None, None]:
    """为每个测试用例隔离独立的临时数据与测试文件目录，并在测试后清理。"""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_files_")
    old_data_dir = os.environ.get("MCP_RAG_DATA_DIR")
    os.environ["MCP_RAG_DATA_DIR"] = temp_dir
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    yield Path(temp_dir)
    if old_data_dir is not None:
        os.environ["MCP_RAG_DATA_DIR"] = old_data_dir
    else:
        os.environ.pop("MCP_RAG_DATA_DIR", None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_index_markdown_file_headers_and_code_blocks(test_data_dir: Path) -> None:
    """验证 Markdown 文件摄取：保留章节标题树上下文，代码块不被从中切断。"""
    rag_create_kb(name="kb_markdown", description="Markdown 测试知识库")

    # 创建测试 Markdown 文件
    md_file = test_data_dir / "architecture.md"
    md_content = (
        "# 系统核心架构\n\n"
        "本文档详细阐述分布式 RAG 服务端与客户端通信架构。\n\n"
        "## 存储与向量引擎\n\n"
        "向量存储采用轻量级嵌入式 ChromaDB 实现。\n\n"
        "```python\n"
        "def connect_chroma(data_dir: str):\n"
        "    client = chromadb.PersistentClient(path=data_dir)\n"
        "    return client\n"
        "```\n\n"
        "## 检索调度模块\n\n"
        "调度模块负责处理来自 Agent 的工具检索请求与评分阈值过滤。\n"
    )
    md_file.write_text(md_content, encoding="utf-8")

    # 1. 首次索引
    res: dict[str, Any] = rag_index_file(
        kb_name="kb_markdown",
        file_path=str(md_file),
    )

    assert res["success"] is True
    assert res["status"] == "indexed"
    assert res["chunk_count"] >= 1
    assert "content_hash" in res

    # 2. 验证检索结果包含标题树上下文与完整代码块
    search_results = rag_search(
        kb_name="kb_markdown",
        query="如何连接 ChromaDB 向量存储引擎",
        top_k=2,
    )
    assert len(search_results) >= 1
    top_chunk = search_results[0]
    # 验证代码块完整无腰斩
    assert "def connect_chroma(data_dir: str):" in top_chunk["text"]
    assert "return client" in top_chunk["text"]
    # 验证切片包含章节标题上下文
    assert "存储与向量引擎" in top_chunk["text"]

    # 3. 验证幂等跳过
    res_skip = rag_index_file(
        kb_name="kb_markdown",
        file_path=str(md_file),
    )
    assert res_skip["success"] is True
    assert res_skip["status"] == "skipped"


def test_index_code_file_syntax_boundaries(test_data_dir: Path) -> None:
    """验证源代码文件摄取：基于类与函数声明行语法边界切分与增量更新。"""
    rag_create_kb(name="kb_code", description="源码测试知识库")

    py_file = test_data_dir / "services.py"
    py_content = (
        '"""业务服务层核心代码模块。"""\n\n'
        "class UserService:\n"
        '    """用户领域核心管理服务。"""\n'
        "    def get_user_by_id(self, user_id: int):\n"
        '        return {"user_id": user_id, "name": "Ateng"}\n\n'
        "class OrderService:\n"
        '    """订单处理领域服务。"""\n'
        "    def create_order(self, order_id: str, amount: float):\n"
        '        return {"order_id": order_id, "amount": amount}\n'
    )
    py_file.write_text(py_content, encoding="utf-8")

    # 1. 索引源码文件
    res = rag_index_file(kb_name="kb_code", file_path=str(py_file))
    assert res["success"] is True
    assert res["status"] == "indexed"
    assert res["chunk_count"] >= 1

    # 2. 检索类与函数定义
    user_results = rag_search(
        kb_name="kb_code",
        query="用户查询服务 get_user_by_id",
        top_k=1,
    )
    assert len(user_results) == 1
    assert "UserService" in user_results[0]["text"]
    assert "get_user_by_id" in user_results[0]["text"]

    order_results = rag_search(
        kb_name="kb_code",
        query="订单处理与创建 create_order",
        top_k=1,
    )
    assert len(order_results) == 1
    assert "OrderService" in order_results[0]["text"]

    # 3. 修改源码文件并验证原子替换更新
    updated_py_content = py_content + (
        "\nclass PaymentService:\n"
        '    """支付网关交互服务。"""\n'
        "    def pay(self, tx_id: str):\n"
        '        return {"tx_id": tx_id, "status": "success"}\n'
    )
    py_file.write_text(updated_py_content, encoding="utf-8")

    res_update = rag_index_file(kb_name="kb_code", file_path=str(py_file))
    assert res_update["success"] is True
    assert res_update["status"] == "updated"
    assert res_update["content_hash"] != res["content_hash"]


def test_index_structured_files(test_data_dir: Path) -> None:
    """验证结构化文件（JSON与CSV）摄取：按顶层键与记录行保留表头切分。"""
    rag_create_kb(name="kb_structured", description="结构化数据知识库")

    # 1. JSON 配置文件
    json_file = test_data_dir / "settings.json"
    json_content = (
        "{\n"
        '  "database": {\n'
        '    "host": "127.0.0.1",\n'
        '    "port": 5432,\n'
        '    "db_name": "prod_users_db"\n'
        "  },\n"
        '  "redis_cache": {\n'
        '    "host": "127.0.0.1",\n'
        '    "port": 6379,\n'
        '    "ttl_seconds": 3600\n'
        "  }\n"
        "}"
    )
    json_file.write_text(json_content, encoding="utf-8")

    res_json = rag_index_file(kb_name="kb_structured", file_path=str(json_file))
    assert res_json["success"] is True
    assert res_json["status"] == "indexed"

    search_json = rag_search(
        kb_name="kb_structured",
        query="数据库配置 端口 与 db_name",
        top_k=1,
    )
    assert len(search_json) == 1
    assert "database" in search_json[0]["text"]
    assert "5432" in search_json[0]["text"]

    # 2. CSV 数据文件
    csv_file = test_data_dir / "employees.csv"
    csv_content = (
        "emp_id,emp_name,department,job_title\n"
        "101,张三,研发部,后端资深专家\n"
        "102,李四,运维部,SRE系统工程师\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    res_csv = rag_index_file(kb_name="kb_structured", file_path=str(csv_file))
    assert res_csv["success"] is True
    assert res_csv["status"] == "indexed"

    search_csv = rag_search(
        kb_name="kb_structured",
        query="张三 研发部 后端专家",
        top_k=1,
    )
    assert len(search_csv) == 1
    assert "张三" in search_csv[0]["text"]
    assert "Headers:" in search_csv[0]["text"] or "emp_name" in search_csv[0]["text"]


def test_index_directory_batch_and_max_files_guard(test_data_dir: Path) -> None:
    """验证目录批量摄取、递归扫描、隐藏目录与二进制过滤及 max_files 超时防御截断。"""
    rag_create_kb(name="kb_dir", description="目录批量摄取知识库")

    docs_dir = test_data_dir / "my_project"
    docs_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = docs_dir / "sub"
    sub_dir.mkdir(parents=True, exist_ok=True)
    git_dir = docs_dir / ".git"
    git_dir.mkdir(parents=True, exist_ok=True)

    # 创建 5 个有效文本文件
    (docs_dir / "readme.md").write_text("# Readme\n这是项目根目录说明文档。", encoding="utf-8")
    (docs_dir / "guide.txt").write_text("这是系统新手引导配置说明。", encoding="utf-8")
    (sub_dir / "api.py").write_text("def fetch_data():\n    return 42\n", encoding="utf-8")
    (sub_dir / "config.json").write_text('{"env": "production"}', encoding="utf-8")
    (sub_dir / "data.csv").write_text("id,val\n1,alpha\n2,beta\n", encoding="utf-8")

    # 创建应该被忽略的隐藏文件与二进制文件
    (git_dir / "config").write_text("git config content", encoding="utf-8")
    (docs_dir / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    # 1. 设置 max_files = 3 触发安全截断
    res_truncated = rag_index_directory(
        kb_name="kb_dir",
        dir_path=str(docs_dir),
        glob_pattern="**/*",
        max_files=3,
        recursive=True,
    )

    assert res_truncated["success"] is True
    assert res_truncated["truncated"] is True
    assert res_truncated["total_found"] == 5  # logo.png 和 .git 被过滤后共 5 个文件
    assert res_truncated["total_indexed"] == 3
    assert res_truncated["indexed_count"] == 3
    assert "max_files" in res_truncated["message"] or "截断" in res_truncated["message"]

    # 2. 再次执行全量摄取（max_files = 50），验证前 3 个文件秒级跳过，剩余 2 个完成摄取
    res_full = rag_index_directory(
        kb_name="kb_dir",
        dir_path=str(docs_dir),
        glob_pattern="**/*",
        max_files=50,
        recursive=True,
    )

    assert res_full["success"] is True
    assert res_full["truncated"] is False
    assert res_full["total_found"] == 5
    assert res_full["total_indexed"] == 5
    assert res_full["skipped_count"] == 3
    assert res_full["indexed_count"] == 2

    # 3. 不存在的目录检查
    res_err = rag_index_directory(
        kb_name="kb_dir",
        dir_path=str(test_data_dir / "not_exist_folder"),
    )
    assert res_err["success"] is False
