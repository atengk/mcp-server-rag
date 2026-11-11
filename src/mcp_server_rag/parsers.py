"""
多格式结构感知解析与语法切片管道。

针对 Markdown、源码文件、结构化文件与纯文本提供零外部依赖的结构感知切分：
- Markdown：保留章节标题树上下文，保护代码块不被从中切断；
- Source Code：基于类与函数声明行语法边界切分；
- Structured Data (JSON/CSV/YAML)：基于顶层键、记录行与表头保留切分；
- Plain Text：基于段落与标点层级递归切分。

@author Ateng
@since 2026-11-11
"""

import csv
import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from mcp_server_rag.splitter import recursive_split_text

# 常见源代码文件扩展名映射
CODE_EXTENSIONS: set[str] = {
    ".py",
    ".ts",
    ".js",
    ".jsx",
    ".tsx",
    ".go",
    ".java",
    ".rs",
    ".sql",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".php",
    ".rb",
    ".sh",
    ".bash",
}

# 结构化文件扩展名
STRUCTURED_EXTENSIONS: set[str] = {
    ".json",
    ".csv",
    ".yaml",
    ".yml",
}


def split_markdown(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> list[str]:
    """对 Markdown 文本执行标题层级感知与代码块保护切分。

    Args:
        text: 待切分的 Markdown 正文文本
        chunk_size: 目标切片字符数上限
        chunk_overlap: 切片滑动重叠字符数

    Returns:
        list[str]: 切分后的 Markdown 切片列表
    """
    if not text or not text.strip():
        return []

    lines = text.splitlines(keepends=True)
    blocks: list[dict[str, Any]] = []

    # 1. 第一遍扫描：识别围栏代码块并保护完整性
    in_code_block = False
    code_fence = ""
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        is_fence_start = (stripped.startswith("```") or stripped.startswith("~~~")) and (
            not in_code_block
        )
        is_fence_end = (
            in_code_block
            and (stripped.startswith("```") or stripped.startswith("~~~"))
            and (code_fence == stripped[:3])
        )

        if is_fence_start:
            if current_lines:
                blocks.append({"type": "text", "lines": current_lines})
                current_lines = []
            in_code_block = True
            code_fence = stripped[:3]
            current_lines.append(line)
        elif is_fence_end:
            current_lines.append(line)
            blocks.append({"type": "code", "lines": current_lines})
            current_lines = []
            in_code_block = False
            code_fence = ""
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append(
            {
                "type": "code" if in_code_block else "text",
                "lines": current_lines,
            }
        )

    # 2. 第二遍扫描：按照 Markdown 标题层级组装 Section
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")
    sections: list[dict[str, Any]] = []
    current_section_headers: list[tuple[int, str]] = []
    current_section_blocks: list[str] = []

    def flush_section() -> None:
        nonlocal current_section_blocks
        if current_section_blocks:
            sections.append(
                {
                    "headers": list(current_section_headers),
                    "text": "".join(current_section_blocks).strip(),
                }
            )
            current_section_blocks = []

    for block in blocks:
        if block["type"] == "code":
            current_section_blocks.append("".join(block["lines"]))
        else:
            text_lines: list[str] = block["lines"]
            for line in text_lines:
                match = header_pattern.match(line.strip())
                if match:
                    flush_section()
                    level = len(match.group(1))
                    title = match.group(2).strip()
                    # 维护标题栈层级
                    while current_section_headers and current_section_headers[-1][0] >= level:
                        current_section_headers.pop()
                    current_section_headers.append((level, title))
                    current_section_blocks.append(line)
                else:
                    current_section_blocks.append(line)

    flush_section()

    if not sections:
        return recursive_split_text(text, chunk_size, chunk_overlap)

    # 3. 聚合与输出切片，为每个切片添加标题面包屑前缀
    chunks: list[str] = []
    for sec in sections:
        header_titles = [h[1] for h in sec["headers"]]
        prefix = f"# {' > '.join(header_titles)}\n\n" if header_titles else ""
        content = sec["text"]

        # 若整个 Section 包含前缀不超过 chunk_size，直接作为一个切片
        full_text = prefix + content if prefix else content
        if len(full_text) <= chunk_size:
            chunks.append(full_text.strip())
            continue

        # 若超过 chunk_size，在保护代码块的前提下按段落分割
        paragraphs = re.split(r"(\n\n+)", content)
        buf: list[str] = []
        buf_len = len(prefix)

        for p in paragraphs:
            if buf_len + len(p) > chunk_size and buf:
                chunk_str = (prefix + "".join(buf)).strip()
                if chunk_str:
                    chunks.append(chunk_str)
                buf = []
                buf_len = len(prefix)

            if len(p) > chunk_size:
                # 极端超长段落/代码块内部切片
                sub_chunks = recursive_split_text(
                    p,
                    chunk_size=chunk_size - len(prefix),
                    chunk_overlap=chunk_overlap,
                )
                for sc in sub_chunks:
                    chunks.append((prefix + sc).strip())
            else:
                buf.append(p)
                buf_len += len(p)

        if buf:
            chunk_str = (prefix + "".join(buf)).strip()
            if chunk_str:
                chunks.append(chunk_str)

    return chunks if chunks else [text.strip()]


def split_code(
    code: str,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> list[str]:
    """对源代码文件基于类、函数与语法定义边界进行结构感知切分。

    Args:
        code: 源代码文件正文
        chunk_size: 目标切片字符数上限
        chunk_overlap: 切片滑动重叠字符数

    Returns:
        list[str]: 语法块切片列表
    """
    if not code or not code.strip():
        return []

    if len(code) <= chunk_size:
        return [code.strip()]

    # 定义多语言通用的顶层定义正则（类、函数、结构体、接口等）
    def_pattern = re.compile(
        r"^(?:"
        r"(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:def|class|function|interface|type)\b|"
        r"(?:pub\s+)?(?:fn|struct|enum|impl|trait)\b|"
        r"func\b|"
        r"(?:public|private|protected)\s+(?:static\s+)?(?:class|interface|enum|void|\w+)\b|"
        r"(?:CREATE|ALTER|DROP)\s+(?:TABLE|VIEW|PROCEDURE|FUNCTION)\b"
        r")",
        re.MULTILINE,
    )

    lines = code.splitlines(keepends=True)
    blocks: list[str] = []
    current_block: list[str] = []

    for line in lines:
        if def_pattern.match(line) and current_block:
            blocks.append("".join(current_block))
            current_block = [line]
        else:
            current_block.append(line)

    if current_block:
        blocks.append("".join(current_block))

    # 按照 chunk_size 合并或切分代码块
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for b in blocks:
        if current_len + len(b) > chunk_size and current_chunk:
            chunks.append("".join(current_chunk).strip())
            current_chunk = []
            current_len = 0

        if len(b) > chunk_size:
            # 单个超长函数/类，按换行符切分
            sub_chunks = recursive_split_text(
                b,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ";\n", ";", " ", ""],
            )
            chunks.extend(sub_chunks)
        else:
            current_chunk.append(b)
            current_len += len(b)

    if current_chunk:
        chunks.append("".join(current_chunk).strip())

    return [c for c in chunks if c.strip()]


def split_structured(
    content: str,
    format_type: str,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> list[str]:
    """对结构化文件（JSON、CSV、YAML）基于记录与表头进行切分。

    Args:
        content: 文件正文内容
        format_type: 文件格式类型（json, csv, yaml）
        chunk_size: 切片字符数上限
        chunk_overlap: 切片重叠字符数

    Returns:
        list[str]: 格式化后的结构切片列表
    """
    if not content or not content.strip():
        return []

    format_lower = format_type.lower().strip(".")

    # 1. JSON 格式切分
    if format_lower == "json":
        try:
            data = json.loads(content)
            if isinstance(data, list):
                # 记录列表：每个/批元素作为一个切片
                chunks: list[str] = []
                cur_items: list[Any] = []
                for item in data:
                    cur_items.append(item)
                    rendered = json.dumps(cur_items, ensure_ascii=False, indent=2)
                    if len(rendered) > chunk_size and len(cur_items) > 1:
                        cur_items.pop()
                        chunks.append(json.dumps(cur_items, ensure_ascii=False, indent=2))
                        cur_items = [item]
                if cur_items:
                    chunks.append(json.dumps(cur_items, ensure_ascii=False, indent=2))
                return chunks
            elif isinstance(data, dict):
                # 顶层键对象：每个键值作为一个切片
                chunks = []
                for k, v in data.items():
                    item_str = f"[{k}]\n" + json.dumps(v, ensure_ascii=False, indent=2)
                    if len(item_str) <= chunk_size:
                        chunks.append(item_str)
                    else:
                        sub = recursive_split_text(item_str, chunk_size, chunk_overlap)
                        chunks.extend(sub)
                return chunks if chunks else [content.strip()]
        except Exception:
            return recursive_split_text(content, chunk_size, chunk_overlap)

    # 2. CSV 格式切分（保留表头）
    elif format_lower == "csv":
        try:
            reader = csv.reader(io.StringIO(content))
            rows = list(reader)
            if not rows:
                return []
            header = rows[0]
            header_line = "Headers: " + ", ".join(header) + "\n"

            chunks = []
            cur_rows: list[list[str]] = []
            for r in rows[1:]:
                cur_rows.append(r)
                rendered = header_line + "\n".join(", ".join(row) for row in cur_rows)
                if len(rendered) > chunk_size and len(cur_rows) > 1:
                    cur_rows.pop()
                    chunks.append(header_line + "\n".join(", ".join(row) for row in cur_rows))
                    cur_rows = [r]
            if cur_rows:
                chunks.append(header_line + "\n".join(", ".join(row) for row in cur_rows))
            return chunks if chunks else [content.strip()]
        except Exception:
            return recursive_split_text(content, chunk_size, chunk_overlap)

    # 3. YAML 格式切分
    elif format_lower in ("yaml", "yml"):
        # 按 --- 文档分隔符或顶层定义切分
        docs = re.split(r"\n---\s*\n", content)
        chunks = []
        for doc in docs:
            doc_str = doc.strip()
            if not doc_str:
                continue
            if len(doc_str) <= chunk_size:
                chunks.append(doc_str)
            else:
                chunks.extend(recursive_split_text(doc_str, chunk_size, chunk_overlap))
        return chunks if chunks else [content.strip()]

    return recursive_split_text(content, chunk_size, chunk_overlap)


def dispatch_and_split(
    file_path: Path,
    content: str,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> list[str]:
    """根据文件扩展名自动分发至对应的结构感知切分器。

    Args:
        file_path: 文件路径（用于判断后缀）
        content: 文件正文内容
        chunk_size: 目标切片字符数上限
        chunk_overlap: 切片滑动重叠字符数

    Returns:
        list[str]: 切分后的文本切片列表
    """
    ext = file_path.suffix.lower()

    if ext in (".md", ".mdx"):
        return split_markdown(content, chunk_size, chunk_overlap)
    elif ext in CODE_EXTENSIONS:
        return split_code(content, chunk_size, chunk_overlap)
    elif ext in STRUCTURED_EXTENSIONS:
        return split_structured(content, ext, chunk_size, chunk_overlap)
    else:
        return recursive_split_text(content, chunk_size, chunk_overlap)


class HTMLTextExtractor(HTMLParser):
    """从 HTML 页面中提取网页标题与正文纯文本，滤除脚本、样式与导航视觉噪点。"""

    IGNORED_TAGS: set[str] = {
        "script",
        "style",
        "noscript",
        "nav",
        "header",
        "footer",
        "svg",
        "form",
        "iframe",
    }

    BLOCK_TAGS: set[str] = {
        "p",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "br",
        "section",
        "article",
        "table",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.title: str = ""
        self._in_title: bool = False
        self._ignore_stack: list[str] = []
        self._pieces: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower == "title":
            self._in_title = True
        elif tag_lower in self.IGNORED_TAGS:
            self._ignore_stack.append(tag_lower)

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower == "title":
            self._in_title = False
        elif self._ignore_stack and self._ignore_stack[-1] == tag_lower:
            self._ignore_stack.pop()
        elif tag_lower in self.BLOCK_TAGS:
            self._pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data.strip()
        elif not self._ignore_stack:
            text = data.strip()
            if text:
                self._pieces.append(text + " ")


def extract_html_content(html: str) -> tuple[str, str]:
    """提取 HTML 页面的标题与主体清洗正文。

    Args:
        html: 原始 HTML 文本字符串

    Returns:
        tuple[str, str]: (页面标题, 清洗后的正文纯文本)
    """
    extractor = HTMLTextExtractor()
    try:
        extractor.feed(html)
        extractor.close()
    except Exception:
        pass

    raw_text = "".join(extractor._pieces)
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    cleaned_text = "\n\n".join(lines)
    return extractor.title.strip(), cleaned_text
