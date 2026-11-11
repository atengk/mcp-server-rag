"""
全文检索分词与原生数据库 FTS 查询表达式构造工具。

提供 SQLite FTS5、PostgreSQL tsvector 与 MySQL FULLTEXT 共享的中英文混合分词
（含 CJK 二元切片与错误码/堆栈/CVE 等复合标识符规范化）及安全防注入查询语句构造方法，
确保精确错误码、专有名词与中文词组在无需外部扩展的条件下具备高召回率与精确度。

@author Ateng
@since 2026-11-11
"""

import re
from collections.abc import Iterator
from typing import Final

# 查询词元数量上限，防止超长查询触发数据库全文解析栈溢出
_MAX_QUERY_TERMS: Final[int] = 64

# 匹配连续复合 ASCII 标识符（如 CVE-2026-0001、ERR_GATEWAY_TIMEOUT_504、java.lang.NPE）、
# 独立 ASCII 字母数字串（如 0x80070005、FastMCP）以及连续中文汉字片段
_TOKEN_SEGMENT_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<compound>[A-Za-z0-9]+(?:[._:/-][A-Za-z0-9]+)+)"
    r"|(?P<ascii>[A-Za-z0-9]+)"
    r"|(?P<cjk>[\u4e00-\u9fff]+)"
)

_COMPOUND_SEP_RE: Final[re.Pattern[str]] = re.compile(r"[._:/-]+")
_MYSQL_SAFE_SEP_RE: Final[re.Pattern[str]] = re.compile(r"[.:/]+")
_HEX_LITERAL_RE: Final[re.Pattern[str]] = re.compile(r"^0[xX][0-9a-fA-F]+$")
_CAMEL_PART_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[0-9]+"
)
_PG_SAFE_TERM_RE: Final[re.Pattern[str]] = re.compile(r"[^0-9a-z_\u4e00-\u9fff]")

# 仅在存在其他有效检索词时过滤单独出现的中文虚词，减少噪音召回
_CJK_SINGLE_STOPWORDS: Final[frozenset[str]] = frozenset(
    {"的", "了", "是", "在", "和", "与", "或", "也", "就", "都", "而", "及", "着", "过"}
)


def _iter_segments(text: str) -> Iterator[tuple[str, str]]:
    """按扫描顺序将输入文本拆分为 ``(segment_kind, raw_value)`` 三类规范片段。"""
    for match in _TOKEN_SEGMENT_RE.finditer(text):
        for kind in ("compound", "ascii", "cjk"):
            value = match.group(kind)
            if value is not None:
                yield kind, value
                break


def _split_camel_subparts(token: str) -> list[str]:
    """将驼峰或混排 ASCII 词元拆分为小写子词（十六进制字面量保持完整不拆分）。"""
    if _HEX_LITERAL_RE.match(token) or token.islower() or token.isupper() or token.isdigit():
        return []
    parts = [p.lower() for p in _CAMEL_PART_RE.findall(token) if p]
    return parts if len(parts) > 1 else []


def tokenize_fts_text(text: str) -> str:
    """将切片正文预分词为空格分隔的标准化倒排索引词元串。

    同时保留复合错误码整体标识符（下划线规范化）、切分后子词、驼峰拆词
    以及中文汉字 Unigram + Bigram 序列，供 SQLite FTS5 与 PostgreSQL ``to_tsvector`` 使用。

    Args:
        text: 待建索引的原始切片文本。

    Returns:
        空格分隔的标准化索引文本；若输入为空白则返回空字符串 ``""``。
    """
    if not text or not text.strip():
        return ""

    tokens: list[str] = []
    for kind, value in _iter_segments(text):
        if kind == "compound":
            canonical = _COMPOUND_SEP_RE.sub("_", value).lower()
            tokens.append(canonical)
            for part in _COMPOUND_SEP_RE.split(value):
                if not part:
                    continue
                lower_part = part.lower()
                if lower_part != canonical:
                    tokens.append(lower_part)
                tokens.extend(_split_camel_subparts(part))
        elif kind == "ascii":
            lower_word = value.lower()
            tokens.append(lower_word)
            for subpart in _split_camel_subparts(value):
                if subpart != lower_word:
                    tokens.append(subpart)
        else:
            tokens.extend(value)
            for idx in range(len(value) - 1):
                tokens.append(value[idx : idx + 2])

    return " ".join(tokens)


def extract_fts_query_terms(query: str) -> list[str]:
    """从检索查询文本中提取去重且有序的全文检索词元列表。

    复合标识符（如 ``CVE-2026-0001``、``ERR_GATEWAY_TIMEOUT_504``）在查询阶段
    仅保留规范化后的完整联合词元，避免纯数字子段引发无关文档误召回；
    长度为 2 及以上的连续中文片段提取二元组（Bigram），单字仅在必要时保留。

    Args:
        query: 用户输入的原始检索问题。

    Returns:
        去重并限长后的标准检索词元列表；若查询无有效词元则返回空列表 ``[]``。
    """
    if not query or not query.strip():
        return []

    raw_terms: list[str] = []
    single_cjk_fallbacks: list[str] = []

    for kind, value in _iter_segments(query):
        if kind == "compound":
            raw_terms.append(_COMPOUND_SEP_RE.sub("_", value).lower())
        elif kind == "ascii":
            raw_terms.append(value.lower())
        elif len(value) == 1:
            if value in _CJK_SINGLE_STOPWORDS:
                single_cjk_fallbacks.append(value)
            else:
                raw_terms.append(value)
        else:
            for idx in range(len(value) - 1):
                raw_terms.append(value[idx : idx + 2])

    candidates = raw_terms if raw_terms else single_cjk_fallbacks
    deduplicated = list(dict.fromkeys(term for term in candidates if term))
    return deduplicated[:_MAX_QUERY_TERMS]


def build_sqlite_fts5_query(query: str) -> str:
    """构造安全的 SQLite FTS5 ``MATCH`` 查询字符串。

    所有词元均经双引号包裹转义并以 ``OR`` 连接，彻底规避 ``-``、``.``、``:``、``*``
    等特殊符号引起的 ``fts5: syntax error`` 异常。

    Args:
        query: 用户检索查询文本。

    Returns:
        适用于 SQLite FTS5 ``MATCH`` 参数的查询表达式；无有效词元时返回空字符串 ``""``。
    """
    terms = extract_fts_query_terms(query)
    if not terms:
        return ""
    quoted = [f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms]
    return " OR ".join(quoted)


def build_postgres_tsquery(query: str) -> str:
    """构造适用于 PostgreSQL ``to_tsquery('simple', ...)`` 的安全检索表达式。

    将下划线连接的复合标识符转换使用位置邻近操作符 ``<->`` 进行精确短语匹配，
    其余词元通过 ``|`` 联合召回。

    Args:
        query: 用户检索查询文本。

    Returns:
        PostgreSQL ``to_tsquery`` 表达式字符串；无有效词元时返回空字符串 ``""``。
    """
    terms = extract_fts_query_terms(query)
    if not terms:
        return ""

    clauses: list[str] = []
    for term in terms:
        clean = _PG_SAFE_TERM_RE.sub("", term)
        if not clean:
            continue
        if "_" in clean:
            parts = [part for part in clean.split("_") if part]
            if len(parts) > 1:
                clauses.append(f"({' <-> '.join(parts)})")
            elif len(parts) == 1:
                clauses.append(parts[0])
        else:
            clauses.append(clean)

    unique_clauses = list(dict.fromkeys(clauses))
    return " | ".join(unique_clauses)


def build_mysql_ft_query(query: str) -> str:
    """构造适用于 MySQL ``MATCH(...) AGAINST (... IN NATURAL LANGUAGE MODE)`` 的安全检索串。

    由于 MySQL ``FULLTEXT`` 直接基于 ``chunks.content`` 原始文本建立索引（如保留 ``CVE-2026-0001``
    与 ``ERR_GATEWAY_TIMEOUT_504`` 原始字符格式），本方法保留原始连字符 ``-`` 与下划线 ``_``，
    仅将路径/包分隔符 ``. : /`` 规整为空格并剥离危险标点，确保与 MySQL ngram/标准分词器一致。

    Args:
        query: 用户检索查询文本。

    Returns:
        清理后的 MySQL 全文检索查询字符串；无有效内容时返回空字符串 ``""``。
    """
    if not query or not query.strip():
        return ""

    segments: list[str] = []
    for kind, value in _iter_segments(query):
        if kind == "compound":
            normalized = _MYSQL_SAFE_SEP_RE.sub(" ", value).lower().strip()
            if normalized:
                segments.extend(normalized.split())
        elif kind == "ascii":
            segments.append(value.lower())
        else:
            segments.append(value)

    deduplicated = list(dict.fromkeys(seg for seg in segments if seg))
    return " ".join(deduplicated[:_MAX_QUERY_TERMS])
