"""
递归字符与语法边界切分器。

按照段落、换行、标点与空格分层级递归切分长文本，并支持滑动重叠窗口。

@author Ateng
@since 2026-11-11
"""

import re

DEFAULT_SEPARATORS: list[str] = [
    "\n\n",
    "\n",
    "。\n",
    "！\n",
    "？\n",
    "。",
    "！",
    "？",
    ".\n",
    ". ",
    "! ",
    "? ",
    "；",
    ";",
    " ",
    "",
]

# 匹配 Markdown 图片语法与 HTML <img> 标签，切分时作为原子单元整体保留并暴露命名分组供解析复用
ATOMIC_IMAGE_PATTERN = re.compile(
    r"!\[(?P<alt>[^\]\r\n]*)\]\(\s*"
    r"(?:<(?P<bracket_url>[^>\r\n]+)>|(?P<url>(?:[^()\s]+|\([^()\r\n]*\))+))"
    r"(?:\s+(?:\"(?P<dq_title>[^\"\r\n]*)\"|'(?P<sq_title>[^'\r\n]*)'))?"
    r"\s*\)"
    r"|<img\b[^>]*>",
    re.IGNORECASE,
)


def _split_atomic_segments(text: str) -> list[tuple[str, bool]]:
    """按原子图片标签边界拆分文本为 `(片段, 是否原子)` 序列。"""
    segments: list[tuple[str, bool]] = []
    last_end = 0
    for match in ATOMIC_IMAGE_PATTERN.finditer(text):
        start, end = match.span()
        if start > last_end:
            segments.append((text[last_end:start], False))
        segments.append((match.group(0), True))
        last_end = end
    if last_end < len(text):
        segments.append((text[last_end:], False))
    return segments


def recursive_split_text(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
    separators: list[str] | None = None,
) -> list[str]:
    """对文本执行递归字符分块。

    Args:
        text: 待切分的完整正文内容
        chunk_size: 目标切片字符数上限
        chunk_overlap: 切片间的滑动重叠字符数
        separators: 优先级递减的分隔符列表

    Returns:
        list[str]: 切分后的切片文本列表
    """
    if not text or not text.strip():
        return []

    effective_chunk_size = max(1, chunk_size)
    if len(text) <= effective_chunk_size:
        return [text.strip()]

    seps = separators or DEFAULT_SEPARATORS
    step = max(1, effective_chunk_size - chunk_overlap)

    # 1. 递归拆分非原子文本单元
    def _split(s: str, sep_idx: int) -> list[str]:
        if len(s) <= effective_chunk_size:
            return [s] if s else []

        if sep_idx >= len(seps):
            # 所有分隔符用尽，强制按字符切片
            return [s[i : i + effective_chunk_size] for i in range(0, len(s), step)]

        sep = seps[sep_idx]
        if sep == "":
            return list(s)

        if sep not in s:
            return _split(s, sep_idx + 1)

        splits = s.split(sep)
        result: list[str] = []
        for i, piece in enumerate(splits):
            piece_with_sep = piece if i == len(splits) - 1 else piece + sep
            if len(piece_with_sep) > effective_chunk_size:
                result.extend(_split(piece_with_sep, sep_idx + 1))
            elif piece_with_sep:
                result.append(piece_with_sep)
        return result

    raw_pieces: list[str] = []
    for seg_text, is_atomic in _split_atomic_segments(text):
        if is_atomic:
            raw_pieces.append(seg_text)
        else:
            raw_pieces.extend(_split(seg_text, 0))

    # 2. 将碎片按 chunk_size 与 chunk_overlap 进行聚合
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for piece in raw_pieces:
        piece_len = len(piece)
        if current_len + piece_len > effective_chunk_size and current_chunk:
            combined = "".join(current_chunk).strip()
            if combined:
                chunks.append(combined)

            # 保留尾部重叠内容
            overlap_pieces: list[str] = []
            overlap_len = 0
            for p in reversed(current_chunk):
                if overlap_len + len(p) <= chunk_overlap:
                    overlap_pieces.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            current_chunk = overlap_pieces
            current_len = overlap_len

        current_chunk.append(piece)
        current_len += piece_len

    if current_chunk:
        combined = "".join(current_chunk).strip()
        if combined:
            chunks.append(combined)

    return chunks
