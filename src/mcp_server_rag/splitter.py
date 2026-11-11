"""
递归字符与语法边界切分器。

按照段落、换行、标点与空格分层级递归切分长文本，并支持滑动重叠窗口。

@author Ateng
@since 2026-11-11
"""

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

    if len(text) <= chunk_size:
        return [text.strip()]

    seps = separators or DEFAULT_SEPARATORS

    # 1. 递归拆分单元
    def _split(s: str, sep_idx: int) -> list[str]:
        if len(s) <= chunk_size:
            return [s] if s else []

        if sep_idx >= len(seps):
            # 所有分隔符用尽，强制按字符切片
            return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size - chunk_overlap)]

        sep = seps[sep_idx]
        if sep == "":
            return list(s)

        if sep not in s:
            return _split(s, sep_idx + 1)

        splits = s.split(sep)
        result: list[str] = []
        for i, piece in enumerate(splits):
            piece_with_sep = piece if i == len(splits) - 1 else piece + sep
            if len(piece_with_sep) > chunk_size:
                result.extend(_split(piece_with_sep, sep_idx + 1))
            elif piece_with_sep:
                result.append(piece_with_sep)
        return result

    raw_pieces = _split(text, 0)

    # 2. 将碎片按 chunk_size 与 chunk_overlap 进行聚合
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for piece in raw_pieces:
        piece_len = len(piece)
        if current_len + piece_len > chunk_size and current_chunk:
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
