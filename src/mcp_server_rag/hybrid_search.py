"""
RRF 双路混合检索与融合重排核心编排引擎。

联合并发调用底层向量存储（Chroma / Milvus）的稠密语义检索与元数据目录（SQLite FTS5 /
PostgreSQL GIN tsvector / MySQL FULLTEXT）的原生倒排检索，基于 Reciprocal Rank Fusion
（``RRF_K = 60``）完成多通道排名融合，并执行单调校准与过滤边界收敛。

@author Ateng
@since 2026-11-11
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Final

from mcp_server_rag.metadata.base import MetadataCatalogAdapter
from mcp_server_rag.vector_store.base import VectorStoreAdapter

# RRF 排名融合平滑常数（行业通用经验值 60）
RRF_K: Final[int] = 60
_MIN_CANDIDATE_POOL: Final[int] = 20
_CANDIDATE_MULTIPLIER: Final[int] = 4
_FILTERED_MIN_CANDIDATE_POOL: Final[int] = 100
_FILTERED_CANDIDATE_MULTIPLIER: Final[int] = 8

# FTS 命中校准分值基准参数，保证精确词元召回稳定超越常规 threshold 且严格维持单调性上界 1.0
_FTS_RANK_BASE: Final[float] = 0.50
_FTS_RANK_SPAN: Final[float] = 0.35
_FTS_SCORE_CAP: Final[float] = 0.85
_MAX_CALIBRATED_SCORE: Final[float] = 1.0
_FTS_FUTURE_TIMEOUT_SECONDS: Final[float] = 30.0


@dataclass(frozen=True)
class HybridSearchOptions:
    """混合检索返回数量、阈值与过滤条件封装对象。"""

    top_k: int = 5
    threshold: float = 0.0
    keyword: str | None = None
    metadata_filter: dict[str, Any] | None = None
    where: dict[str, Any] | None = None
    where_document: dict[str, Any] | None = None


@dataclass
class _FusionCandidate:
    """内部双路候选融合状态记录。"""

    hit: dict[str, Any]
    vec_rank: int | None = None
    vec_score: float | None = None
    fts_rank: int | None = None
    fts_score: float | None = None
    rrf_score: float = 0.0
    score: float = 0.0


def _passes_post_filters(
    item: dict[str, Any],
    clean_kw: str | None,
    metadata_filter: dict[str, Any] | None,
) -> bool:
    """校验候选切片是否严格满足关键词包含条件与标量元数据过滤条件。"""
    if clean_kw and clean_kw not in str(item.get("text") or "").lower():
        return False
    if metadata_filter:
        item_meta = item.get("metadata") or {}
        for mk, mv in metadata_filter.items():
            if not mk.startswith("$") and item_meta.get(mk) != mv:
                return False
    return True


def _compute_fts_floor(fts_rank: int) -> float:
    """将 FTS 倒排排名映射至单调递减的校准相似度保底分值 [0.50, 0.85]。"""
    return round(min(_FTS_SCORE_CAP, _FTS_RANK_BASE + _FTS_RANK_SPAN / fts_rank), 4)


def _merge_metadata(target_meta: dict[str, Any], source_meta: dict[str, Any]) -> None:
    """在不覆盖向量库既有自定义字段的前提下，补充 FTS 关联返回的缺失基础元数据。"""
    for key, val in source_meta.items():
        if key not in target_meta or target_meta[key] in ("", None, []):
            target_meta[key] = val


def _fuse_ranked_channels(
    vector_candidates: list[dict[str, Any]],
    fts_candidates: list[dict[str, Any]],
    has_metadata_filter: bool,
) -> list[dict[str, Any]]:
    """基于标准 RRF 算法融合向量检索与 FTS 倒排检索双路候选序列。"""
    ordered_vector = sorted(
        vector_candidates,
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )

    entries: dict[tuple[str, str], _FusionCandidate] = {}

    for vec_rank, raw_vec in enumerate(ordered_vector, start=1):
        kb_name = str(raw_vec.get("kb_name") or "")
        chunk_id = str(raw_vec.get("chunk_id") or "")
        key = (kb_name, chunk_id)
        item = dict(raw_vec)
        item["metadata"] = dict(raw_vec.get("metadata") or {})
        item["kb_name"] = kb_name
        item["metadata"]["kb_name"] = kb_name
        score_val = float(raw_vec.get("score") or 0.0)
        entries[key] = _FusionCandidate(
            hit=item,
            vec_rank=vec_rank,
            vec_score=score_val,
        )

    for fts_rank, raw_fts in enumerate(fts_candidates, start=1):
        kb_name = str(raw_fts.get("kb_name") or "")
        chunk_id = str(raw_fts.get("chunk_id") or "")
        key = (kb_name, chunk_id)
        fts_score_val = float(raw_fts.get("fts_score") or 0.0)

        existing = entries.get(key)
        if existing is not None:
            existing.fts_rank = fts_rank
            existing.fts_score = fts_score_val
            _merge_metadata(existing.hit["metadata"], dict(raw_fts.get("metadata") or {}))
            if not existing.hit.get("images") and raw_fts.get("images"):
                existing.hit["images"] = list(raw_fts["images"])
        elif not has_metadata_filter:
            # 当存在 metadata_filter 时，关系表未冗余存储自定义向量元数据，
            # 仅允许通过向量过滤引擎验证的切片合入，杜绝 FTS 越权泄漏非匹配切片。
            item = {
                "chunk_id": chunk_id,
                "text": str(raw_fts.get("text") or ""),
                "score": 0.0,
                "metadata": dict(raw_fts.get("metadata") or {}),
                "kb_name": kb_name,
            }
            if raw_fts.get("images"):
                item["images"] = list(raw_fts["images"])
            entries[key] = _FusionCandidate(
                hit=item,
                fts_rank=fts_rank,
                fts_score=fts_score_val,
            )

    if not entries:
        return []

    fused = list(entries.values())
    for entry in fused:
        vec_rrf = 1.0 / (RRF_K + entry.vec_rank) if entry.vec_rank is not None else 0.0
        fts_rrf = 1.0 / (RRF_K + entry.fts_rank) if entry.fts_rank is not None else 0.0
        entry.rrf_score = vec_rrf + fts_rrf
        if entry.fts_rank is not None:
            fts_floor = _compute_fts_floor(entry.fts_rank)
            entry.score = (
                max(entry.vec_score, fts_floor) if entry.vec_score is not None else fts_floor
            )
        else:
            entry.score = float(entry.vec_score) if entry.vec_score is not None else 0.0

    fused.sort(
        key=lambda entry: (
            entry.rrf_score,
            entry.score,
            -(entry.fts_rank if entry.fts_rank is not None else 10**9),
            -(entry.vec_rank if entry.vec_rank is not None else 10**9),
        ),
        reverse=True,
    )

    # 保证外显综合分值与 RRF 全局排序严格保持同向非递增单调性
    for idx in range(len(fused) - 2, -1, -1):
        if fused[idx].score < fused[idx + 1].score:
            fused[idx].score = min(_MAX_CALIBRATED_SCORE, fused[idx + 1].score)

    results: list[dict[str, Any]] = []
    for entry in fused:
        hit = entry.hit
        hit["score"] = entry.score
        channels: list[str] = []
        if entry.vec_rank is not None:
            channels.append("vector")
        if entry.fts_rank is not None:
            channels.append("fts")
        meta = hit["metadata"]
        meta["rrf_score"] = round(entry.rrf_score, 6)
        meta["retrieval_channels"] = channels
        results.append(hit)

    return results


def execute_hybrid_search(
    catalog: MetadataCatalogAdapter,
    vector_store: VectorStoreAdapter,
    target_kbs: list[str],
    query: str,
    query_vector: list[float],
    options: HybridSearchOptions | None = None,
) -> list[dict[str, Any]]:
    """执行跨库向量检索 + 原生数据库 FTS 并发混合召回、RRF 融合与过滤截断。

    Args:
        catalog: 遵循 ``MetadataCatalogAdapter`` 契约的元数据目录实例。
        vector_store: 向量存储适配器实例。
        target_kbs: 已完成存在性与权限解析的目标知识库名称列表。
        query: 原始查询文本。
        query_vector: 由嵌入模型生成的查询稠密向量。
        options: 可选混合检索参数与过滤条件对象；缺省时采用标准默认配置。

    Returns:
        按 RRF 与校准分值降序排列、经过阈值截断后的前 ``top_k`` 条命中切片。
    """
    cfg = options or HybridSearchOptions()
    clean_query = query.strip()
    if not clean_query or not target_kbs or cfg.top_k <= 0:
        return []

    clean_kw = cfg.keyword.strip().lower() if cfg.keyword and cfg.keyword.strip() else None
    has_meta_filter = bool(cfg.metadata_filter)
    candidate_top_k = (
        max(cfg.top_k * _FILTERED_CANDIDATE_MULTIPLIER, _FILTERED_MIN_CANDIDATE_POOL)
        if has_meta_filter
        else max(cfg.top_k * _CANDIDATE_MULTIPLIER, _MIN_CANDIDATE_POOL)
    )

    def _collect_vector_candidates() -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for target_kb in target_kbs:
            raw_hits = vector_store.search(
                kb_name=target_kb,
                query_embedding=query_vector,
                top_k=candidate_top_k,
                where=cfg.where,
                where_document=cfg.where_document,
            )
            for item in raw_hits:
                item_copy = dict(item)
                item_meta = dict(item.get("metadata") or {})
                item_copy["kb_name"] = target_kb
                item_meta["kb_name"] = target_kb
                item_copy["metadata"] = item_meta
                if _passes_post_filters(item_copy, clean_kw, cfg.metadata_filter):
                    collected.append(item_copy)
        return collected

    # 并发调度数据库原生 FTS 检索与向量库稠密检索，并配置显式超时防止外部连接无限阻塞
    with ThreadPoolExecutor(max_workers=1) as executor:
        fts_future = executor.submit(
            catalog.search_chunks_fts,
            target_kbs,
            clean_query,
            candidate_top_k,
        )
        vector_candidates = _collect_vector_candidates()
        raw_fts = fts_future.result(timeout=_FTS_FUTURE_TIMEOUT_SECONDS)

    fts_candidates = [item for item in raw_fts if _passes_post_filters(item, clean_kw, None)]

    fused = _fuse_ranked_channels(
        vector_candidates=vector_candidates,
        fts_candidates=fts_candidates,
        has_metadata_filter=has_meta_filter,
    )
    filtered = [hit for hit in fused if float(hit.get("score") or 0.0) >= cfg.threshold]
    return filtered[: cfg.top_k]
