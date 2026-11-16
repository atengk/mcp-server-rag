"""
后台异步文档摄取任务管理器。

为大文档或大规模目录提供有界并发、同目标串行防竞态与有界历史容量控制的后台任务状态机，
配合 ``rag_get_task_status`` 工具避免长耗时摄取触发 MCP 客户端同步超时。

@author Ateng
@since 2026-11-16
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Literal

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, float, str], None]
TaskStatus = Literal["pending", "running", "completed", "failed"]


def _utc_now_iso() -> str:
    """返回标准 UTC ISO-8601 时间戳字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _is_failed_task_result(res: dict[str, Any]) -> bool:
    """判定后台摄取返回值是否应视为任务失败（含目录内全部文件均失败场景）。"""
    if res.get("success") is False:
        return True
    total_indexed = int(res.get("total_indexed") or 0)
    failed_count = int(res.get("failed_count") or 0)
    return total_indexed > 0 and failed_count == total_indexed


class IngestionTaskManager:
    """线程安全的后台摄取任务注册与状态查询管理器。"""

    def __init__(self, *, max_workers: int = 2, max_tasks: int = 200) -> None:
        """初始化后台摄取任务管理器。

        Args:
            max_workers: 最大并发后台摄取工作线程数，必须 >= 1。
            max_tasks: 内存中保留的历史任务记录上限，必须 >= 1（仅淘汰已终态任务）。

        Raises:
            ValueError: ``max_workers < 1`` 或 ``max_tasks < 1`` 时抛出。
        """
        if max_workers < 1:
            raise ValueError(f"max_workers 必须 >= 1，当前值: {max_workers}")
        if max_tasks < 1:
            raise ValueError(f"max_tasks 必须 >= 1，当前值: {max_tasks}")
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_workers)
        self._max_tasks = max_tasks
        self._tasks: dict[str, dict[str, Any]] = {}
        self._key_locks: dict[str, threading.Lock] = {}

    def _evict_if_needed(self) -> None:
        """当历史记录达到上限时仅淘汰最早结束的终态任务，绝不删除活跃中任务。"""
        while len(self._tasks) >= self._max_tasks:
            finished_key = next(
                (
                    tid
                    for tid, info in self._tasks.items()
                    if info.get("status") in {"completed", "failed"}
                ),
                None,
            )
            if finished_key is None:
                break
            self._tasks.pop(finished_key, None)

    def _get_target_lock(self, lock_key: str) -> threading.Lock | None:
        """按目标键获取串行互斥锁，防止并发后台任务对同一文档发生读写竞态。"""
        if not lock_key:
            return None
        with self._lock:
            lock = self._key_locks.get(lock_key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[lock_key] = lock
            return lock

    def _update_task(self, task_id: str, **updates: Any) -> None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is not None:
                record.update(updates)
                record["updated_at"] = _utc_now_iso()

    def submit(
        self,
        *,
        task_type: str,
        target: str,
        runner: Callable[[ProgressCallback], dict[str, Any]],
        lock_key: str = "",
    ) -> dict[str, Any]:
        """提交后台摄取任务并立即返回携带 ``task_id`` 的受理响应。

        Args:
            task_type: 任务类型标识（如 ``"index_file"`` / ``"index_directory"``）。
            target: 目标文件或目录路径描述。
            runner: 接收进度回调 ``(progress, total, message) -> None`` 并返回结果字典的同步函数。
            lock_key: 可选并发串行锁键（如 ``"{kb_name}:{resolved_path}"``），防止同文档写覆盖竞态。

        Returns:
            包含 ``success``、``status="accepted"``、``task_id`` 与 ``task_status`` 的响应字典。
        """
        task_id = uuid.uuid4().hex[:12]
        now = _utc_now_iso()
        pending_status: TaskStatus = "pending"
        initial_record: dict[str, Any] = {
            "task_id": task_id,
            "task_type": task_type,
            "target": target,
            "status": pending_status,
            "progress": 0.0,
            "total": 1.0,
            "message": "任务已提交，等待执行",
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._evict_if_needed()
            self._tasks[task_id] = initial_record

        def _progress_cb(progress: float, total: float, message: str) -> None:
            self._update_task(
                task_id,
                status="running",
                progress=float(progress),
                total=float(total),
                message=message,
            )

        def _execute_locked() -> dict[str, Any]:
            target_lock = self._get_target_lock(lock_key)
            if target_lock is None:
                return runner(_progress_cb)
            with target_lock:
                return runner(_progress_cb)

        def _worker() -> None:
            with self._semaphore:
                self._update_task(task_id, status="running", message="后台摄取任务正在执行")
                try:
                    res = _execute_locked()
                except Exception as exc:
                    logger.exception("后台摄取任务执行失败 (task_id=%s): %s", task_id, exc)
                    self._update_task(
                        task_id,
                        status="failed",
                        error=str(exc),
                        message=f"后台摄取任务失败: {exc}",
                    )
                    return

                if _is_failed_task_result(res):
                    err_msg = str(res.get("message") or "后台摄取任务失败")
                    self._update_task(
                        task_id,
                        status="failed",
                        error=err_msg,
                        result=res,
                        message=err_msg,
                    )
                else:
                    with self._lock:
                        current_total = float(self._tasks.get(task_id, {}).get("total", 1.0))
                    self._update_task(
                        task_id,
                        status="completed",
                        progress=current_total,
                        result=res,
                        message=str(res.get("message") or "后台摄取任务已完成"),
                    )

        worker = threading.Thread(
            target=_worker,
            name=f"rag-ingest-{task_id}",
            daemon=True,
        )
        worker.start()

        return {
            "success": True,
            "status": "accepted",
            "task_id": task_id,
            "task_status": "pending",
            "message": f"已启动后台摄取任务 '{task_id}'，请通过 rag_get_task_status 查询进度与结果",
        }

    def get_status(self, task_id: str) -> dict[str, Any]:
        """查询指定后台摄取任务的最新状态、进度与执行结果。

        Args:
            task_id: 后台摄取任务唯一标识；为空或未找到时返回 ``success=False``。

        Returns:
            包含 ``success`` 与任务状态字段（``status``、``progress``、``total``、``result``、
            ``error``）的字典；任务不存在或 ``task_id`` 为空时返回包含错误说明的字典。
        """
        clean_id = task_id.strip()
        if not clean_id:
            return {"success": False, "message": "task_id 不能为空"}
        with self._lock:
            record = self._tasks.get(clean_id)
            if record is None:
                return {
                    "success": False,
                    "task_id": clean_id,
                    "message": f"未找到后台摄取任务 '{clean_id}'",
                }
            return {"success": True, **dict(record)}


_DEFAULT_TASK_MANAGER = IngestionTaskManager()


def get_ingestion_task_manager() -> IngestionTaskManager:
    """获取进程级共享后台摄取任务管理器实例。"""
    return _DEFAULT_TASK_MANAGER
