"""内存型异步知识导入任务队列，提供实时进度、自动重试和手动重试。"""

# 导入单线程执行器，让耗时导入离开 FastAPI 请求线程并避免多个任务争抢本地模型。
from concurrent.futures import ThreadPoolExecutor
# 导入 dataclass，用结构化字段保存任务状态。
from dataclasses import dataclass, field
# 导入 UTC 时间，用于页面展示任务创建和更新时间。
from datetime import datetime, timezone
# 导入日志模块，记录任务开始、重试、完成与失败。
import logging
# 导入线程锁，保护多个 API 请求与后台线程共享的任务字典。
from threading import RLock
# 导入重试等待函数。
from time import sleep
# 导入函数类型标注。
from collections.abc import Callable
# 导入 UUID，为每个任务生成不可预测的唯一 ID。
from uuid import uuid4

# 导入自动重试配置。
from config import IMPORT_MAX_ATTEMPTS, IMPORT_RETRY_DELAY_SECONDS
# 导入实际文件和网页处理函数。
from knowledge_service import import_bytes, import_url

# 创建当前模块日志记录器。
logger = logging.getLogger(__name__)
# 单线程串行处理任务，保护本地 Ollama 与 Qdrant 不被并发导入压垮。
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="knowledge-import")
# 任务状态和执行输入只保存在当前 FastAPI 进程内存中。
JOBS: dict[str, "ImportJob"] = {}
# 所有状态读写通过同一把可重入锁完成。
JOBS_LOCK = RLock()
# 最多保留最近任务，避免上传文件字节长期无限占用内存。
MAX_RETAINED_JOBS = 50


def utc_now() -> str:
    """返回适合 JSON 的 UTC ISO 时间。"""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ImportJob:
    """表示一次文件或网页知识导入任务。"""

    # 前端轮询使用的唯一任务 ID。
    id: str
    # files 或 url，用于页面说明任务来源类型。
    kind: str
    # 真正执行导入的闭包；文件任务会在闭包中保留上传字节以支持失败重试。
    runner: Callable[[Callable[[str, int, str], None]], dict]
    # queued、running、retrying、completed、failed。
    status: str = "queued"
    # 当前业务阶段，例如 parsing、contextualizing、embedding。
    step: str = "queued"
    # 0 到 100 的整体完成百分比。
    progress: int = 0
    # 页面显示的当前状态消息。
    message: str = "任务已进入队列"
    # 已经开始执行的次数。
    attempt: int = 0
    # 包含首次在内的最大尝试次数。
    max_attempts: int = IMPORT_MAX_ATTEMPTS
    # 成功后的文件与切片摘要。
    result: dict | None = None
    # 最终失败原因；重试开始时会清空。
    error: str | None = None
    # 创建和更新时间。
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def public_dict(self) -> dict:
        """只返回 JSON 安全状态，不暴露 runner 和上传文件内容。"""
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "step": self.step,
            "progress": self.progress,
            "message": self.message,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def update_job(job: ImportJob, **changes) -> None:
    """线程安全地更新任务字段和时间。"""
    with JOBS_LOCK:
        for key, value in changes.items():
            setattr(job, key, value)
        job.updated_at = utc_now()


def execute_job(job: ImportJob) -> None:
    """执行任务，并对非数据校验异常进行自动重试。"""
    while job.attempt < job.max_attempts:
        # 每次尝试开始时清除旧错误并显示实际次数。
        update_job(
            job,
            status="running",
            step="starting",
            progress=1,
            message=f"开始第 {job.attempt + 1}/{job.max_attempts} 次尝试",
            attempt=job.attempt + 1,
            error=None,
        )

        # runner 调用此函数即可实时更新阶段和百分比。
        def report(step: str, progress: int, message: str) -> None:
            update_job(job, step=step, progress=max(0, min(99, int(progress))), message=message)

        try:
            result = job.runner(report)
            # 成功时保存摘要并完成任务。
            update_job(job, status="completed", step="completed", progress=100, message="知识导入完成", result=result)
            logger.info("import job completed | job_id=%s | attempts=%d", job.id, job.attempt)
            return
        except ValueError as exc:
            # 格式错误、空文档等输入问题重复执行不会恢复，因此立即结束。
            update_job(job, status="failed", step="failed", message="导入内容校验失败", error=str(exc))
            logger.warning("import job validation failed | job_id=%s | error=%s", job.id, exc)
            return
        except Exception as exc:
            logger.exception("import job attempt failed | job_id=%s | attempt=%d", job.id, job.attempt)
            # 还有机会时进入 retrying，页面可以看到倒计时原因和尝试次数。
            if job.attempt < job.max_attempts:
                update_job(
                    job,
                    status="retrying",
                    step="retrying",
                    message=f"本次失败，{IMPORT_RETRY_DELAY_SECONDS:g} 秒后自动重试",
                    error=str(exc),
                )
                sleep(IMPORT_RETRY_DELAY_SECONDS)
                continue
            # 达到最大次数后等待用户点击手动重试。
            update_job(job, status="failed", step="failed", message="自动重试次数已用完", error=str(exc))
            return


def register_job(kind: str, runner: Callable[[Callable[[str, int, str], None]], dict]) -> dict:
    """保存并异步提交一个新任务。"""
    job = ImportJob(id=uuid4().hex, kind=kind, runner=runner)
    with JOBS_LOCK:
        # 超过保留上限时优先移除最早且已经结束的任务。
        finished_ids = [job_id for job_id, item in JOBS.items() if item.status in {"completed", "failed"}]
        while len(JOBS) >= MAX_RETAINED_JOBS and finished_ids:
            JOBS.pop(finished_ids.pop(0), None)
        JOBS[job.id] = job
    EXECUTOR.submit(execute_job, job)
    logger.info("import job queued | job_id=%s | kind=%s", job.id, kind)
    return job.public_dict()


def submit_file_job(files: list[tuple[str, bytes]], category: str) -> dict:
    """创建一个可处理多个上传文件的异步任务。"""
    def runner(report) -> dict:
        results = []
        total = len(files)
        for index, (filename, content) in enumerate(files):
            # 把单文件 0-100 进度映射到整个多文件任务的 0-98 区间。
            def file_report(step: str, progress: int, message: str) -> None:
                overall = int(((index + progress / 100) / total) * 98)
                report(step, overall, f"[{index + 1}/{total}] {message}")

            results.append(import_bytes(content, filename, category, progress_callback=file_report))
        return {"file_count": len(results), "chunk_count": sum(item["chunks"] for item in results), "items": results}

    return register_job("files", runner)


def submit_url_job(url: str, category: str) -> dict:
    """创建一个网页抓取异步任务。"""
    def runner(report) -> dict:
        return {"item": import_url(url, category, progress_callback=report)}

    return register_job("url", runner)


def get_job(job_id: str) -> dict:
    """读取任务公开状态。"""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job.public_dict()


def retry_job(job_id: str) -> dict:
    """把最终失败的任务重新加入队列，并重新获得完整自动重试次数。"""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status != "failed":
            raise ValueError("只有失败任务可以重新尝试")
        job.status = "queued"
        job.step = "queued"
        job.progress = 0
        job.message = "任务已重新进入队列"
        job.attempt = 0
        job.error = None
        job.updated_at = utc_now()
    EXECUTOR.submit(execute_job, job)
    return job.public_dict()
