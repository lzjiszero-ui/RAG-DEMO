import import_jobs


def test_execute_job_reports_progress_and_completes() -> None:
    def runner(report):
        report("embedding", 75, "正在生成向量")
        return {"items": [{"chunks": 2}]}

    job = import_jobs.ImportJob(id="success", kind="files", runner=runner)
    import_jobs.execute_job(job)

    assert job.status == "completed"
    assert job.progress == 100
    assert job.attempt == 1
    assert job.result["items"][0]["chunks"] == 2


def test_execute_job_automatically_retries_transient_failure(monkeypatch) -> None:
    attempts = []

    def runner(report):
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("Qdrant temporarily unavailable")
        return {"item": {"chunks": 1}}

    monkeypatch.setattr(import_jobs, "sleep", lambda seconds: None)
    job = import_jobs.ImportJob(id="retry", kind="url", runner=runner, max_attempts=3)
    import_jobs.execute_job(job)

    assert len(attempts) == 2
    assert job.attempt == 2
    assert job.status == "completed"


def test_execute_job_does_not_retry_invalid_document(monkeypatch) -> None:
    attempts = []

    def runner(report):
        attempts.append(1)
        raise ValueError("文件内容为空")

    monkeypatch.setattr(import_jobs, "sleep", lambda seconds: None)
    job = import_jobs.ImportJob(id="invalid", kind="files", runner=runner, max_attempts=3)
    import_jobs.execute_job(job)

    assert len(attempts) == 1
    assert job.status == "failed"
    assert job.error == "文件内容为空"
