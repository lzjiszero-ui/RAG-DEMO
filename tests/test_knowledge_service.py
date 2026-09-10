import pytest

import knowledge_service


def test_safe_name_removes_path_separators() -> None:
    assert knowledge_service.safe_name("../危险/分类", "通用") == "_危险_分类"


def test_empty_category_uses_general_folder(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(knowledge_service, "KNOWLEDGE_DIR", tmp_path)

    destination, source, category = knowledge_service.source_location("", "guide.txt")

    assert category == "通用"
    assert source == "通用/guide.txt"
    assert destination.parent == tmp_path / "通用"
    assert destination.parent.is_dir()


def test_validate_public_url_rejects_localhost(monkeypatch) -> None:
    monkeypatch.setattr(knowledge_service.socket, "getaddrinfo", lambda *args, **kwargs: [(None, None, None, None, ("127.0.0.1", 80))])

    with pytest.raises(ValueError, match="内网"):
        knowledge_service.validate_public_url("http://localhost/page")


def test_import_url_keeps_original_url_in_metadata(monkeypatch, tmp_path) -> None:
    captured = {}
    def fake_upsert(documents):
        captured["documents"] = documents
        return len(documents)

    monkeypatch.setattr(knowledge_service, "KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(knowledge_service, "fetch_web_page", lambda url: (b"<h1>Public guide</h1>", url))
    monkeypatch.setattr("ingest.contextualize_chunk", lambda source, document, chunk: "page context")
    monkeypatch.setattr(knowledge_service, "upsert_source_documents", fake_upsert)

    result = knowledge_service.import_url("https://example.com/guide", "网页")

    assert result["url"] == "https://example.com/guide"
    assert captured["documents"][0].metadata["url"] == "https://example.com/guide"


def test_web_import_limits_characters_and_chunks(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_upsert(documents):
        captured["documents"] = documents
        return len(documents)

    large_html = ("<article>" + "".join(f"<p>{index}-" + "x" * 100 + "</p>" for index in range(30)) + "</article>").encode()
    monkeypatch.setattr(knowledge_service, "KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(knowledge_service, "MAX_WEB_DOCUMENT_CHARS", 1200)
    monkeypatch.setattr(knowledge_service, "MAX_WEB_CHUNKS", 1)
    monkeypatch.setattr(knowledge_service, "fetch_web_page", lambda url: (large_html, url))
    monkeypatch.setattr("ingest.contextualize_chunk", lambda source, document, chunk: "context")
    monkeypatch.setattr(knowledge_service, "upsert_source_documents", fake_upsert)

    result = knowledge_service.import_url("https://example.com/large", "网页")

    assert result["characters"] == 1200
    assert result["chunks"] == 1
    assert result["truncated"] is True
    assert len(captured["documents"]) == 1


def test_delete_category_rejects_protected_categories() -> None:
    with pytest.raises(ValueError, match="不能删除"):
        knowledge_service.delete_category("通用")


def test_delete_category_removes_only_selected_directory_and_points(monkeypatch, tmp_path) -> None:
    target = tmp_path / "待删除"
    target.mkdir()
    (target / "doc.txt").write_text("content", encoding="utf-8")
    kept = tmp_path / "保留"
    kept.mkdir()
    deleted = {}

    class FakeCount:
        count = 4

    class FakeClient:
        def __init__(self, url):
            pass

        def collection_exists(self, name):
            return True

        def count(self, **kwargs):
            return FakeCount()

        def delete(self, **kwargs):
            deleted.update(kwargs)

    monkeypatch.setattr(knowledge_service, "KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(knowledge_service, "QdrantClient", FakeClient)

    result = knowledge_service.delete_category("待删除")

    assert result["deleted_points"] == 4
    assert not target.exists()
    assert kept.exists()
    assert deleted["collection_name"] == knowledge_service.COLLECTION_NAME
