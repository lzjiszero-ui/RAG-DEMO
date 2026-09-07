import pytest

import knowledge_service


def test_safe_name_removes_path_separators() -> None:
    assert knowledge_service.safe_name("../危险/分类", "通用") == "_危险_分类"


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
