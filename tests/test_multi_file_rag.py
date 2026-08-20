from pathlib import Path

from ingest import load_documents
from rag import ask


def test_load_documents_keeps_source_and_chunk_index(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("first knowledge", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "second.txt").write_text("second knowledge", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("not imported", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [document.metadata for document in documents] == [
        {"source": "first.txt", "chunk_index": 0},
        {"source": "nested/second.txt", "chunk_index": 0},
    ]


def test_ask_does_not_call_model_when_nothing_passes_threshold(monkeypatch) -> None:
    monkeypatch.setattr("rag.retrieve", lambda question: [])

    answer, hits = ask("unrelated question")

    assert answer == "知识库中没有找到足够相关的资料。"
    assert hits == []
