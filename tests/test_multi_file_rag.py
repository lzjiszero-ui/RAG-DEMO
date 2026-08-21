from pathlib import Path

from langchain_core.documents import Document

import rag
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


def test_retrieve_sorts_candidates_by_reranker_score(monkeypatch) -> None:
    class FakeVectorStore:
        def similarity_search_with_score(self, **kwargs):
            return [
                (
                    Document(
                        page_content="vector search first",
                        metadata={"source": "first.txt", "chunk_index": 0},
                    ),
                    0.9,
                ),
                (
                    Document(
                        page_content="reranker prefers this",
                        metadata={"source": "second.txt", "chunk_index": 1},
                    ),
                    0.8,
                ),
            ]

    class FakeReranker:
        def rerank(self, question, documents):
            return [-2.0, 3.0]

    monkeypatch.setattr(rag, "vector_store", lambda: FakeVectorStore())
    monkeypatch.setattr(rag, "reranker", lambda: FakeReranker())

    hits = rag.retrieve("test question")

    assert [hit.source for hit in hits] == ["second.txt", "first.txt"]
    assert hits[0].vector_score == 0.8
    assert hits[0].rerank_score == 3.0
