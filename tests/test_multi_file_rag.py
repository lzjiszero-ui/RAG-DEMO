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

    documents = load_documents(tmp_path, contextual_retrieval=False)

    assert [document.metadata for document in documents] == [
        {"source": "first.txt", "category": "通用", "point_name": "first-1", "chunk_index": 0, "contextual_summary": "", "original_text": "first knowledge", "contextualized": False},
        {"source": "nested/second.txt", "category": "nested", "point_name": "second-1", "chunk_index": 0, "contextual_summary": "", "original_text": "second knowledge", "contextualized": False},
    ]


def test_load_documents_adds_contextual_retrieval_text(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "book.txt").write_text("武松在快活林醉打蒋门神。", encoding="utf-8")
    monkeypatch.setattr("ingest.contextualize_chunk", lambda source, document, chunk: "《水浒传》中武松在快活林帮助施恩。")

    documents = load_documents(tmp_path, contextual_retrieval=True)

    assert documents[0].page_content.startswith("文档来源：book.txt\n切片上下文：《水浒传》")
    assert documents[0].metadata["original_text"] == "武松在快活林醉打蒋门神。"
    assert documents[0].metadata["contextualized"] is True


def test_ask_does_not_call_model_when_nothing_passes_threshold(monkeypatch) -> None:
    monkeypatch.setattr(rag, "rewrite_query", lambda question: "rewritten question")
    monkeypatch.setattr(rag, "retrieve_candidates", lambda question, category: [])

    answer, hits, rewritten_query = ask("unrelated question")

    assert answer == "知识库中没有找到足够相关的资料。"
    assert hits == []
    assert rewritten_query == "rewritten question"


def test_rewrite_query_returns_model_output(monkeypatch) -> None:
    class FakeChain:
        def __or__(self, other):
            return self

        def invoke(self, values):
            assert values == {"question": "它用什么模型？", "chat_history": "（无历史对话）"}
            return "“项目使用 bge-m3 Embedding 模型生成向量”"

    fake_chain = FakeChain()
    monkeypatch.setattr(rag.ChatPromptTemplate, "from_messages", lambda messages: fake_chain)
    monkeypatch.setattr(rag, "ChatOllama", lambda **kwargs: fake_chain)
    monkeypatch.setattr(rag, "StrOutputParser", lambda: fake_chain)

    assert rag.rewrite_query("它用什么模型？") == "项目使用 bge-m3 Embedding 模型生成向量"


def test_rewrite_query_falls_back_when_model_fails(monkeypatch) -> None:
    class FailingChain:
        def __or__(self, other):
            return self

        def invoke(self, values):
            raise RuntimeError("Ollama unavailable")

    failing_chain = FailingChain()
    monkeypatch.setattr(rag.ChatPromptTemplate, "from_messages", lambda messages: failing_chain)
    monkeypatch.setattr(rag, "ChatOllama", lambda **kwargs: failing_chain)
    monkeypatch.setattr(rag, "StrOutputParser", lambda: failing_chain)

    assert rag.rewrite_query("原始问题") == "原始问题"


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


def test_retrieve_candidates_applies_category_metadata_filter(monkeypatch) -> None:
    captured = {}

    class FakeVectorStore:
        def similarity_search_with_score(self, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr(rag, "vector_store", lambda: FakeVectorStore())

    rag.retrieve_candidates("宋江是谁", "水浒传")

    category_filter = captured["filter"]
    assert category_filter.must[0].key == "metadata.category"
    assert category_filter.must[0].match.value == "水浒传"
    assert captured["score_threshold"] is None


def test_dense_retrieve_candidates_keeps_global_threshold(monkeypatch) -> None:
    captured = {}

    class FakeVectorStore:
        def similarity_search_with_score(self, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr(rag, "dense_vector_store", lambda: FakeVectorStore())

    rag.retrieve_dense_candidates("向量模型是什么", "全部")

    assert captured["filter"] is None
    assert captured["score_threshold"] == rag.SCORE_THRESHOLD


def test_retrieve_mode_candidates_supports_all_four_modes(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(rag, "retrieve_dense_candidates", lambda question, category: calls.append("vector") or [])
    monkeypatch.setattr(rag, "retrieve_sparse_candidates", lambda question, category: calls.append("bm25") or [])
    monkeypatch.setattr(rag, "retrieve_candidates", lambda question, category: calls.append("hybrid") or [])

    for mode in ("vector", "bm25", "hybrid", "hybrid_rerank"):
        monkeypatch.setattr(rag, "RETRIEVAL_MODE", mode)
        rag.retrieve_mode_candidates("问题", "全部")

    assert calls == ["vector", "bm25", "hybrid", "hybrid"]


def test_retrieve_skips_reranker_unless_mode_requires_it(monkeypatch) -> None:
    document = Document(page_content="context", metadata={"source": "book.txt", "chunk_index": 0})
    monkeypatch.setattr(rag, "retrieve_mode_candidates", lambda question, category: [(document, 0.8)])
    monkeypatch.setattr(rag, "rerank_candidates", lambda question, candidates: (_ for _ in ()).throw(AssertionError("reranker should be skipped")))
    monkeypatch.setattr(rag, "RETRIEVAL_MODE", "hybrid")

    hits = rag.retrieve("问题")

    assert len(hits) == 1
    assert hits[0].vector_score == 0.8
    assert hits[0].rerank_score == 0.0


def test_bm25_tokenize_adds_chinese_bigrams_and_keeps_technical_terms() -> None:
    tokens = rag.bm25_tokenize("谁醉打蒋门神？TOP_K").split()

    assert "蒋门" in tokens
    assert "门神" in tokens
    assert "top_k" in tokens
