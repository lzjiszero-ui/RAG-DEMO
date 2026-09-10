from pathlib import Path

from langchain_core.documents import Document

import rag
from ingest import extract_sections, extract_text, load_documents, table_to_markdown


def test_load_documents_keeps_source_and_chunk_index(tmp_path: Path) -> None:
    (tmp_path / "first.txt").write_text("first knowledge", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "second.txt").write_text("second knowledge", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("not imported", encoding="utf-8")

    documents = load_documents(tmp_path, contextual_retrieval=False)

    assert [document.metadata for document in documents] == [
        {"source": "first.txt", "category": "通用", "point_name": "first-1", "chunk_index": 0, "parent_index": 0, "parent_text": "first knowledge", "contextual_summary": "", "original_text": "first knowledge", "contextualized": False, "page": None, "section": "正文", "content_type": "text", "ocr_used": False},
        {"source": "nested/second.txt", "category": "nested", "point_name": "second-1", "chunk_index": 0, "parent_index": 0, "parent_text": "second knowledge", "contextual_summary": "", "original_text": "second knowledge", "contextualized": False, "page": None, "section": "正文", "content_type": "text", "ocr_used": False},
    ]


def test_extract_html_keeps_visible_text_and_removes_script() -> None:
    content = b"<html><head><style>.x{}</style></head><body><h1>Guide</h1><p>Useful text</p><script>alert(1)</script></body></html>"

    text = extract_text(content, ".html")

    assert "Guide" in text
    assert "Useful text" in text
    assert "alert" not in text


def test_extract_html_prefers_article_and_removes_template_text() -> None:
    content = """
    <html><body>
      <textarea>very large template data</textarea>
      <nav>navigation</nav>
      <article><h1>Article title</h1><p>Useful paragraph</p><p>Useful paragraph</p></article>
      <footer>footer text</footer>
    </body></html>
    """.encode()

    text = extract_text(content, ".html")

    assert text == "Article title\nUseful paragraph"
    assert "template" not in text
    assert "navigation" not in text


def test_load_documents_supports_html(tmp_path: Path) -> None:
    (tmp_path / "guide.html").write_text("<h1>HTML knowledge</h1><p>Imported body</p>", encoding="utf-8")

    documents = load_documents(tmp_path, contextual_retrieval=False)

    assert documents[0].metadata["source"] == "guide.html"
    assert "HTML knowledge" in documents[0].page_content


def test_parent_child_chunking_keeps_large_parent_context(tmp_path: Path, monkeypatch) -> None:
    long_text = "段落内容。" * 350
    (tmp_path / "long.txt").write_text(long_text, encoding="utf-8")
    monkeypatch.setattr("ingest.PARENT_CHUNK_SIZE", 900)
    monkeypatch.setattr("ingest.PARENT_CHUNK_OVERLAP", 100)

    documents = load_documents(tmp_path, contextual_retrieval=False)

    assert len(documents) > 2
    assert len(documents[0].metadata["parent_text"]) > len(documents[0].metadata["original_text"])
    assert documents[0].metadata["parent_index"] == 0


def test_load_documents_adds_contextual_retrieval_text(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "book.txt").write_text("武松在快活林醉打蒋门神。", encoding="utf-8")
    monkeypatch.setattr("ingest.contextualize_chunk", lambda source, document, chunk: "《水浒传》中武松在快活林帮助施恩。")

    documents = load_documents(tmp_path, contextual_retrieval=True)

    assert documents[0].page_content.startswith("文档来源：book.txt\n切片上下文：《水浒传》")
    assert documents[0].metadata["original_text"] == "武松在快活林醉打蒋门神。"
    assert documents[0].metadata["contextualized"] is True


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


def test_candidates_use_parent_context_and_deduplicate_same_parent() -> None:
    documents = [
        (
            Document(page_content=f"indexed child {index}", metadata={
                "source": "book.txt",
                "chunk_index": index,
                "parent_index": 0,
                "parent_text": "complete parent paragraph",
                "original_text": f"child {index}",
            }),
            0.9 - index * 0.1,
        )
        for index in range(2)
    ]

    hits = rag.candidates_to_hits(documents, limit=3)

    assert len(hits) == 1
    assert hits[0].text == "complete parent paragraph"
    assert hits[0].matched_text == "child 0"


def test_bm25_tokenize_adds_chinese_bigrams_and_keeps_technical_terms() -> None:
    tokens = rag.bm25_tokenize("谁醉打蒋门神？TOP_K").split()

    assert "蒋门" in tokens
    assert "门神" in tokens
    assert "top_k" in tokens


def test_table_to_markdown_preserves_rows() -> None:
    markdown = table_to_markdown([["名称", "数量"], ["苹果", "2"]])

    assert "| 名称 | 数量 |" in markdown
    assert "| 苹果 | 2 |" in markdown


def test_pdf_extraction_preserves_page_numbers(tmp_path: Path, monkeypatch) -> None:
    import pymupdf

    pdf_path = tmp_path / "pages.pdf"
    document = pymupdf.open()
    for text in ("First page knowledge", "Second page knowledge"):
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(pdf_path)
    document.close()
    monkeypatch.setattr("ingest.OCR_ENABLED", False)

    sections = extract_sections(pdf_path.read_bytes(), ".pdf")

    assert [section.page for section in sections] == [1, 2]
    assert "First page knowledge" in sections[0].text
    assert "Second page knowledge" in sections[1].text


def test_query_router_keeps_forced_category_and_returns_multiple_queries(monkeypatch) -> None:
    class FakeChain:
        def __or__(self, other):
            return self

        def invoke(self, values):
            return '{"category":"西游记","queries":["武松 蒋门神","快活林 事件"]}'

    chain = FakeChain()
    monkeypatch.setattr(rag, "MULTI_QUERY_ENABLED", True)
    monkeypatch.setattr(rag.ChatPromptTemplate, "from_messages", lambda messages: chain)
    monkeypatch.setattr(rag, "ChatOllama", lambda **kwargs: chain)
    monkeypatch.setattr(rag, "StrOutputParser", lambda: chain)

    plan = rag.plan_queries("谁打了蒋门神", "蒋门神是谁打的", "（无历史对话）", "水浒传", ["全部", "水浒传", "西游记"])

    assert plan.category == "水浒传"
    assert plan.queries == ["蒋门神是谁打的", "武松 蒋门神", "快活林 事件"]


def test_multi_query_merges_duplicate_candidates_with_rrf(monkeypatch) -> None:
    shared = Document(page_content="shared", metadata={"source": "book.txt", "chunk_index": 0})
    unique = Document(page_content="unique", metadata={"source": "other.txt", "chunk_index": 1})
    monkeypatch.setattr(rag, "retrieve_mode_candidates", lambda query, category: [(shared, 0.9)] if query == "q1" else [(shared, 0.8), (unique, 0.7)])

    candidates = rag.retrieve_multi_query_candidates(["q1", "q2"], "全部")

    assert [document.page_content for document, _ in candidates] == ["shared", "unique"]
    assert candidates[0][1] > candidates[1][1]


def test_citation_verification_reports_supported_claim(monkeypatch) -> None:
    class FakeChain:
        def __or__(self, other):
            return self

        def invoke(self, values):
            return '{"claims":[{"claim":"武松打了蒋门神","references":[1],"supported":true,"reason":"资料明确记载"}]}'

    chain = FakeChain()
    monkeypatch.setattr(rag, "CITATION_VERIFY_ENABLED", True)
    monkeypatch.setattr(rag.ChatPromptTemplate, "from_messages", lambda messages: chain)
    monkeypatch.setattr(rag, "ChatOllama", lambda **kwargs: chain)
    monkeypatch.setattr(rag, "StrOutputParser", lambda: chain)
    hit = rag.SearchHit("武松醉打蒋门神", 0.8, 0.9, 0, "水浒传.txt")

    result = rag.verify_citations("武松打了蒋门神。[Reference 1]", [hit])

    assert result["status"] == "passed"
    assert result["claims"][0]["supported"] is True
