"""The LangChain RAG pipeline: split, index, retrieve, augment, generate."""

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHAT_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_URL,
    QDRANT_URL,
    SCORE_THRESHOLD,
    TOP_K,
)


def embeddings() -> OllamaEmbeddings:
    """Create LangChain's Ollama embedding component."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Clean text and split it with LangChain's recursive text splitter."""
    clean = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not clean:
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("require chunk_size > overlap >= 0")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_text(clean)


def rebuild_index(documents: list[Document]) -> int:
    """Replace the collection using LangChain Documents and QdrantVectorStore."""
    if not documents:
        raise ValueError("no .txt knowledge documents to index")
    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings(),
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )
    return len(documents)


@dataclass
class SearchHit:
    text: str
    score: float
    chunk_index: int
    source: str


def vector_store() -> QdrantVectorStore:
    """Connect LangChain to the existing Qdrant collection."""
    return QdrantVectorStore.from_existing_collection(
        collection_name=COLLECTION_NAME,
        embedding=embeddings(),
        url=QDRANT_URL,
    )


def retrieve(question: str) -> list[SearchHit]:
    """Retrieve similar LangChain Documents from Qdrant."""
    documents_with_scores = vector_store().similarity_search_with_score(
        query=question,
        k=TOP_K,
        score_threshold=SCORE_THRESHOLD,
    )
    return [
        SearchHit(
            text=document.page_content,
            score=float(score),
            chunk_index=int(document.metadata.get("chunk_index", 0)),
            source=str(document.metadata.get("source", "unknown")),
        )
        for document, score in documents_with_scores
    ]


def generate(question: str, hits: list[SearchHit]) -> str:
    """Augment the prompt with retrieved context and invoke ChatOllama."""
    context = "\n\n".join(
        f"[Reference {index} | Source: {hit.source}]\n{hit.text}"
        for index, hit in enumerate(hits, start=1)
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "只根据参考资料回答；资料不足时明确说明不知道。"
                "回答时使用[Reference N]标明依据。",
            ),
            ("human", "参考资料：\n{context}\n\n问题：{question}"),
        ]
    )
    model = ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
        reasoning=False,
    )
    generation_chain = prompt | model | StrOutputParser()
    return generation_chain.invoke({"context": context, "question": question})


def ask(question: str) -> tuple[str, list[SearchHit]]:
    hits = retrieve(question)
    if not hits:
        return "知识库中没有找到足够相关的资料。", []
    return generate(question, hits), hits
