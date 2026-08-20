"""Load every knowledge/*.txt file and store its chunks in Qdrant."""

from pathlib import Path

from langchain_core.documents import Document

from rag import rebuild_index, split_text


def load_documents(knowledge_dir: Path) -> list[Document]:
    """Create LangChain Documents while preserving each chunk's source file."""
    documents: list[Document] = []
    for path in sorted(knowledge_dir.rglob("*.txt")):
        source = path.relative_to(knowledge_dir).as_posix()
        chunks = split_text(path.read_text(encoding="utf-8"))
        documents.extend(
            Document(
                page_content=chunk,
                metadata={"source": source, "chunk_index": index},
            )
            for index, chunk in enumerate(chunks)
        )
    return documents


def main() -> None:
    knowledge_dir = Path(__file__).with_name("knowledge")
    documents = load_documents(knowledge_dir)
    count = rebuild_index(documents)
    file_count = len({document.metadata["source"] for document in documents})
    print(f"Imported {count} chunks from {file_count} files into Qdrant.")


if __name__ == "__main__":
    main()
