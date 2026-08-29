"""读取 knowledge 目录下的所有 TXT 文件，并将切片写入 Qdrant。"""

# 导入 Path，用于递归查找知识文件和处理相对路径。
from pathlib import Path
# 导入缓存装饰器，复用上下文生成 Chain 和模型连接。
from functools import lru_cache
# 导入日志模块，用于记录文件读取和索引进度。
import logging

# 导入 LangChain Document，用统一结构保存正文和元数据。
from langchain_core.documents import Document
# 导入字符串解析器，把模型消息转换成普通文本。
from langchain_core.output_parsers import StrOutputParser
# 导入 Prompt 模板，规定 Contextual Retrieval 的生成规则。
from langchain_core.prompts import ChatPromptTemplate
# 导入本地 Ollama 聊天模型，用于给每个切片生成短上下文。
from langchain_ollama import ChatOllama

# 导入重建索引和文本切分函数。
from rag import rebuild_index, split_text
# 导入统一日志初始化函数。
from logging_config import configure_logging
# 导入 Contextual Retrieval 开关、文档长度限制和模型配置。
from config import CHAT_MODEL, CONTEXTUAL_MAX_DOCUMENT_CHARS, CONTEXTUAL_RETRIEVAL, OLLAMA_URL

# 创建当前导入模块的日志记录器。
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def contextualization_chain():
    """创建并缓存只负责生成切片上下文的本地 Qwen Chain。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是知识库索引助手。根据文档整体内容，生成一段不超过80字的切片上下文。补充作品名、主题、人物或事件，使片段能独立理解。不要回答问题，不要添加文档外知识，只输出上下文本身。"),
            ("human", "来源：{source}\n\n完整文档：\n{document}\n\n当前切片：\n{chunk}\n\n切片上下文："),
        ]
    )
    model = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_URL, temperature=0, reasoning=False, num_predict=120)
    return prompt | model | StrOutputParser()


def contextualize_chunk(source: str, document_text: str, chunk: str) -> str:
    """结合完整文档为单个切片生成简短的检索上下文。"""
    try:
        return contextualization_chain().invoke(
            {"source": source, "document": document_text[:CONTEXTUAL_MAX_DOCUMENT_CHARS], "chunk": chunk}
        ).strip()
    except Exception:
        logger.exception("contextualization failed | source=%s", source)
        return ""


# 定义多文件加载函数，参数是知识库目录，返回 Document 列表。
def load_documents(knowledge_dir: Path, contextual_retrieval: bool = CONTEXTUAL_RETRIEVAL) -> list[Document]:
    """创建 LangChain Document，并保留每个切片的来源文件。"""
    # 创建空列表，用于收集所有文件产生的 Document。
    documents: list[Document] = []
    # 递归查找所有 .txt 文件，并排序以保证每次导入顺序稳定。
    for path in sorted(knowledge_dir.rglob("*.txt")):
        # 计算相对 knowledge 目录的路径，作为可读的 source 元数据。
        relative_path = path.relative_to(knowledge_dir)
        # 使用一级子目录作为知识分类；根目录文件归入“通用”。
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else "通用"
        # 将相对路径转换成跨平台统一的 source 字符串。
        source = relative_path.as_posix()
        # 以 UTF-8 读取完整文档，供切片和上下文生成共同使用。
        document_text = path.read_text(encoding="utf-8")
        # 调用文本切分器生成多个字符串切片。
        chunks = split_text(document_text)
        # 输出当前知识文件产生的切片数量。
        logger.info("knowledge file loaded | source=%s | chunks=%d", source, len(chunks))
        # 逐个处理切片，因为 Contextual Retrieval 需要分别调用模型。
        for index, chunk in enumerate(chunks):
            # 开启时生成文档级上下文；关闭时保留原有索引行为。
            contextual_summary = contextualize_chunk(source, document_text, chunk) if contextual_retrieval else ""
            # 把短上下文放在原始片段之前，让 Dense 与 BM25 都能检索到补充信息。
            indexed_content = f"文档来源：{source}\n切片上下文：{contextual_summary}\n原始片段：{chunk}" if contextual_summary else chunk
            # 创建包含增强正文和可追踪 Metadata 的 LangChain Document。
            documents.append(
                Document(
                    page_content=indexed_content,
                    metadata={
                        "source": source,
                        "category": category,
                        "point_name": f"{path.stem}-{index + 1}",
                        "chunk_index": index,
                        "contextual_summary": contextual_summary,
                        "original_text": chunk,
                        "contextualized": bool(contextual_summary),
                    },
                )
            )
    # 返回所有知识文件产生的 Document。
    return documents


# 定义脚本入口函数，负责加载文件并重建整个向量索引。
def main() -> None:
    # 初始化导入脚本的日志格式和级别。
    configure_logging()
    # 定位项目根目录下的 knowledge 文件夹。
    knowledge_dir = Path(__file__).with_name("knowledge")
    # 读取并切分 knowledge 下的全部 TXT 文件。
    documents = load_documents(knowledge_dir)
    # 输出知识文件扫描完成后的总切片数量。
    logger.info("knowledge scan completed | chunks=%d", len(documents))
    # 重新创建 Qdrant Collection，并返回写入的切片数量。
    count = rebuild_index(documents)
    # 用集合去重 source，统计实际导入的文件数量。
    file_count = len({document.metadata["source"] for document in documents})
    # 在终端输出本次导入结果，方便确认是否成功。
    logger.info("knowledge import completed | files=%d | chunks=%d", file_count, count)


# 只有直接执行 ingest.py 时才调用 main；被其他模块导入时不会自动运行。
if __name__ == "__main__":
    # 启动知识库导入流程。
    main()
