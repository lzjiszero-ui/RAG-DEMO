"""读取 knowledge 目录下的所有 TXT 文件，并将切片写入 Qdrant。"""

# 导入 Path，用于递归查找知识文件和处理相对路径。
from pathlib import Path
# 导入日志模块，用于记录文件读取和索引进度。
import logging

# 导入 LangChain Document，用统一结构保存正文和元数据。
from langchain_core.documents import Document

# 导入重建索引和文本切分函数。
from rag import rebuild_index, split_text
# 导入统一日志初始化函数。
from logging_config import configure_logging

# 创建当前导入模块的日志记录器。
logger = logging.getLogger(__name__)


# 定义多文件加载函数，参数是知识库目录，返回 Document 列表。
def load_documents(knowledge_dir: Path) -> list[Document]:
    """创建 LangChain Document，并保留每个切片的来源文件。"""
    # 创建空列表，用于收集所有文件产生的 Document。
    documents: list[Document] = []
    # 递归查找所有 .txt 文件，并排序以保证每次导入顺序稳定。
    for path in sorted(knowledge_dir.rglob("*.txt")):
        # 计算相对 knowledge 目录的路径，作为可读的 source 元数据。
        source = path.relative_to(knowledge_dir).as_posix()
        # 以 UTF-8 读取文件，并调用文本切分器生成多个字符串切片。
        chunks = split_text(path.read_text(encoding="utf-8"))
        # 输出当前知识文件产生的切片数量。
        logger.info("knowledge file loaded | source=%s | chunks=%d", source, len(chunks))
        # 把当前文件的每个切片转换成 Document，并追加到总列表。
        documents.extend(
            # page_content 保存正文；metadata 保存来源和片段编号。
            Document(
                # 将当前切片正文保存到 LangChain 的标准正文字段。
                page_content=chunk,
                # 记录来源文件和当前文件内的切片序号。
                metadata={"source": source, "chunk_index": index},
            )
            # enumerate 同时产生从 0 开始的编号和对应切片。
            for index, chunk in enumerate(chunks)
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
