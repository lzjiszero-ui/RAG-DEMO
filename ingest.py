"""读取 knowledge 目录下的 TXT、PDF、HTML 文件，并将切片写入 Qdrant。"""

# 导入 Path，用于递归查找知识文件和处理相对路径。
from pathlib import Path
from io import BytesIO
# 导入缓存装饰器，复用上下文生成 Chain 和模型连接。
from functools import lru_cache
# 导入日志模块，用于记录文件读取和索引进度。
import logging
import re

from bs4 import BeautifulSoup
from pypdf import PdfReader

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

# 声明前台和命令行导入共同支持的文件扩展名。
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".html", ".htm"}


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


def html_to_text(content: bytes) -> str:
    """从 HTML 字节中删除脚本和样式，只保留适合检索的可见正文。"""
    # BeautifulSoup 会读取 meta charset；没有声明编码时使用其自动检测结果。
    soup = BeautifulSoup(content, "html.parser")
    # 删除不会展示给读者、也不应进入知识库的节点。
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()
    # 用换行保留标题和段落边界，便于递归切片器优先保持段落完整。
    text = soup.get_text("\n", strip=True)
    # 合并过多空行，减少导航布局产生的无意义空白。
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_text(content: bytes, suffix: str) -> str:
    """按照扩展名把 TXT、PDF 或 HTML 文件转换成纯文本。"""
    # 统一扩展名大小写，兼容 .PDF 等文件名。
    extension = suffix.lower()
    # TXT 优先使用 UTF-8，带 BOM 的文件也能正常读取。
    if extension == ".txt":
        try:
            return content.decode("utf-8-sig").strip()
        except UnicodeDecodeError:
            # 一些中文 Windows 文本使用 GB18030，作为明确的兼容回退。
            return content.decode("gb18030").strip()
    # PDF 逐页提取文本；扫描图片型 PDF 没有文本层时会得到空字符串。
    if extension == ".pdf":
        reader = PdfReader(BytesIO(content))
        return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
    # 本地 HTML 与网页 URL 使用相同的正文清洗规则。
    if extension in {".html", ".htm"}:
        return html_to_text(content)
    # 调用方漏做格式校验时给出清晰错误。
    raise ValueError(f"unsupported file type: {extension}")


def build_documents(
    document_text: str,
    source: str,
    category: str,
    point_stem: str,
    contextual_retrieval: bool = CONTEXTUAL_RETRIEVAL,
) -> list[Document]:
    """把一份已解析文本转换成可写入 Qdrant 的 Document 切片。"""
    # 拒绝空文件以及没有文本层的扫描 PDF。
    if not document_text.strip():
        raise ValueError("文件中没有可提取的文本；扫描版 PDF 需要先进行 OCR")
    # 使用项目统一的切片大小和 overlap 配置。
    chunks = split_text(document_text)
    # 收集这一个来源生成的全部 LangChain Document。
    documents: list[Document] = []
    # 逐片生成可检索正文及来源元数据。
    for index, chunk in enumerate(chunks):
        contextual_summary = contextualize_chunk(source, document_text, chunk) if contextual_retrieval else ""
        indexed_content = f"文档来源：{source}\n切片上下文：{contextual_summary}\n原始片段：{chunk}" if contextual_summary else chunk
        documents.append(Document(page_content=indexed_content, metadata={
            "source": source,
            "category": category,
            "point_name": f"{point_stem}-{index + 1}",
            "chunk_index": index,
            "contextual_summary": contextual_summary,
            "original_text": chunk,
            "contextualized": bool(contextual_summary),
        }))
    return documents


# 定义多文件加载函数，参数是知识库目录，返回 Document 列表。
def load_documents(knowledge_dir: Path, contextual_retrieval: bool = CONTEXTUAL_RETRIEVAL) -> list[Document]:
    """创建 LangChain Document，并保留每个切片的来源文件。"""
    # 创建空列表，用于收集所有文件产生的 Document。
    documents: list[Document] = []
    # 递归查找支持的文件，并排序以保证每次导入顺序稳定。
    for path in sorted(item for item in knowledge_dir.rglob("*") if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS):
        # 计算相对 knowledge 目录的路径，作为可读的 source 元数据。
        relative_path = path.relative_to(knowledge_dir)
        # 使用一级子目录作为知识分类；根目录文件归入“通用”。
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else "通用"
        # 将相对路径转换成跨平台统一的 source 字符串。
        source = relative_path.as_posix()
        # 按文件格式提取正文，再走统一的切片与 Contextual Retrieval 流程。
        document_text = extract_text(path.read_bytes(), path.suffix)
        file_documents = build_documents(document_text, source, category, path.stem, contextual_retrieval)
        # 输出当前知识文件产生的切片数量。
        logger.info("knowledge file loaded | source=%s | chunks=%d", source, len(file_documents))
        # 合并到本次完整重建列表。
        documents.extend(file_documents)
    # 返回所有知识文件产生的 Document。
    return documents


# 定义脚本入口函数，负责加载文件并重建整个向量索引。
def main() -> None:
    # 初始化导入脚本的日志格式和级别。
    configure_logging()
    # 定位项目根目录下的 knowledge 文件夹。
    knowledge_dir = Path(__file__).with_name("knowledge")
    # 读取并切分 knowledge 下的全部 TXT、PDF 和 HTML 文件。
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
