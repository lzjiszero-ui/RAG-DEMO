"""读取 knowledge 目录下的 TXT、PDF、HTML 文件，并将切片写入 Qdrant。"""

# 导入 Path，用于递归查找知识文件和处理相对路径。
from pathlib import Path
from io import BytesIO
from dataclasses import dataclass
# 导入回调类型，用于把长文档逐切片进度通知给后台任务管理器。
from collections.abc import Callable
# 导入缓存装饰器，复用上下文生成 Chain 和模型连接。
from functools import lru_cache
# 导入日志模块，用于记录文件读取和索引进度。
import logging
import re

from bs4 import BeautifulSoup
from pypdf import PdfReader
import pdfplumber
import pymupdf
from rapidocr_onnxruntime import RapidOCR

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
from config import CHAT_MODEL, CONTEXTUAL_MAX_DOCUMENT_CHARS, CONTEXTUAL_RETRIEVAL, OCR_ENABLED, OCR_MIN_TEXT_CHARS, OLLAMA_URL, PARENT_CHUNK_OVERLAP, PARENT_CHUNK_SIZE

# 创建当前导入模块的日志记录器。
logger = logging.getLogger(__name__)

# 声明前台和命令行导入共同支持的文件扩展名。
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".html", ".htm"}


@dataclass
class ExtractedSection:
    """表示带页码和内容类型的一段解析结果。"""
    text: str
    page: int | None = None
    section: str = "正文"
    content_type: str = "text"
    ocr_used: bool = False


@lru_cache(maxsize=1)
def ocr_engine() -> RapidOCR:
    """延迟加载本地 OCR 模型，普通文本 PDF 不承担模型启动成本。"""
    return RapidOCR()


def table_to_markdown(table: list[list[object | None]]) -> str:
    """把 PDF 表格转换成适合检索与回答引用的 Markdown。"""
    rows = [[str(cell or "").replace("|", "\\|").replace("\n", " ").strip() for cell in row] for row in table if row]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    body = normalized[1:]
    return "\n".join([
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
        *("| " + " | ".join(row) + " |" for row in body),
    ])


def ocr_pdf_page(pdf_document: pymupdf.Document, page_index: int) -> str:
    """把指定 PDF 页面渲染成图片并使用 RapidOCR 提取文字。"""
    page = pdf_document.load_page(page_index)
    image = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).tobytes("png")
    result, _ = ocr_engine()(image)
    return "\n".join(str(item[1]).strip() for item in (result or []) if len(item) > 1 and str(item[1]).strip())


def extract_sections(content: bytes, suffix: str) -> list[ExtractedSection]:
    """解析文件并保留 PDF 页码、表格和 OCR 来源信息。"""
    extension = suffix.lower()
    if extension != ".pdf":
        return [ExtractedSection(text=extract_text(content, extension))]
    reader = PdfReader(BytesIO(content))
    pdf_document = pymupdf.open(stream=content, filetype="pdf")
    sections: list[ExtractedSection] = []
    with pdfplumber.open(BytesIO(content)) as plumber_document:
        for page_index, page in enumerate(reader.pages):
            page_number = page_index + 1
            text = (page.extract_text() or "").strip()
            ocr_used = False
            if OCR_ENABLED and len(re.sub(r"\s+", "", text)) < OCR_MIN_TEXT_CHARS:
                text = ocr_pdf_page(pdf_document, page_index).strip()
                ocr_used = bool(text)
            tables = plumber_document.pages[page_index].extract_tables()
            markdown_tables = [table_to_markdown(table) for table in tables]
            markdown_tables = [table for table in markdown_tables if table]
            combined = "\n\n".join(part for part in [text, *markdown_tables] if part).strip()
            if combined:
                content_type = "table" if markdown_tables and not text else "mixed" if markdown_tables else "ocr" if ocr_used else "text"
                sections.append(ExtractedSection(combined, page_number, f"第 {page_number} 页", content_type, ocr_used))
    pdf_document.close()
    return sections


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
    for node in soup(["script", "style", "noscript", "svg", "template", "textarea", "nav", "footer", "aside", "form"]):
        node.decompose()
    # 删除通过标准属性明确隐藏的模板节点。
    for node in soup.select("[hidden], [aria-hidden='true']"):
        node.decompose()
    # 文章页优先使用 article 或 main 中正文最多的候选区域。
    candidates = soup.select("article, main, [role='main']")
    content_root = max(candidates, key=lambda node: len(node.get_text(" ", strip=True))) if candidates else (soup.body or soup)
    # 用换行保留标题和段落边界，便于递归切片器优先保持段落完整。
    raw_text = content_root.get_text("\n", strip=True)
    # 删除完全相同的重复行，常见于响应式导航和重复模板。
    seen: set[str] = set()
    lines = []
    for line in raw_text.splitlines():
        normalized = re.sub(r"\s+", " ", line).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            lines.append(normalized)
    text = "\n".join(lines)
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
    progress_callback: Callable[[str, int, str], None] | None = None,
    max_chunks: int | None = None,
    base_metadata: dict | None = None,
    chunk_offset: int = 0,
) -> list[Document]:
    """把一份已解析文本转换成可写入 Qdrant 的 Document 切片。"""
    # 拒绝空文件以及没有文本层的扫描 PDF。
    if not document_text.strip():
        raise ValueError("文件中没有可提取的文本；扫描版 PDF 需要先进行 OCR")
    # 使用项目统一的切片大小和 overlap 配置。
    # 先切成较大的父段落，再在每个父段落内部生成用于精准检索的子切片。
    parent_chunks = split_text(document_text, chunk_size=PARENT_CHUNK_SIZE, overlap=PARENT_CHUNK_OVERLAP)
    child_records = [
        (parent_index, parent_text, child_text)
        for parent_index, parent_text in enumerate(parent_chunks)
        for child_text in split_text(parent_text)
    ]
    all_chunks = child_records
    # 网页导入可以设置切片保护上限；命令行和本地文件默认不限制。
    chunks = all_chunks[:max_chunks] if max_chunks else all_chunks
    # 切片完成后立即报告数量，页面无需等待全部上下文生成才有反馈。
    if progress_callback:
        suffix = f"，已限制为前 {len(chunks)} 个" if len(all_chunks) > len(chunks) else ""
        progress_callback("chunking", 20, f"文本切分完成，共 {len(all_chunks)} 个切片{suffix}")
    # 收集这一个来源生成的全部 LangChain Document。
    documents: list[Document] = []
    # 逐片生成可检索正文及来源元数据。
    for index, (parent_index, parent_text, chunk) in enumerate(chunks):
        absolute_index = chunk_offset + index
        # Contextual Retrieval 较慢，因此在每个切片前报告真实序号。
        if progress_callback and contextual_retrieval:
            progress_callback("contextualizing", 20 + int(50 * index / max(1, len(chunks))), f"正在生成切片上下文 {index + 1}/{len(chunks)}")
        contextual_summary = contextualize_chunk(source, document_text, chunk) if contextual_retrieval else ""
        indexed_content = f"文档来源：{source}\n切片上下文：{contextual_summary}\n原始片段：{chunk}" if contextual_summary else chunk
        documents.append(Document(page_content=indexed_content, metadata={
            "source": source,
            "category": category,
            "point_name": f"{point_stem}-{absolute_index + 1}",
            "chunk_index": absolute_index,
            "parent_index": chunk_offset + parent_index,
            "parent_text": parent_text,
            "contextual_summary": contextual_summary,
            "original_text": chunk,
            "contextualized": bool(contextual_summary),
            **(base_metadata or {}),
        }))
    # 无论是否开启 Contextual Retrieval，都明确表示文档构建阶段完成。
    if progress_callback:
        progress_callback("contextualizing" if contextual_retrieval else "chunking", 70, f"已准备 {len(documents)} 个待向量化切片")
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
        sections = extract_sections(path.read_bytes(), path.suffix)
        document_text = "\n\n".join(section.text for section in sections)
        file_documents = []
        for section in sections:
            section_documents = build_documents(
                section.text,
                source,
                category,
                path.stem,
                contextual_retrieval,
                base_metadata={"page": section.page, "section": section.section, "content_type": section.content_type, "ocr_used": section.ocr_used},
                chunk_offset=len(file_documents),
            )
            file_documents.extend(section_documents)
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
