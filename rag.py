"""LangChain RAG 主流程：切分、索引、召回、重排、增强和生成。"""

# 导入 dataclass，用简洁的数据类表示最终检索结果。
from dataclasses import dataclass
# 导入正则表达式，用于给 BM25 补充中文字符与二元词分词。
import re
# 导入日志模块，用于输出 RAG 各阶段的重要状态。
import logging
# 导入 lru_cache，用于缓存已经加载到内存的 Reranker 对象。
from functools import lru_cache
# 导入 Path，用于指定 Reranker 模型的本地缓存目录。
from pathlib import Path
# 导入高精度计时器，用于统计各阶段耗时。
from time import perf_counter

# 导入 FastEmbed Cross-Encoder，用于对 Qdrant 候选结果重新评分。
from fastembed.rerank.cross_encoder import TextCrossEncoder
# 导入 LangChain Document，用统一结构保存正文和 metadata。
from langchain_core.documents import Document
# 导入字符串输出解析器，把 AIMessage 转换成普通字符串。
from langchain_core.output_parsers import StrOutputParser
# 导入聊天 Prompt 模板，用于组合系统规则、参考资料和问题。
from langchain_core.prompts import ChatPromptTemplate
# 导入 Ollama 的聊天模型组件和 Embedding 组件。
from langchain_ollama import ChatOllama, OllamaEmbeddings
# 导入 Qdrant 向量存储组件，负责写入和检索 Document。
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
# 导入递归文本切分器，尽量按段落、换行、空格等自然边界切分。
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 导入 Qdrant Filter 数据结构，用 metadata.category 限制召回范围。
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

# 从配置模块导入模型、服务、分块和检索相关参数。
from config import (
    # 最终生成答案时使用的聊天模型。
    CHAT_MODEL,
    # 相邻切片期望重叠的字符数。
    CHUNK_OVERLAP,
    # 单个切片允许的最大字符数。
    CHUNK_SIZE,
    # Qdrant 中保存本项目向量的 Collection 名称。
    COLLECTION_NAME,
    # 文档和问题向量化时使用的 Embedding 模型。
    EMBEDDING_MODEL,
    # Ollama 服务地址。
    OLLAMA_URL,
    # Qdrant 服务地址。
    QDRANT_URL,
    # Query Rewrite 是否开启模型推理模式。
    QUERY_REWRITE_REASONING,
    # 在线问答使用的检索模式。
    RETRIEVAL_MODE,
    # Cross-Encoder Reranker 模型名称。
    RERANKER_MODEL,
    # Qdrant 第一阶段最多召回的候选数量。
    RETRIEVAL_K,
    # Qdrant 候选必须达到的最低向量相似度。
    SCORE_THRESHOLD,
    # Reranker 排序后最终保留的片段数量。
    TOP_K,
)

# 创建当前模块的日志记录器。
logger = logging.getLogger(__name__)


# 创建 LangChain Ollama Embedding 组件。
def embeddings() -> OllamaEmbeddings:
    """返回连接本地 bge-m3 的 Embedding 组件。"""
    # 指定模型名称和 Ollama 地址，但此时还没有立即生成向量。
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)


def bm25_tokenize(text: str) -> str:
    """把连续中文补充为单字与二元词，让 BM25 可以匹配中文关键词。"""
    output: list[str] = []
    for token in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_.:-]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            output.extend(token)
            output.extend(token[index:index + 2] for index in range(len(token) - 1))
        else:
            output.append(token)
    return " ".join(output)


class ChineseBM25Sparse(SparseEmbeddings):
    """在 Qdrant/bm25 前增加中文字符与二元词预处理。"""

    def __init__(self, backend: FastEmbedSparse) -> None:
        self.backend = backend

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return self.backend.embed_documents([bm25_tokenize(text) for text in texts])

    def embed_query(self, text: str) -> SparseVector:
        return self.backend.embed_query(bm25_tokenize(text))


@lru_cache(maxsize=1)
def sparse_embeddings() -> SparseEmbeddings:
    """返回带中文分词预处理的本地 BM25 Sparse Embedding。"""
    cache_dir = Path(__file__).with_name(".models")
    backend = FastEmbedSparse(model_name="Qdrant/bm25", cache_dir=str(cache_dir))
    return ChineseBM25Sparse(backend)


# 定义文本切分函数，并允许测试或调用方覆盖默认参数。
def split_text(
    # 要清理和切分的原始文本。
    text: str,
    # 每个切片的最大字符数，默认读取 .env。
    chunk_size: int = CHUNK_SIZE,
    # 相邻切片的期望重叠数，默认读取 .env。
    overlap: int = CHUNK_OVERLAP,
# 返回值是多个字符串切片组成的列表。
) -> list[str]:
    """清理空行，再使用 LangChain 递归切分器切分文本。"""
    # 去除每行首尾空格、跳过空行，再用换行符重新连接。
    clean = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    # 空文本没有可索引内容，直接返回空列表。
    if not clean:
        # 返回空切片集合。
        return []
    # 校验分块参数，防止步长无效或重叠大于切片本身。
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        # 参数非法时抛出清晰异常。
        raise ValueError("require chunk_size > overlap >= 0")

    # 创建递归字符切分器实例。
    splitter = RecursiveCharacterTextSplitter(
        # 设置每个切片最大长度。
        chunk_size=chunk_size,
        # 设置期望重叠长度；递归切分时不保证精确达到。
        chunk_overlap=overlap,
        # 使用 Python len 按字符数衡量文本长度。
        length_function=len,
    )
    # 执行切分并返回字符串列表。
    return splitter.split_text(clean)


# 定义索引重建函数，接收已带 metadata 的 Document 列表。
def rebuild_index(documents: list[Document]) -> int:
    """用 Document 重建 Qdrant Collection，并返回写入数量。"""
    # 没有 Document 时不能创建有效索引，因此主动报错。
    if not documents:
        # 抛出异常提示 knowledge 目录中没有可导入的 TXT 内容。
        raise ValueError("no .txt knowledge documents to index")
    # 记录索引开始信息，但不输出知识正文和向量。
    logger.info("index rebuild started | collection=%s | chunks=%d", COLLECTION_NAME, len(documents))
    # 记录开始时间，用于统计 Embedding 和写入总耗时。
    started = perf_counter()
    # LangChain 自动生成文档向量并写入 Qdrant。
    QdrantVectorStore.from_documents(
        # 传入包含正文和来源信息的 Document。
        documents=documents,
        # 传入 bge-m3 Embedding 组件，让 LangChain 自动向量化。
        embedding=embeddings(),
        # 同时生成 BM25 稀疏向量，用于关键词精确匹配。
        sparse_embedding=sparse_embeddings(),
        # 建立 Dense + Sparse 两套向量并由 Qdrant RRF 融合。
        retrieval_mode=RetrievalMode.HYBRID,
        # 指定 Qdrant HTTP 地址。
        url=QDRANT_URL,
        # 指定要创建的 Collection 名称。
        collection_name=COLLECTION_NAME,
        # 强制删除并重建同名 Collection，保证知识库与文件完全一致。
        force_recreate=True,
    )
    # 输出索引完成日志。
    logger.info("index rebuild completed | chunks=%d | elapsed_ms=%.1f", len(documents), (perf_counter() - started) * 1000)
    # 返回成功提交给向量库的 Document 数量。
    return len(documents)


# 使用 dataclass 自动生成初始化方法和字段表示。
@dataclass
# 定义应用层检索结果，避免把 Qdrant SDK 原始对象暴露给 API。
class SearchHit:
    # 检索到的知识片段正文。
    text: str
    # Qdrant 余弦相似度分数。
    vector_score: float
    # Cross-Encoder 对问题和片段直接比较后的分数。
    rerank_score: float
    # 片段在其来源文件中的编号。
    chunk_index: int
    # 片段来源文件相对于 knowledge 目录的路径。
    source: str
    # 片段所属知识分类。
    category: str = "通用"
    # 方便页面和 Dashboard 识别的 Point 名称。
    point_name: str = "unknown"


# 创建一个连接现有 Qdrant Collection 的 LangChain VectorStore。
def vector_store() -> QdrantVectorStore:
    """连接现有 Qdrant Collection，并启用 Dense + BM25 Hybrid Search。"""
    return QdrantVectorStore(
        client=QdrantClient(url=QDRANT_URL),
        collection_name=COLLECTION_NAME,
        embedding=embeddings(),
        sparse_embedding=sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
    )


def dense_vector_store() -> QdrantVectorStore:
    """只查询 Dense 向量，作为 Hybrid Search 的评估基线。"""
    return QdrantVectorStore(
        client=QdrantClient(url=QDRANT_URL),
        collection_name=COLLECTION_NAME,
        embedding=embeddings(),
        retrieval_mode=RetrievalMode.DENSE,
    )


def sparse_vector_store() -> QdrantVectorStore:
    """只查询 BM25 Sparse 向量，用于精确关键词检索。"""
    return QdrantVectorStore(
        client=QdrantClient(url=QDRANT_URL),
        collection_name=COLLECTION_NAME,
        sparse_embedding=sparse_embeddings(),
        retrieval_mode=RetrievalMode.SPARSE,
    )


# 缓存一个 Reranker 对象，避免每次提问都重新加载约 1 GB 的模型。
@lru_cache(maxsize=1)
# 定义 Reranker 工厂函数。
def reranker() -> TextCrossEncoder:
    """从本地缓存加载 Cross-Encoder，并在进程内复用。"""
    # 将模型文件保存在项目根目录的 .models 文件夹。
    cache_dir = Path(__file__).with_name(".models")
    # 创建 FastEmbed TextCrossEncoder；没有模型文件时会自动下载。
    return TextCrossEncoder(model_name=RERANKER_MODEL, cache_dir=str(cache_dir))


# 定义第一阶段召回函数，返回 Document 与向量分数组成的元组列表。
def retrieve_candidates(question: str, category: str = "全部") -> list[tuple[Document, float]]:
    """使用 Dense + BM25 Hybrid Search 召回候选 Document。"""
    # 记录向量召回阶段开始时间。
    started = perf_counter()
    # “全部”不限制分类；具体分类使用 Qdrant Payload Filter。
    category_filter = None if category == "全部" else Filter(
        # must 表示返回的 Point 必须满足下面的 category 条件。
        must=[FieldCondition(key="metadata.category", match=MatchValue(value=category))]
    )
    # similarity_search_with_score 内部会调用 embed_query(question)。
    candidates = vector_store().similarity_search_with_score(
        # 传入用户的自然语言问题。
        query=question,
        # 最多召回 RETRIEVAL_K 个候选。
        k=RETRIEVAL_K,
        # Hybrid 返回的是 RRF 融合分数，不再套用 Dense 余弦阈值。
        score_threshold=None,
        # 在向量相似度检索前限制允许参与检索的分类。
        filter=category_filter,
    )
    # 输出候选数量、最高分和耗时；没有候选时最高分显示 none。
    logger.info(
        "hybrid retrieval completed | category=%s | fusion=rrf | candidates=%d | top_score=%s | elapsed_ms=%.1f",
        category,
        len(candidates),
        f"{candidates[0][1]:.4f}" if candidates else "none",
        (perf_counter() - started) * 1000,
    )
    # 返回原始 Qdrant 候选及向量分数。
    return candidates


def retrieve_dense_candidates(question: str, category: str = "全部") -> list[tuple[Document, float]]:
    """仅使用 Dense Embedding 召回，供评估页面与 Hybrid Search 对比。"""
    category_filter = None if category == "全部" else Filter(
        must=[FieldCondition(key="metadata.category", match=MatchValue(value=category))]
    )
    return dense_vector_store().similarity_search_with_score(
        query=question,
        k=RETRIEVAL_K,
        score_threshold=SCORE_THRESHOLD,
        filter=category_filter,
    )


def retrieve_sparse_candidates(question: str, category: str = "全部") -> list[tuple[Document, float]]:
    """仅使用 BM25 Sparse Embedding 召回关键词匹配片段。"""
    category_filter = None if category == "全部" else Filter(
        must=[FieldCondition(key="metadata.category", match=MatchValue(value=category))]
    )
    return sparse_vector_store().similarity_search_with_score(
        query=question,
        k=RETRIEVAL_K,
        filter=category_filter,
    )


def retrieve_mode_candidates(question: str, category: str = "全部") -> list[tuple[Document, float]]:
    """按照 .env 的 RETRIEVAL_MODE 选择在线召回方式。"""
    if RETRIEVAL_MODE == "vector":
        return retrieve_dense_candidates(question, category)
    if RETRIEVAL_MODE == "bm25":
        return retrieve_sparse_candidates(question, category)
    return retrieve_candidates(question, category)


def candidates_to_hits(documents_with_scores: list[tuple[Document, float]], limit: int = TOP_K) -> list[SearchHit]:
    """把不需要 Reranker 的原始召回结果转换成页面统一结构。"""
    return [
        SearchHit(
            text=document.page_content,
            vector_score=float(score),
            rerank_score=0.0,
            chunk_index=int(document.metadata.get("chunk_index", 0)),
            source=str(document.metadata.get("source", "unknown")),
            category=str(document.metadata.get("category", "通用")),
            point_name=str(document.metadata.get("point_name", "unknown")),
        )
        for document, score in documents_with_scores[:limit]
    ]


# 定义第二阶段重排函数，并允许评估时取消最终数量限制。
def rerank_candidates(
    # 用户问题，Cross-Encoder 会将它与每个候选片段一起编码。
    question: str,
    # Qdrant 返回的 Document 和 vector_score 列表。
    documents_with_scores: list[tuple[Document, float]],
    # 默认只保留 TOP_K；传入 None 时保留全部重排结果。
    limit: int | None = TOP_K,
# 返回应用层 SearchHit 列表。
) -> list[SearchHit]:
    """用 Cross-Encoder 重排候选，并转换成 SearchHit。"""
    # 如果 Qdrant 没有返回候选，就无需加载和调用 Reranker。
    if not documents_with_scores:
        # 返回空列表，让上层决定如何拒答。
        return []

    # 记录重排开始时间，首次加载 ONNX 模型的时间也会包含在内。
    started = perf_counter()
    # 计算问题与每个候选片段的 Cross-Encoder 相关性分数。
    rerank_scores = list(
        # 获取缓存的 Reranker，并执行批量重排评分。
        reranker().rerank(
            # 第一个参数是所有候选共用的用户问题。
            question,
            # 从 (Document, vector_score) 中只提取 Document 正文。
            [document.page_content for document, _ in documents_with_scores],
        )
    )
    # 将候选和 Reranker 分数一一配对，再按 Reranker 分数降序排列。
    ranked_results = sorted(
        # strict=True 确保候选数量与评分数量完全一致。
        zip(documents_with_scores, rerank_scores, strict=True),
        # 每项结构是 ((Document, vector_score), rerank_score)，索引 1 是重排分数。
        key=lambda result: result[1],
        # 分数越高越相关，因此采用从高到低排序。
        reverse=True,
    )
    # 正常问答时只保留最终 TOP_K；评估时传入 None 保留完整排名。
    if limit is not None:
        # 使用列表切片取得排序后的前 limit 项。
        ranked_results = ranked_results[:limit]
    # 输出重排后的数量、最高分和耗时。
    logger.info(
        "rerank completed | input=%d | output=%d | top_score=%.4f | elapsed_ms=%.1f",
        len(documents_with_scores),
        len(ranked_results),
        float(ranked_results[0][1]),
        (perf_counter() - started) * 1000,
    )
    # 将内部元组结构转换成更清晰的 SearchHit 列表。
    return [
        # 为每个重排结果创建一个 SearchHit。
        SearchHit(
            # 保存 LangChain Document 正文。
            text=document.page_content,
            # 将 Qdrant 分数统一转换成 Python float。
            vector_score=float(vector_score),
            # 将 ONNX 模型输出统一转换成 Python float。
            rerank_score=float(rerank_score),
            # 从 metadata 读取切片编号，缺失时使用 0。
            chunk_index=int(document.metadata.get("chunk_index", 0)),
            # 从 metadata 读取来源文件，缺失时使用 unknown。
            source=str(document.metadata.get("source", "unknown")),
            # 从 metadata 读取知识分类。
            category=str(document.metadata.get("category", "通用")),
            # 从 metadata 读取可读 Point 名称。
            point_name=str(document.metadata.get("point_name", "unknown")),
        )
        # 对排序结果做嵌套解包，获得 Document、两个分数。
        for (document, vector_score), rerank_score in ranked_results
    ]


# 定义完整检索函数，串联 Qdrant 召回和 Cross-Encoder 重排。
def retrieve(question: str, category: str = "全部") -> list[SearchHit]:
    """按照配置运行召回，并仅在 hybrid_rerank 模式执行 Cross-Encoder。"""
    candidates = retrieve_mode_candidates(question, category)
    if RETRIEVAL_MODE == "hybrid_rerank":
        return rerank_candidates(question, candidates)
    return candidates_to_hits(candidates)


# 定义 Query Rewrite 函数，把口语化问题转换成更适合向量召回的独立查询。
def rewrite_query(question: str, chat_history: str = "（无历史对话）") -> str:
    """使用本地聊天模型改写检索查询；失败或结果为空时回退原问题。"""
    # 创建只负责查询改写的 Prompt，不给模型知识资料，也不让它回答问题。
    rewrite_prompt = ChatPromptTemplate.from_messages(
        # 两条消息分别规定改写规则并提供原始问题。
        [
            # system 消息要求保留实体、技术名词、数字和原始语言。
            (
                "system",
                "你是向量检索查询改写器。将用户问题改写成一条语义明确、可独立理解、"
                "适合知识库向量检索的查询。结合历史对话消解‘它、这个、上一点’等指代，"
                "但不要改变用户意图。保留重要实体、模型名、数字和限制条件。"
                "不要回答问题，不要解释，不要添加知识，只输出改写后的查询。",
            ),
            # human 消息把用户原始问题填入 question 占位符。
            ("human", "历史对话：\n{chat_history}\n\n当前问题：{question}\n独立检索查询："),
        ]
    )
    # 使用与最终生成相同的本地 Qwen，并将温度设为零以提高稳定性。
    rewrite_model = ChatOllama(
        # 从配置读取聊天模型名称。
        model=CHAT_MODEL,
        # 连接本机 Ollama 服务。
        base_url=OLLAMA_URL,
        # 零温度减少不必要的随机改写。
        temperature=0,
        # 根据 .env 决定是否让模型先推理再输出改写文本。
        reasoning=QUERY_REWRITE_REASONING,
    )
    # 把改写 Prompt、模型和字符串解析器组成独立 LCEL Chain。
    rewrite_chain = rewrite_prompt | rewrite_model | StrOutputParser()
    # 记录 Query Rewrite 开始时间。
    started = perf_counter()
    # 捕获模型服务异常，确保改写失败不会阻断原本可用的检索流程。
    try:
        # 调用模型并清理首尾空白及模型偶尔输出的包裹引号。
        rewritten_query = rewrite_chain.invoke({"question": question, "chat_history": chat_history}).strip().strip('"“”')
    # 任何改写阶段异常都回退为原始问题。
    except Exception:
        # 保存异常堆栈，并说明当前请求将回退到原问题。
        logger.exception("query rewrite failed; falling back to original query")
        # 返回原问题，让 Qdrant 仍可继续工作。
        return question
    # 模型返回空字符串时使用原问题，否则返回改写后的查询。
    effective_query = rewritten_query or question
    # 输出原问题、有效改写结果和耗时，便于学习和调试检索效果。
    logger.info(
        "query rewrite completed | original=%r | rewritten=%r | elapsed_ms=%.1f",
        question,
        effective_query,
        (perf_counter() - started) * 1000,
    )
    # 返回最终用于 Qdrant 的查询。
    return effective_query


# 定义生成函数，接收原问题和最终检索片段。
def generate(question: str, hits: list[SearchHit], chat_history: str = "（无历史对话）") -> str:
    """用检索片段增强 Prompt，并调用本地 Qwen 生成答案。"""
    # 给每个片段添加 Reference 和来源标签，再用空行连接成上下文。
    context = "\n\n".join(
        # 每条参考资料包含顺序编号、来源文件和实际正文。
        f"[Reference {index} | Source: {hit.source}]\n{hit.text}"
        # 从 1 开始给重排后的 hits 编号。
        for index, hit in enumerate(hits, start=1)
    )
    # 创建包含 system 和 human 两种角色的聊天 Prompt 模板。
    prompt = ChatPromptTemplate.from_messages(
        # 消息按顺序发送给聊天模型。
        [
            # system 消息用于限制模型只能依据检索资料回答。
            (
                # 声明当前消息角色是 system。
                "system",
                # 第一段规则用于降低无资料时的编造风险。
                "只根据参考资料回答；资料不足时明确说明不知道。"
                # 第二段规则要求回答显式标注引用编号。
                "回答时使用[Reference N]标明依据。",
            ),
            # human 消息中的占位符会在 invoke 时替换为真实上下文和问题。
            ("human", "历史对话：\n{chat_history}\n\n参考资料：\n{context}\n\n当前问题：{question}"),
        ]
    )
    # 创建连接本地 Ollama 的 LangChain 聊天模型。
    model = ChatOllama(
        # 指定 .env 中配置的 Qwen 模型。
        model=CHAT_MODEL,
        # 指定本机 Ollama API 地址。
        base_url=OLLAMA_URL,
        # 使用零温度，让回答更加稳定和可复现。
        temperature=0,
        # 关闭单独的推理过程输出，只返回最终内容。
        reasoning=False,
    )
    # 使用 LCEL 将 Prompt、模型和字符串解析器连接成执行链。
    generation_chain = prompt | model | StrOutputParser()
    # 记录答案生成开始时间。
    started = perf_counter()
    # 填充 Prompt 占位符并执行整条链，最终返回普通字符串。
    answer = generation_chain.invoke({"context": context, "question": question, "chat_history": chat_history})
    # 输出参考数量、答案字符数和生成耗时，不打印完整 Prompt 和知识正文。
    logger.info("generation completed | references=%d | answer_chars=%d | elapsed_ms=%.1f", len(hits), len(answer), (perf_counter() - started) * 1000)
    # 返回模型生成的最终答案。
    return answer
