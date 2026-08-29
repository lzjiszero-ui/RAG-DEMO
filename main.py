"""Simple RAG 的 FastAPI 入口，同时提供 API 和本地图形界面。"""

# 导入 Path，用于定位 frontend 静态文件目录。
from pathlib import Path
# 导入 json，用于把流式进度事件编码成 NDJSON。
import json
# 导入日志模块，用于记录 API 请求和异常。
import logging
# 导入高精度计时器，用于统计接口总耗时。
from time import perf_counter

# 导入 httpx，用于健康检查时访问 Ollama 和 Qdrant。
import httpx
# 导入 FastAPI 应用类和 HTTP 异常类型。
from fastapi import FastAPI, HTTPException
# 导入 FileResponse，用于返回前端首页文件。
from fastapi.responses import FileResponse, StreamingResponse
# 导入 StaticFiles，用于提供 JavaScript 和 CSS 静态资源。
from fastapi.staticfiles import StaticFiles
# 导入 Pydantic 模型基类和字段校验工具。
from pydantic import BaseModel, Field

# 导入两个本地服务的配置地址。
from config import OLLAMA_URL, QDRANT_URL, QUERY_REWRITE_REASONING, RETRIEVAL_MODE
# 导入内存会话的读取、保存、清除和 Prompt 格式化工具。
from chat_memory import ChatTurn, append_turn, clear_history, format_history, get_history, serialize_history
# 导入检索评估主函数。
from evaluation import evaluate
# 导入完整 RAG 问答入口。
from rag import ask, candidates_to_hits, generate, rerank_candidates, retrieve_mode_candidates, rewrite_query
# 导入统一日志初始化函数。
from logging_config import configure_logging

# 初始化项目日志格式和级别。
configure_logging()
# 创建当前 API 模块的日志记录器。
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用，并设置 Swagger 中显示的标题和版本。
app = FastAPI(title="Simple RAG", version="0.1.0")
# 定位与 main.py 同级的 frontend 文件夹。
FRONTEND_DIR = Path(__file__).with_name("frontend")
# 把 frontend 目录挂载到 /static，提供 CSS 和 JavaScript。
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# 定义 POST /ask 的请求体结构。
class AskRequest(BaseModel):
    # question 必须为 1 到 1000 个字符的字符串。
    question: str = Field(min_length=1, max_length=1000)
    # “全部”表示跨分类检索，其他值用于构造 Qdrant Metadata Filter。
    category: str = Field(default="全部", min_length=1, max_length=100)


# 定义多轮聊天请求，在问题之外携带会话标识。
class ChatRequest(AskRequest):
    # session_id 由浏览器生成，只允许安全的字母、数字、下划线和连字符。
    session_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")


# 定义单个参考来源在 API 中的返回结构。
class Source(BaseModel):
    # 检索到的知识片段正文。
    text: str
    # Qdrant 计算的向量相似度分数。
    vector_score: float
    # Cross-Encoder 计算的重排分数。
    rerank_score: float
    # 片段在来源文件中的编号。
    chunk_index: int
    # 片段的来源文件名。
    source: str
    # 当前片段所属知识分类。
    category: str = "通用"
    # 导入时生成的可读 Point 名称。
    point_name: str = "unknown"


# 定义 POST /ask 的完整响应结构。
class AskResponse(BaseModel):
    # Qwen 根据参考资料生成的最终回答。
    answer: str
    # Query Rewrite 生成并实际送入 Qdrant 的检索查询。
    rewritten_query: str
    # 实际进入 Prompt 的参考片段列表。
    sources: list[Source]


# 把一个进度事件转换成一行 JSON，前端可以边接收边解析。
def stream_line(payload: dict) -> str:
    # ensure_ascii=False 保留中文，末尾换行用于标记一条事件结束。
    return json.dumps(payload, ensure_ascii=False) + "\n"


# 将浏览器会话和知识分类组合，防止不同分类之间共享聊天上下文。
def scoped_session_id(session_id: str, category: str) -> str:
    # 使用不可出现在浏览器 session_id 中的双冒号作为分隔符。
    return f"{session_id}::{category}"


# 定义流式 RAG 执行器，每到一个真实阶段就向浏览器发送状态。
def ask_event_stream(question: str, session_id: str, category: str = "全部"):
    """按 Query Rewrite、召回、重排、生成的顺序输出 NDJSON 事件。"""
    # 从收到问题开始记录总耗时，覆盖改写、召回、重排和生成阶段。
    started = perf_counter()
    # 为当前会话和分类生成隔离的内存 Key。
    memory_id = scoped_session_id(session_id, category)
    # 读取该分类会话之前保存的问答轮次。
    history = get_history(memory_id)
    # 把结构化历史转换成两个 Prompt 都能读取的文本。
    chat_history = format_history(history)
    # 告诉页面当前有多少轮历史被用于理解问题。
    yield stream_line({"type": "status", "step": "memory", "state": "completed", "message": f"分类：{category} · 已加载 {len(history)} 轮历史对话", "history_turns": len(history), "category": category})
    # 通知前端 Query Rewrite 已开始，并告知 reasoning 配置。
    yield stream_line({"type": "status", "step": "rewrite", "state": "running", "message": "正在改写检索问题", "reasoning_enabled": QUERY_REWRITE_REASONING})
    # 调用 Qwen 完成查询改写。
    rewritten_query = rewrite_query(question, chat_history)
    # 在流式接口层再次记录问题映射，方便按一次 HTTP 请求直接查找。
    logger.info("stream query rewritten | original=%r | rewritten=%r", question, rewritten_query)
    # 返回实际用于 Qdrant 的查询。
    yield stream_line({"type": "status", "step": "rewrite", "state": "completed", "message": "查询改写完成", "detail": rewritten_query})
    # 通知前端 Qdrant 向量召回已开始。
    yield stream_line({"type": "status", "step": "retrieve", "state": "running", "message": f"正在执行 {RETRIEVAL_MODE} 检索", "retrieval_mode": RETRIEVAL_MODE})
    # 使用改写后的查询召回候选片段。
    candidates = retrieve_mode_candidates(rewritten_query, category)
    # 返回候选数量和最高 RRF 融合分数。
    yield stream_line({"type": "status", "step": "retrieve", "state": "completed", "message": f"召回 {len(candidates)} 个候选片段", "candidate_count": len(candidates), "top_score": float(candidates[0][1]) if candidates else None})
    # 没有候选时直接返回拒答，不再加载 Reranker 或调用生成模型。
    if not candidates:
        # 构造没有相关资料时的拒答文本。
        answer = "知识库中没有找到足够相关的资料。"
        # 拒答也属于完整一轮会话，保存后续问题可能需要的上下文。
        append_turn(memory_id, ChatTurn(question=question, answer=answer, rewritten_query=rewritten_query))
        # 输出无命中的最终结果。
        yield stream_line({"type": "result", "answer": answer, "rewritten_query": rewritten_query, "sources": [], "elapsed_ms": (perf_counter() - started) * 1000, "session_id": session_id, "category": category, "retrieval_mode": RETRIEVAL_MODE})
        # 结束生成器。
        return
    # hybrid_rerank 执行 Cross-Encoder，其他模式直接保留召回排名。
    if RETRIEVAL_MODE == "hybrid_rerank":
        yield stream_line({"type": "status", "step": "rerank", "state": "running", "message": "正在使用 Cross-Encoder 重排"})
        hits = rerank_candidates(question, candidates)
        yield stream_line({"type": "status", "step": "rerank", "state": "completed", "message": f"重排完成，保留 {len(hits)} 个参考片段", "reference_count": len(hits), "top_score": hits[0].rerank_score if hits else None})
    else:
        hits = candidates_to_hits(candidates)
        yield stream_line({"type": "status", "step": "rerank", "state": "completed", "message": f"{RETRIEVAL_MODE} 模式不执行 Reranker", "reference_count": len(hits), "skipped": True})
    # 通知前端最终回答生成已开始。
    yield stream_line({"type": "status", "step": "generate", "state": "running", "message": "正在让 Qwen 根据参考资料生成回答"})
    # 使用原始问题和重排后的参考资料生成答案。
    answer = generate(question, hits, chat_history)
    # 通知前端生成阶段已经结束。
    yield stream_line({"type": "status", "step": "generate", "state": "completed", "message": "回答生成完成"})
    # 将当前问答保存到所属会话，供下一轮理解指代。
    append_turn(memory_id, ChatTurn(question=question, answer=answer, rewritten_query=rewritten_query))
    # 把最终答案、改写查询和来源作为最后一个事件返回。
    yield stream_line({"type": "result", "answer": answer, "rewritten_query": rewritten_query, "sources": [hit.__dict__ for hit in hits], "elapsed_ms": (perf_counter() - started) * 1000, "session_id": session_id, "category": category, "retrieval_mode": RETRIEVAL_MODE})


# 注册首页 GET 路由，并从 Swagger 中隐藏该页面路由。
@app.get("/", include_in_schema=False)
# 定义首页处理函数。
def index() -> FileResponse:
    # 返回前端目录中的 index.html。
    return FileResponse(FRONTEND_DIR / "index.html")


# 注册健康检查 GET 路由。
@app.get("/health")
# 定义健康检查处理函数。
def health() -> dict[str, str]:
    # 使用 try 捕获 Ollama 或 Qdrant 的网络错误。
    try:
        # 请求 Ollama 版本接口，并在非 2xx 时抛出异常。
        httpx.get(f"{OLLAMA_URL}/api/version", timeout=3).raise_for_status()
        # 请求 Qdrant 健康接口，并在非 2xx 时抛出异常。
        httpx.get(f"{QDRANT_URL}/healthz", timeout=3).raise_for_status()
    # 捕获 httpx 产生的所有 HTTP 和连接异常。
    except httpx.HTTPError as exc:
        # 将服务依赖错误转换成 HTTP 503 响应。
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    # 两个依赖都正常时返回 ok。
    return {"status": "ok"}


# 注册问答 POST 路由，并声明响应模型以生成文档和校验返回值。
@app.post("/ask", response_model=AskResponse)
# 定义问答接口处理函数。
def ask_endpoint(request: AskRequest) -> AskResponse:
    # 记录接口开始时间。
    started = perf_counter()
    # 输出 API 收到请求的日志。
    logger.info("POST /ask started | question_chars=%d", len(request.question))
    # 捕获检索、重排或模型调用阶段可能产生的异常。
    try:
        # 把校验后的问题交给 RAG，并取得答案与来源。
        answer, hits, rewritten_query = ask(request.question, request.category)
    # 捕获业务流程中的未处理异常。
    except Exception as exc:
        # 输出完整异常堆栈，方便定位 Ollama、Qdrant 或代码错误。
        logger.exception("POST /ask failed | elapsed_ms=%.1f", (perf_counter() - started) * 1000)
        # 将内部错误转换成 HTTP 500，方便前台显示失败原因。
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    # 输出请求完成、命中来源数量和总耗时。
    logger.info("POST /ask completed | sources=%d | elapsed_ms=%.1f", len(hits), (perf_counter() - started) * 1000)
    # 使用 Pydantic 模型构造结构化响应。
    return AskResponse(
        # 写入模型生成的答案。
        answer=answer,
        # 返回改写结果，让前台能够对比原问题和实际检索词。
        rewritten_query=rewritten_query,
        # 把每个 SearchHit 字段转换成 Source 模型。
        sources=[Source(**hit.__dict__) for hit in hits],
    )


# 注册流式问答接口，供图形界面实时显示每个处理阶段。
@app.post("/ask/stream")
# 定义流式问答接口处理函数。
def ask_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    # 输出流式请求开始日志。
    logger.info("POST /ask/stream started | question_chars=%d", len(request.question))
    # 返回 NDJSON 响应；浏览器每收到一行就能立即刷新进度。
    return StreamingResponse(
        # 把问题交给流式 RAG 生成器。
        ask_event_stream(request.question, request.session_id, request.category),
        # 声明内容类型为逐行 JSON，并使用 UTF-8 中文。
        media_type="application/x-ndjson; charset=utf-8",
    )


# 注册会话历史读取接口，页面刷新后可以恢复已有消息。
@app.get("/chat/{session_id}")
# 定义历史读取处理函数。
def chat_history_endpoint(session_id: str, category: str = "全部") -> dict:
    # 返回会话标识、分类和按时间排序的隔离历史轮次。
    return {"session_id": session_id, "category": category, "turns": serialize_history(scoped_session_id(session_id, category))}


# 注册新建会话时调用的历史清除接口。
@app.delete("/chat/{session_id}")
# 定义历史清除处理函数。
def clear_chat_endpoint(session_id: str, category: str = "全部") -> dict[str, str]:
    # 从进程内存中删除指定会话分类的历史。
    clear_history(scoped_session_id(session_id, category))
    # 返回明确状态供前端确认。
    return {"status": "cleared", "session_id": session_id, "category": category}


# 注册知识分类列表接口，页面不需要硬编码未来新增的一级目录。
@app.get("/categories")
# 扫描 knowledge 根目录并返回可选择分类。
def categories_endpoint() -> dict[str, list[str]]:
    # 定位知识文件目录。
    knowledge_dir = Path(__file__).with_name("knowledge")
    # 根目录 TXT 属于“通用”，一级子目录名作为其他分类。
    categories = {"通用"} if any(knowledge_dir.glob("*.txt")) else set()
    # 只添加实际包含 TXT 文件的一级子目录。
    categories.update(path.name for path in knowledge_dir.iterdir() if path.is_dir() and any(path.rglob("*.txt")))
    # “全部”固定排在第一项，其余分类按名称排序。
    return {"categories": ["全部", *sorted(categories)]}


# 注册检索评估 POST 路由。
@app.post("/evaluate")
# 定义评估接口处理函数。
def evaluate_endpoint() -> dict:
    # 记录评估接口开始时间。
    started = perf_counter()
    # 输出评估开始日志。
    logger.info("POST /evaluate started")
    # 捕获评估过程中可能出现的服务或模型错误。
    try:
        # 运行 questions.json 中的全部评估问题并返回指标。
        result = evaluate()
        # 输出测试数量和总耗时。
        logger.info("POST /evaluate completed | cases=%d | elapsed_ms=%.1f", result["case_count"], (perf_counter() - started) * 1000)
        # 返回完整评估结果。
        return result
    # 捕获未处理异常。
    except Exception as exc:
        # 输出评估异常堆栈。
        logger.exception("POST /evaluate failed | elapsed_ms=%.1f", (perf_counter() - started) * 1000)
        # 将评估错误转换成 HTTP 500 响应。
        raise HTTPException(status_code=500, detail=str(exc)) from exc
