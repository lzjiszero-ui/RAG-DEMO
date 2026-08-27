# Simple RAG 学习项目

这是一个使用 LangChain 组件实现核心流程的 RAG 示例：

```text
knowledge/*.txt（多个文件）
  → RecursiveCharacterTextSplitter 分块
  → OllamaEmbeddings（bge-m3）生成 Embedding
  → QdrantVectorStore 保存向量

用户问题
  → 根据 session_id 加载最近的多轮对话
  → ChatOllama（qwen3.5:4b）执行 Query Rewrite
  → QdrantVectorStore 召回候选 Document
  → bge-reranker-base 对候选结果重新排序
  → ChatPromptTemplate 注入参考资料
  → ChatOllama（qwen3.5:4b）根据资料回答
```

## 已有环境

- Ollama：`http://127.0.0.1:11434`
- Qdrant：`http://127.0.0.1:6333`
- 模型：`bge-m3`、`qwen3.5:4b`

如果 Qdrant 尚未运行，可执行：

```powershell
docker compose up -d
```

本机已有其他 Qdrant 占用 6333 时，不要重复执行此命令，直接复用现有服务即可。

## 1. 安装依赖

```powershell
uv sync
```

## 2. 导入知识库

```powershell
uv run python ingest.py
```

把 `.txt` 文件放入 `knowledge/` 目录（支持子目录），然后执行导入。这一步会重建 Qdrant 中的 `simple_rag_docs` Collection，但不会影响其他项目的 Collection。

文本分块参数可在 `.env` 中调整：

```dotenv
CHUNK_SIZE=500
CHUNK_OVERLAP=80
SCORE_THRESHOLD=0.5
RETRIEVAL_K=10
TOP_K=3
RERANKER_MODEL=BAAI/bge-reranker-base
LOG_LEVEL=INFO
QUERY_REWRITE_REASONING=true
```

`LOG_LEVEL=INFO` 会输出 Query Rewrite、向量召回、重排、生成和总请求耗时等关键日志。调试时可改为 `DEBUG`，只关注错误时可改为 `ERROR`。日志不会输出 Embedding 向量或完整知识片段。

`QUERY_REWRITE_REASONING=true` 会让 Qwen 在改写查询时启用推理模式；设为 `false` 可降低改写耗时。图形界面通过 `/ask/stream` 接收真实进度事件，会依次显示 Query Rewrite、Qdrant Retrieval、Cross-Encoder 和 Qwen Generation 的执行状态。

`MAX_HISTORY_TURNS=6` 表示每个 `session_id` 最多保留最近 6 轮问答。历史会同时提供给 Query Rewrite 和最终生成 Prompt，因此“它是什么”“上一点再解释一下”这类追问可以结合上下文理解。当前实现使用进程内存保存历史，重启 FastAPI 后会清空；生产环境可进一步替换为 Redis 或数据库。

修改分块参数或知识文件后，需要重新执行导入命令。查询先按 `SCORE_THRESHOLD` 从 Qdrant 召回最多 `RETRIEVAL_K` 条，再由 Reranker 重新评分并保留 `TOP_K` 条。首次查询会下载约 1 GB 的 Reranker 模型到 `.models/`。

## 3. 启动 API

```powershell
uv run uvicorn main:app --reload --port 8001
```

打开图形界面：<http://127.0.0.1:8001/>

打开 Swagger：<http://127.0.0.1:8001/docs>

先访问 `GET /health`，再调用 `POST /ask`：

```json
{
  "question": "这个项目怎样查找相关文档？"
}
```

响应中会同时显示 Query Rewrite 结果、最终答案、来源文件、Qdrant 的 `vector_score` 和重排后的 `rerank_score`。改写查询只用于 Qdrant 召回；Reranker 和最终回答继续使用用户原始问题。改写失败时自动回退到原问题，低于 `SCORE_THRESHOLD` 的片段不会进入重排阶段。

## 4. 检索评估

图形界面的“检索评估”页会读取 `eval/questions.json`，对比三条管线：原问题直接检索 Qdrant、原问题召回后使用 Cross-Encoder 重排、先用 Qwen Query Rewrite 再重新召回并重排。第三条会逐题调用 Qwen，但不会执行最终答案生成。

也可以从命令行执行：

```powershell
uv run python evaluation.py
```

指标含义：

- `Hit@1`：正确来源是否排在第 1 名。
- `Hit@3`：正确来源是否出现在前 3 名。
- `MRR`：正确来源排名倒数的平均值，越接近 1 越好。
- `AVG TIME`：每个问题的平均检索耗时；Reranker 耗时包含 Qdrant 召回阶段。

要增加评估问题，请在 `eval/questions.json` 中添加 `question` 和 `expected_source`。

## 5. VS Code Debug

项目已包含 `.vscode/launch.json`。选择 `.venv/Scripts/python.exe` 解释器，然后在“运行和调试”中选择 `Debug Simple RAG API`，按 F5。

推荐断点顺序：

1. `main.py` 的 `ask_endpoint`
2. `rag.py` 的 `ask`
3. `rag.py` 的 `retrieve`
4. `rag.py` 的 `vector_store`
5. `rag.py` 的 `generate`

## 6. 测试

```powershell
uv run pytest -q
```
