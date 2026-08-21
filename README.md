# Simple RAG 学习项目

这是一个使用 LangChain 组件实现核心流程的 RAG 示例：

```text
knowledge/*.txt（多个文件）
  → RecursiveCharacterTextSplitter 分块
  → OllamaEmbeddings（bge-m3）生成 Embedding
  → QdrantVectorStore 保存向量

用户问题
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
```

修改分块参数或知识文件后，需要重新执行导入命令。查询先按 `SCORE_THRESHOLD` 从 Qdrant 召回最多 `RETRIEVAL_K` 条，再由 Reranker 重新评分并保留 `TOP_K` 条。首次查询会下载约 1 GB 的 Reranker 模型到 `.models/`。

## 3. 启动 API

```powershell
uv run uvicorn main:app --reload --port 8001
```

打开 Swagger：<http://127.0.0.1:8001/docs>

先访问 `GET /health`，再调用 `POST /ask`：

```json
{
  "question": "这个项目怎样查找相关文档？"
}
```

响应中会同时显示最终答案、来源文件、Qdrant 的 `vector_score` 和重排后的 `rerank_score`。低于 `SCORE_THRESHOLD` 的片段不会进入重排阶段。

## 4. VS Code Debug

项目已包含 `.vscode/launch.json`。选择 `.venv/Scripts/python.exe` 解释器，然后在“运行和调试”中选择 `Debug Simple RAG API`，按 F5。

推荐断点顺序：

1. `main.py` 的 `ask_endpoint`
2. `rag.py` 的 `ask`
3. `rag.py` 的 `retrieve`
4. `rag.py` 的 `vector_store`
5. `rag.py` 的 `generate`

## 5. 测试

```powershell
uv run pytest -q
```
