# Simple RAG 学习项目

这是一个使用 LangChain 组件实现核心流程的 RAG 示例：

```text
knowledge/*.txt 与 knowledge/分类名/*.txt（多个文件）
  → RecursiveCharacterTextSplitter 分块
  → OllamaEmbeddings（bge-m3）生成 Dense Embedding
  → FastEmbed BM25 生成 Sparse Embedding
  → QdrantVectorStore 同时保存 Dense / Sparse 向量

用户问题
  → 根据 session_id 加载最近的多轮对话
  → ChatOllama（qwen3.5:4b）执行 Query Rewrite
  → Qdrant 使用 Dense + BM25 + RRF Hybrid Search 召回候选 Document
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

本地实际配置写在 `.env`，该文件已被 Git 忽略。新增环境变量时，必须同时在 `.env.sample` 中添加同名配置键，但等号后保持为空，不要提交本地地址、密码、Token 或其他实际值。

## 2. 导入知识库

```powershell
uv run python ingest.py
```

把 `.txt` 文件放入 `knowledge/` 目录，然后执行导入。这一步会重建 Qdrant 中的 `simple_rag_docs` Collection，但不会影响其他项目的 Collection。根目录文件归为“通用”，一级子目录名会成为分类，例如：

```text
knowledge/
├─ rag_basics.txt          → 通用
├─ 三国演义/三国演义.txt   → 三国演义
├─ 水浒传/水浒传.txt       → 水浒传
└─ 西游记/西游记.txt       → 西游记
```

每个 Qdrant Point 的 Payload 会保存 `metadata.category`、`metadata.source`、`metadata.point_name` 和 `metadata.chunk_index`。页面选择具体分类后，后端会用 `metadata.category` Filter 限制向量检索；选择“全部”时不添加 Filter。不同分类的多轮聊天上下文也会相互隔离。

文本分块参数可在 `.env` 中调整：

```dotenv
CHUNK_SIZE=500
CHUNK_OVERLAP=80
SCORE_THRESHOLD=0.3
RETRIEVAL_MODE=hybrid_rerank
CONTEXTUAL_RETRIEVAL=true
CONTEXTUAL_MAX_DOCUMENT_CHARS=12000
RETRIEVAL_K=10
TOP_K=3
RERANKER_MODEL=BAAI/bge-reranker-base
LOG_LEVEL=INFO
QUERY_REWRITE_REASONING=true
```

`RETRIEVAL_MODE` 决定在线问答使用的检索管线：

| 配置值 | 实际流程 | 适用场景 |
| --- | --- | --- |
| `vector` | 纯 Dense 向量检索 | 语义相似召回优先 |
| `bm25` | 纯 BM25 关键词检索 | 编号、字段名和精确词命中 |
| `hybrid` | Dense + BM25，经 RRF 融合 | 平衡语义与关键词召回 |
| `hybrid_rerank` | Hybrid 后执行 Cross-Encoder | 更高精度上限，耗时也最高 |

修改 `RETRIEVAL_MODE` 后只需重启 FastAPI，不需要重新执行 `ingest.py`；但 Collection 必须已经包含 Dense 和 Sparse 两种向量。

BM25 写入和查询前会对连续中文生成单字与二元词，例如“蒋门神”会补充“蒋门”“门神”，避免默认空格分词导致纯 `bm25` 模式无法命中中文。修改这段分词规则后需要重新执行 `ingest.py`。

`CONTEXTUAL_RETRIEVAL=true` 会在导入阶段调用本地 Qwen，为每个切片生成不超过 80 字的文档级上下文，再把“来源 + 上下文 + 原始片段”一起生成 Dense 和 Sparse 向量。原文、上下文和是否成功增强会分别保存在 `metadata.original_text`、`metadata.contextual_summary`、`metadata.contextualized`。`CONTEXTUAL_MAX_DOCUMENT_CHARS` 限制提供给模型的完整文档长度。关闭该功能或修改 Prompt 后，需要重新执行 `ingest.py` 才会影响 Collection。

`LOG_LEVEL=INFO` 会输出 Query Rewrite、向量召回、重排、生成和总请求耗时等关键日志。调试时可改为 `DEBUG`，只关注错误时可改为 `ERROR`。日志不会输出 Embedding 向量或完整知识片段。

`QUERY_REWRITE_REASONING=true` 会让 Qwen 在改写查询时启用推理模式；设为 `false` 可降低改写耗时。图形界面通过 `/ask/stream` 接收真实进度事件，会依次显示 Query Rewrite、Qdrant Retrieval、Cross-Encoder 和 Qwen Generation 的执行状态。

`MAX_HISTORY_TURNS=6` 表示每个 `session_id` 最多保留最近 6 轮问答。历史会同时提供给 Query Rewrite 和最终生成 Prompt，因此“它是什么”“上一点再解释一下”这类追问可以结合上下文理解。当前实现使用进程内存保存历史，重启 FastAPI 后会清空；生产环境可进一步替换为 Redis 或数据库。

修改分块参数或知识文件后，需要重新执行导入命令。纯向量模式统一使用 `SCORE_THRESHOLD=0.3`，不再区分“全部”和具体分类阈值；Qdrant 最多召回 `RETRIEVAL_K` 条，`hybrid_rerank` 模式再由 Reranker 重新评分并保留 `TOP_K` 条。首次重排会加载约 1 GB 的 Reranker 模型到 `.models/`。

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

响应中会同时显示 Query Rewrite 结果、最终答案、来源文件、Qdrant 的 `vector_score` 和重排后的 `rerank_score`。改写查询只用于 Qdrant 召回；Reranker 和最终回答继续使用用户原始问题。改写失败时自动回退到原问题，低于当前检索模式所用阈值的片段不会进入重排阶段。

## 4. 检索评估

图形界面的“检索评估”页会读取 `eval/questions.json`，对比四条管线：Dense 向量召回、Dense + BM25 的 RRF 融合召回、Hybrid + Cross-Encoder、Query Rewrite + Hybrid + Cross-Encoder。评估还会执行最终回答生成，用 `expected_answer` 检查关键答案命中率和引用率。

“加入 BM25 带来的变化”区域会直接计算 Hybrid 相对 Dense 的 Hit@1、Hit@3、MRR、nDCG@5 和平均耗时差值，并统计逐题排名提升、持平与下降的数量；逐题表也会单独显示 Dense → Hybrid 的排名变化。

也可以从命令行执行：

```powershell
uv run python evaluation.py
```

指标含义：

- `Hit@1`：正确来源是否排在第 1 名。
- `Hit@3`：正确来源是否出现在前 3 名。
- `Hit@5`：正确来源是否出现在前 5 名。
- `MRR`：正确来源排名倒数的平均值，越接近 1 越好。
- `nDCG@5`：前五名排序质量，正确来源越靠前分数越高。
- `AVG TIME`：每个问题的平均检索耗时；Reranker 耗时包含 Qdrant 召回阶段。
- `ANSWER MATCH`：最终回答包含 `expected_answer` 的问题比例。
- `CITATION RATE`：最终回答带有 Reference 引用标记的问题比例。
- `END-TO-END`：Query Rewrite、Hybrid Search、Reranker 和回答生成的平均总耗时。

要增加评估问题，请在 `eval/questions.json` 中添加 `question`、`expected_source`、`category` 和 `expected_answer`。

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

## 7. Agent、Tool Calling 与 MCP

页面“运行方式”选择 **Agent + Tools** 后，Qwen 不再执行写死的 RAG 顺序，而是先读取工具描述，自主生成 Tool Call。当前提供两个工具：

- `search_local_knowledge`：按当前检索模式查询 Qdrant，并在需要时执行 Reranker。
- `list_knowledge_categories`：列出可以检索的知识分类。

后端把工具结果作为 `ToolMessage` 返回给 Qwen，直到模型输出最终答案或达到 `AGENT_MAX_STEPS`。页面会显示工具名、参数与结果数量。`AGENT_REASONING` 控制 Agent 判断阶段是否启用 Qwen reasoning。

同一组能力也通过 MCP Server 暴露。手动启动 stdio MCP Server：

```powershell
uv run python mcp_server.py
```

外部 MCP 客户端应配置命令 `uv`，参数为 `--directory`、本工程绝对路径、`run`、`python`、`mcp_server.py`。MCP 本身不提供网页；它是一套让 Claude Desktop、Codex 等 AI 客户端发现并调用本工程工具的标准协议。页面 Agent 使用的是 LangChain Tool Calling，MCP 客户端使用的是 MCP，但二者最终复用 `agent_tools.py` 中完全相同的检索函数。
