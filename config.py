"""集中读取项目配置，方便通过 .env 调整参数。"""

# 导入 Python 标准库，用于读取系统环境变量。
import os
# 导入 Path，用于定位与当前文件同目录的 .env 文件。
from pathlib import Path

# 导入 dotenv 加载器，让普通 Python 命令也能读取 .env。
from dotenv import load_dotenv

# 找到 config.py 同目录下的 .env，并把其中的键值加载到环境变量。
load_dotenv(Path(__file__).with_name(".env"))

# 读取项目日志级别；常用值包括 DEBUG、INFO、WARNING 和 ERROR。
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# 读取 Ollama 服务地址；未配置时使用本机默认地址。
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
# 读取 Qdrant 服务地址；未配置时使用本机默认地址。
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
# 读取 Embedding 模型名称，用于文档和问题向量化。
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
# 读取聊天模型名称，用于根据参考资料生成答案。
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:4b")
# 读取 Query Rewrite 是否开启推理模式，并兼容 true、1、yes、on 写法。
QUERY_REWRITE_REASONING = os.getenv("QUERY_REWRITE_REASONING", "true").lower() in {"true", "1", "yes", "on"}
# 读取每个会话最多保留的完整问答轮数。
MAX_HISTORY_TURNS = max(1, int(os.getenv("MAX_HISTORY_TURNS", "6")))
# 读取 Qdrant Collection 名称，用于隔离本项目的向量数据。
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "simple_rag_docs")
# 读取重排后最终保留的片段数量，并转换为整数。
TOP_K = int(os.getenv("TOP_K", "3"))
# 读取 Qdrant 第一阶段最多召回的候选数量，并转换为整数。
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))
# 读取 Qdrant 最低向量相似度，低于该值的片段会被过滤。
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
# 读取指定分类时使用的宽松阈值；Metadata 已限制范围，因此可允许更多候选进入 Reranker。
CATEGORY_SCORE_THRESHOLD = float(os.getenv("CATEGORY_SCORE_THRESHOLD", "0.3"))
# 读取 Cross-Encoder Reranker 的模型名称。
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
# 读取每个文本切片允许包含的最大字符数。
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
# 读取相邻文本切片期望保留的重叠字符数。
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
