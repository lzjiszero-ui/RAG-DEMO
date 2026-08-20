"""Environment-based settings kept deliberately small for learning."""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).with_name(".env"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.5:4b")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "simple_rag_docs")
TOP_K = int(os.getenv("TOP_K", "3"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
