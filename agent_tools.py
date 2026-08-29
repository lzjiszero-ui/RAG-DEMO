"""Agent 与 MCP 共用的知识库工具，保证两种调用方式行为一致。"""

from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from config import RETRIEVAL_MODE
from rag import SearchHit, candidates_to_hits, rerank_candidates, retrieve_mode_candidates


def list_categories() -> list[str]:
    """扫描 knowledge 目录，返回当前可检索的知识分类。"""
    knowledge_dir = Path(__file__).with_name("knowledge")
    categories = {"通用"} if any(knowledge_dir.glob("*.txt")) else set()
    categories.update(path.name for path in knowledge_dir.iterdir() if path.is_dir() and any(path.rglob("*.txt")))
    return ["全部", *sorted(categories)]


def search_knowledge(query: str, category: str = "全部") -> tuple[list[SearchHit], dict[str, Any]]:
    """执行项目当前配置的检索管线，并返回模型可读结果与结构化命中。"""
    allowed_categories = list_categories()
    selected_category = category if category in allowed_categories else "全部"
    candidates = retrieve_mode_candidates(query, selected_category)
    hits = (
        rerank_candidates(query, candidates)
        if RETRIEVAL_MODE == "hybrid_rerank"
        else candidates_to_hits(candidates)
    )
    payload = {
        "query": query,
        "category": selected_category,
        "retrieval_mode": RETRIEVAL_MODE,
        "count": len(hits),
        "results": [
            {
                "reference": index,
                "text": hit.text,
                "source": hit.source,
                "category": hit.category,
                "point_name": hit.point_name,
                "score": hit.rerank_score if RETRIEVAL_MODE == "hybrid_rerank" else hit.vector_score,
            }
            for index, hit in enumerate(hits, start=1)
        ],
    }
    return hits, payload


@tool
def list_knowledge_categories() -> list[str]:
    """列出本地知识库目前有哪些可用分类。用户询问可用资料范围时调用。"""
    return list_categories()


@tool
def search_local_knowledge(query: str, category: str = "全部") -> dict[str, Any]:
    """搜索本地 Qdrant 知识库。回答小说情节或项目知识问题前必须调用；category 可填全部、三国演义、水浒传或西游记。"""
    _, payload = search_knowledge(query, category)
    return payload


AGENT_TOOLS = [list_knowledge_categories, search_local_knowledge]
AGENT_TOOL_MAP = {item.name: item for item in AGENT_TOOLS}
