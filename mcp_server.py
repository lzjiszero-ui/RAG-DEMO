"""把本地 RAG 检索能力通过标准 MCP 协议暴露给 Claude Desktop、Codex 等客户端。"""

from typing import Any

from mcp.server.fastmcp import FastMCP

from agent_tools import list_categories, search_knowledge

mcp = FastMCP(
    "simple-rag",
    instructions="用于检索本项目 Qdrant 本地知识库。回答资料问题前先调用 search_knowledge。",
)


@mcp.tool()
def list_knowledge_categories() -> list[str]:
    """列出本地知识库可检索的分类。"""
    return list_categories()


@mcp.tool()
def search_local_knowledge(query: str, category: str = "全部") -> dict[str, Any]:
    """通过项目当前配置的 Dense/BM25/Hybrid/Reranker 管线检索本地知识库。"""
    _, payload = search_knowledge(query, category)
    return payload


if __name__ == "__main__":
    mcp.run(transport="stdio")
