from agent_tools import list_categories


def test_agent_tools_list_existing_categories() -> None:
    """Agent 与 MCP 共用的分类工具应反映 knowledge 目录。"""
    assert {"全部", "三国演义", "水浒传", "西游记"}.issubset(list_categories())
