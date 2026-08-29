"""基于 Qwen Tool Calling 的轻量 Agent 执行循环。"""

import json
import logging
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from agent_tools import AGENT_TOOL_MAP, AGENT_TOOLS, search_knowledge
from config import AGENT_MAX_STEPS, AGENT_REASONING, CHAT_MODEL, OLLAMA_URL
from rag import SearchHit

logger = logging.getLogger(__name__)


def _message_text(message: AIMessage) -> str:
    """兼容字符串和内容块两种 AIMessage.content 格式。"""
    if isinstance(message.content, str):
        return message.content.strip()
    return "\n".join(str(block.get("text", "")) if isinstance(block, dict) else str(block) for block in message.content).strip()


def run_agent(question: str, category: str, chat_history: str) -> Iterator[dict[str, Any]]:
    """让模型自主选择工具，并逐步产出供页面显示的 Agent 事件。"""
    model = ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
        reasoning=AGENT_REASONING,
    ).bind_tools(AGENT_TOOLS)
    messages = [
        SystemMessage(content=(
            "你是本地知识库 Agent。回答知识问题前必须调用 search_local_knowledge，"
            "不知道分类时使用用户当前选择的分类；只有用户询问知识库范围时才调用 list_knowledge_categories。"
            "必须只根据工具返回资料回答，并用 [Reference N] 标明依据；没有结果就明确说不知道。"
            f"\n页面当前分类：{category}\n历史对话：\n{chat_history}"
        )),
        HumanMessage(content=question),
    ]
    collected_hits: list[SearchHit] = []
    tool_trace: list[dict[str, Any]] = []

    for step in range(1, AGENT_MAX_STEPS + 1):
        yield {"type": "status", "step": "agent", "state": "running", "message": f"Agent 正在进行第 {step} 轮判断"}
        response = model.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            answer = _message_text(response) or "Agent 没有生成最终回答。"
            yield {"type": "status", "step": "agent", "state": "completed", "message": "Agent 已根据工具结果生成回答"}
            yield {"type": "agent_result", "answer": answer, "hits": collected_hits, "tool_trace": tool_trace}
            return

        for call in response.tool_calls:
            name = call["name"]
            arguments = call.get("args", {})
            yield {"type": "status", "step": "tool", "state": "running", "message": f"调用工具：{name}", "tool_name": name, "tool_args": arguments}
            logger.info("agent tool calling | tool=%s | args=%s", name, arguments)
            selected_tool = AGENT_TOOL_MAP.get(name)
            if selected_tool is None:
                output: Any = {"error": f"未知工具：{name}"}
            elif name == "search_local_knowledge":
                collected_hits, output = search_knowledge(
                    str(arguments.get("query", question)),
                    str(arguments.get("category", category)),
                )
            else:
                output = selected_tool.invoke(arguments)
            tool_trace.append({"name": name, "args": arguments, "result_count": output.get("count") if isinstance(output, dict) else None})
            messages.append(ToolMessage(content=json.dumps(output, ensure_ascii=False), tool_call_id=call["id"]))
            yield {"type": "status", "step": "tool", "state": "completed", "message": f"工具 {name} 执行完成", "tool_name": name, "result_count": output.get("count") if isinstance(output, dict) else None}

    yield {
        "type": "agent_result",
        "answer": f"Agent 已达到最大工具调用轮数（{AGENT_MAX_STEPS}），未能生成最终回答。",
        "hits": collected_hits,
        "tool_trace": tool_trace,
    }
