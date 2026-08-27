"""进程内多轮会话记忆；服务重启后自动清空。"""

from dataclasses import asdict, dataclass
from threading import Lock

from config import MAX_HISTORY_TURNS


@dataclass(frozen=True)
class ChatTurn:
    # 用户当时提出的原始问题。
    question: str
    # 模型根据 RAG 资料生成的回答。
    answer: str
    # 当时真正用于 Qdrant 的改写查询。
    rewritten_query: str


# 使用字典按 session_id 隔离不同浏览器会话。
_sessions: dict[str, list[ChatTurn]] = {}
# 使用互斥锁保护内存字典。
_lock = Lock()


def get_history(session_id: str) -> list[ChatTurn]:
    """返回指定会话的历史副本。"""
    # 加锁后复制列表，避免调用方直接修改内部状态。
    with _lock:
        return list(_sessions.get(session_id, []))


def append_turn(session_id: str, turn: ChatTurn) -> None:
    """保存一轮问答，并只保留最近的有限轮次。"""
    # 加锁执行读取、追加和截断。
    with _lock:
        turns = _sessions.setdefault(session_id, [])
        turns.append(turn)
        # 删除超出限制的最旧轮次，避免 Prompt 无限增长。
        del turns[:-MAX_HISTORY_TURNS]


def clear_history(session_id: str) -> None:
    """清除一个会话的全部历史。"""
    # pop 默认值让重复清除保持幂等。
    with _lock:
        _sessions.pop(session_id, None)


def format_history(turns: list[ChatTurn]) -> str:
    """把历史转换成供 Prompt 阅读的文本。"""
    # 无历史时给模型明确标记。
    if not turns:
        return "（无历史对话）"
    # 按时间顺序连接每轮用户问题和助手回答。
    return "\n\n".join(f"用户：{turn.question}\n助手：{turn.answer}" for turn in turns)


def serialize_history(session_id: str) -> list[dict[str, str]]:
    """把会话历史转换成 API 可返回的普通字典。"""
    return [asdict(turn) for turn in get_history(session_id)]
