from fastapi.testclient import TestClient
from langchain_core.documents import Document

import main
from rag import SearchHit


client = TestClient(main.app)


def test_frontend_is_served() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "RAG Lab" in response.text
    assert client.get("/static/app.js").status_code == 200


def test_removed_non_streaming_ask_endpoint_is_not_registered() -> None:
    """工程只保留 /ask/stream，旧的非流式 /ask 应返回 404。"""
    assert client.post("/ask", json={"question": "test"}).status_code == 404


def test_evaluate_endpoint(monkeypatch) -> None:
    expected = {"case_count": 1, "summary": {}, "cases": []}
    monkeypatch.setattr(main, "evaluate", lambda: expected)

    response = client.post("/evaluate")

    assert response.status_code == 200
    assert response.json() == expected


def test_ask_stream_reports_real_pipeline_steps(monkeypatch) -> None:
    candidate = (Document(page_content="context", metadata={"source": "guide.txt", "chunk_index": 0}), 0.8)
    hit = SearchHit(text="context", vector_score=0.8, rerank_score=0.9, chunk_index=0, source="guide.txt")
    monkeypatch.setattr(main, "rewrite_query", lambda question, history: "rewritten query")
    monkeypatch.setattr(main, "retrieve_mode_candidates", lambda question, category: [candidate])
    monkeypatch.setattr(main, "rerank_candidates", lambda question, candidates: [hit])
    monkeypatch.setattr(main, "generate", lambda question, hits, history: "streamed answer")

    response = client.post("/ask/stream", json={"question": "original question", "session_id": "test-session"})
    events = [main.json.loads(line) for line in response.text.splitlines()]

    assert response.status_code == 200
    assert [(event.get("step"), event.get("state")) for event in events[:-1]] == [
        ("memory", "completed"),
        ("rewrite", "running"),
        ("rewrite", "completed"),
        ("retrieve", "running"),
        ("retrieve", "completed"),
        ("rerank", "running"),
        ("rerank", "completed"),
        ("generate", "running"),
        ("generate", "completed"),
    ]
    assert events[2]["detail"] == "rewritten query"
    assert events[-1]["type"] == "result"
    assert events[-1]["answer"] == "streamed answer"
    assert events[-1]["elapsed_ms"] >= 0


def test_second_turn_receives_first_turn_history(monkeypatch) -> None:
    seen_histories = []
    candidate = (Document(page_content="context", metadata={"source": "guide.txt", "chunk_index": 0}), 0.8)
    hit = SearchHit(text="context", vector_score=0.8, rerank_score=0.9, chunk_index=0, source="guide.txt")

    def fake_rewrite(question, history):
        seen_histories.append(history)
        return question

    monkeypatch.setattr(main, "rewrite_query", fake_rewrite)
    monkeypatch.setattr(main, "retrieve_mode_candidates", lambda question, category: [candidate])
    monkeypatch.setattr(main, "rerank_candidates", lambda question, candidates: [hit])
    monkeypatch.setattr(main, "generate", lambda question, hits, history: f"answer for {question}")

    client.delete("/chat/multi-turn-test")
    client.post("/ask/stream", json={"question": "第一个问题", "session_id": "multi-turn-test"})
    client.post("/ask/stream", json={"question": "它呢？", "session_id": "multi-turn-test"})

    assert seen_histories[0] == "（无历史对话）"
    assert "用户：第一个问题" in seen_histories[1]
    assert "助手：answer for 第一个问题" in seen_histories[1]

    history_response = client.get("/chat/multi-turn-test")
    assert len(history_response.json()["turns"]) == 2

    clear_response = client.delete("/chat/multi-turn-test")
    assert clear_response.json()["status"] == "cleared"
    assert client.get("/chat/multi-turn-test").json()["turns"] == []


def test_categories_endpoint_lists_knowledge_folders() -> None:
    response = client.get("/categories")

    assert response.status_code == 200
    assert {"全部", "三国演义", "水浒传", "西游记"}.issubset(response.json()["categories"])


def test_chat_history_is_isolated_by_category(monkeypatch) -> None:
    candidate = (Document(page_content="context", metadata={"source": "book.txt", "category": "水浒传", "point_name": "水浒传-1", "chunk_index": 0}), 0.8)
    hit = SearchHit(text="context", vector_score=0.8, rerank_score=0.9, chunk_index=0, source="book.txt", category="水浒传", point_name="水浒传-1")
    monkeypatch.setattr(main, "rewrite_query", lambda question, history: question)
    monkeypatch.setattr(main, "retrieve_mode_candidates", lambda question, category: [candidate])
    monkeypatch.setattr(main, "rerank_candidates", lambda question, candidates: [hit])
    monkeypatch.setattr(main, "generate", lambda question, hits, history: "水浒传答案")

    client.delete("/chat/category-test?category=水浒传")
    client.delete("/chat/category-test?category=西游记")
    client.post("/ask/stream", json={"question": "人物是谁", "session_id": "category-test", "category": "水浒传"})

    assert len(client.get("/chat/category-test?category=水浒传").json()["turns"]) == 1
    assert client.get("/chat/category-test?category=西游记").json()["turns"] == []


def test_agent_stream_returns_tool_trace(monkeypatch) -> None:
    """Agent 模式应透传工具轨迹、来源并保存最终回答。"""
    hit = SearchHit(text="武松醉打蒋门神", vector_score=0.8, rerank_score=0.9, chunk_index=0, source="水浒传.txt")

    def fake_agent(question, history):
        yield {"type": "status", "step": "tool", "state": "running", "message": "调用工具"}
        yield {"type": "agent_result", "answer": "武松 [Reference 1]", "hits": [hit], "tool_trace": [{"name": "search_local_knowledge", "args": {"query": question, "category": "水浒传"}, "result_count": 1, "selected_category": "水浒传"}]}

    monkeypatch.setattr(main, "run_agent", fake_agent)
    client.delete("/chat/agent-test?category=水浒传")
    response = client.post("/ask/stream", json={"question": "谁打了蒋门神", "session_id": "agent-test", "category": "水浒传", "mode": "agent"})
    events = [main.json.loads(line) for line in response.text.splitlines()]

    assert response.status_code == 200
    assert events[-1]["answer"] == "武松 [Reference 1]"
    assert events[-1]["tool_trace"][0]["name"] == "search_local_knowledge"
    assert events[-1]["agent_selected_category"] == "水浒传"
    assert events[-1]["category"] == "水浒传"
    assert events[-1]["retrieval_mode"].startswith("agent/")
