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


def test_evaluate_endpoint(monkeypatch) -> None:
    expected = {"case_count": 1, "summary": {}, "cases": []}
    monkeypatch.setattr(main, "evaluate", lambda: expected)

    response = client.post("/evaluate")

    assert response.status_code == 200
    assert response.json() == expected


def test_ask_endpoint_returns_rewritten_query(monkeypatch) -> None:
    monkeypatch.setattr(main, "ask", lambda question: ("answer", [], "rewritten query"))

    response = client.post("/ask", json={"question": "original question"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "answer",
        "rewritten_query": "rewritten query",
        "sources": [],
    }


def test_ask_stream_reports_real_pipeline_steps(monkeypatch) -> None:
    candidate = (Document(page_content="context", metadata={"source": "guide.txt", "chunk_index": 0}), 0.8)
    hit = SearchHit(text="context", vector_score=0.8, rerank_score=0.9, chunk_index=0, source="guide.txt")
    monkeypatch.setattr(main, "rewrite_query", lambda question, history: "rewritten query")
    monkeypatch.setattr(main, "retrieve_candidates", lambda question: [candidate])
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
    monkeypatch.setattr(main, "retrieve_candidates", lambda question: [candidate])
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
