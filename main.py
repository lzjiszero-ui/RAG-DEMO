"""FastAPI entry point for the simple RAG learning project."""

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import OLLAMA_URL, QDRANT_URL
from rag import ask

app = FastAPI(title="Simple RAG", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Source(BaseModel):
    text: str
    vector_score: float
    rerank_score: float
    chunk_index: int
    source: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict[str, str]:
    try:
        httpx.get(f"{OLLAMA_URL}/api/version", timeout=3).raise_for_status()
        httpx.get(f"{QDRANT_URL}/healthz", timeout=3).raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    try:
        answer, hits = ask(request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(
        answer=answer,
        sources=[Source(**hit.__dict__) for hit in hits],
    )
