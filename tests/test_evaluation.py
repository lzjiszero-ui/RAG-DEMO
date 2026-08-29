from langchain_core.documents import Document

import evaluation
from evaluation import EvaluationCase, evaluate, ranking_metrics, source_rank, summarize, summarize_bm25_impact
from rag import SearchHit


def test_source_rank_uses_first_matching_source() -> None:
    ranking = [
        {"source": "other.txt"},
        {"source": "expected.txt"},
        {"source": "expected.txt"},
    ]

    assert source_rank(ranking, "expected.txt") == 2
    assert source_rank(ranking, "missing.txt") is None


def test_ranking_metrics() -> None:
    assert ranking_metrics(1) == {
        "hit_at_1": 1.0,
        "hit_at_3": 1.0,
        "hit_at_5": 1.0,
        "reciprocal_rank": 1.0,
        "ndcg_at_5": 1.0,
    }
    assert ranking_metrics(2)["reciprocal_rank"] == 0.5
    assert ranking_metrics(None) == {
        "hit_at_1": 0.0,
        "hit_at_3": 0.0,
        "hit_at_5": 0.0,
        "reciprocal_rank": 0.0,
        "ndcg_at_5": 0.0,
    }


def test_summarize_averages_metrics_and_latency() -> None:
    results = [
        {
            "qdrant": {
                "hit_at_1": 1.0,
                "hit_at_3": 1.0,
                "hit_at_5": 1.0,
                "reciprocal_rank": 1.0,
                "ndcg_at_5": 1.0,
                "latency_ms": 10.0,
            }
        },
        {
            "qdrant": {
                "hit_at_1": 0.0,
                "hit_at_3": 1.0,
                "hit_at_5": 1.0,
                "reciprocal_rank": 0.5,
                "ndcg_at_5": 1.0 / __import__("math").log2(3),
                "latency_ms": 20.0,
            }
        },
    ]

    assert summarize(results, "qdrant") == {
        "hit_at_1": 0.5,
        "hit_at_3": 1.0,
        "hit_at_5": 1.0,
        "mrr": 0.75,
        "ndcg_at_5": (1.0 + 1.0 / __import__("math").log2(3)) / 2,
        "avg_latency_ms": 15.0,
    }


def test_summarize_bm25_impact_compares_dense_and_hybrid() -> None:
    results = [{
        "qdrant": {"rank": 2, "hit_at_1": 0.0, "hit_at_3": 1.0, "hit_at_5": 1.0, "reciprocal_rank": 0.5, "ndcg_at_5": 0.63, "latency_ms": 10.0},
        "hybrid": {"rank": 1, "hit_at_1": 1.0, "hit_at_3": 1.0, "hit_at_5": 1.0, "reciprocal_rank": 1.0, "ndcg_at_5": 1.0, "latency_ms": 14.0},
    }]

    impact = summarize_bm25_impact(results)

    assert impact["hit_at_1_delta"] == 1.0
    assert impact["mrr_delta"] == 0.5
    assert impact["latency_delta_ms"] == 4.0
    assert impact["improved_cases"] == 1


def test_evaluate_compares_original_rerank_and_rewritten_pipeline(monkeypatch) -> None:
    queries = []

    def fake_retrieve(query, category="全部"):
        queries.append(query)
        source = "wrong.txt" if query == "ambiguous" else "expected.txt"
        return [(Document(page_content="context", metadata={"source": source, "chunk_index": 0}), 0.8)]

    def fake_rerank(question, candidates, limit=None):
        document, vector_score = candidates[0]
        return [SearchHit(text=document.page_content, vector_score=vector_score, rerank_score=0.9, chunk_index=0, source=document.metadata["source"])]

    monkeypatch.setattr(evaluation, "retrieve_dense_candidates", fake_retrieve)
    monkeypatch.setattr(evaluation, "retrieve_candidates", fake_retrieve)
    monkeypatch.setattr(evaluation, "rerank_candidates", fake_rerank)
    monkeypatch.setattr(evaluation, "rewrite_query", lambda question: "precise rewritten query")
    monkeypatch.setattr(evaluation, "generate", lambda question, hits: "expected answer [Reference 1]")

    result = evaluate([EvaluationCase(question="ambiguous", expected_source="expected.txt")])

    assert queries == ["ambiguous", "ambiguous", "precise rewritten query"]
    assert result["cases"][0]["qdrant"]["rank"] is None
    assert result["cases"][0]["reranker"]["rank"] is None
    assert result["cases"][0]["rewrite_reranker"]["rank"] == 1
    assert result["summary"]["rewrite_reranker"]["hit_at_1"] == 1.0
