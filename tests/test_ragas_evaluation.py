import ragas_evaluation


class FakeMetric:
    def __init__(self, score):
        self.score = score

    def single_turn_score(self, sample):
        assert sample.user_input == "谁醉打蒋门神？"
        assert sample.reference == "武松"
        assert sample.retrieved_contexts == ["武松醉打蒋门神"]
        return self.score


def test_ragas_evaluation_summarizes_answer_quality(monkeypatch) -> None:
    monkeypatch.setattr(ragas_evaluation, "RAGAS_MAX_CASES", 4)
    monkeypatch.setattr(ragas_evaluation, "ragas_components", lambda: {
        "faithfulness": FakeMetric(1.0),
        "answer_relevancy": FakeMetric(0.8),
        "context_precision": FakeMetric(0.9),
        "context_recall": FakeMetric(0.7),
    })
    retrieval_result = {"cases": [{
        "question": "谁醉打蒋门神？",
        "generation": {
            "answer": "武松。",
            "expected_answer": "武松",
            "contexts": ["武松醉打蒋门神"],
        },
    }]}

    result = ragas_evaluation.evaluate_answer_quality(retrieval_result)

    assert result["framework"] == "RAGAS"
    assert result["summary"]["faithfulness"] == 1.0
    assert result["summary"]["context_recall"] == 0.7
