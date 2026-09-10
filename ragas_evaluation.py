"""使用本地 Ollama 模型运行 RAGAS 回答质量评估。"""

# 导入缓存装饰器，避免每次评估重复创建模型适配器。
from functools import lru_cache
# 导入日志和计时工具。
import logging
from time import perf_counter

# 导入 LangChain 的 Ollama 聊天与向量组件。
from langchain_ollama import ChatOllama, OllamaEmbeddings
# 导入 RAGAS 单轮样本、模型适配器和四个 RAG 指标。
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

# 导入本地模型和评估数量配置。
from config import CHAT_MODEL, EMBEDDING_MODEL, OLLAMA_URL, RAGAS_MAX_CASES

# 创建当前模块日志记录器。
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def ragas_components() -> dict:
    """创建并缓存使用 Ollama 的 RAGAS 指标对象。"""
    # RAGAS 的 LLM-as-a-Judge 使用零温度 Qwen，减少多次评估的随机差异。
    evaluator_llm = LangchainLLMWrapper(ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_URL, temperature=0, reasoning=False))
    # Answer Relevancy 需要 Embedding 计算原问题与反向生成问题的相似度。
    evaluator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL))
    return {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        "context_precision": LLMContextPrecisionWithReference(llm=evaluator_llm),
        "context_recall": LLMContextRecall(llm=evaluator_llm),
    }


def evaluate_answer_quality(retrieval_evaluation: dict) -> dict:
    """对现有检索评估生成结果运行 RAGAS，并返回逐题及平均分。"""
    started = perf_counter()
    metrics = ragas_components()
    case_results = []
    # 限制题数，因为四项指标中的每项都可能进行一次或多次本地模型调用。
    selected_cases = retrieval_evaluation.get("cases", [])[:RAGAS_MAX_CASES]
    for item in selected_cases:
        generation = item.get("generation", {})
        sample = SingleTurnSample(
            user_input=item["question"],
            response=generation.get("answer", ""),
            retrieved_contexts=generation.get("contexts", []),
            reference=generation.get("expected_answer", ""),
        )
        scores = {}
        errors = {}
        for name, metric in metrics.items():
            try:
                # RAGAS 的同步单样本 API 返回 0 到 1 的质量分数。
                scores[name] = float(metric.single_turn_score(sample))
            except Exception as exc:
                logger.exception("RAGAS metric failed | metric=%s | question=%s", name, item["question"])
                scores[name] = None
                errors[name] = str(exc)
        case_results.append({"question": item["question"], "scores": scores, "errors": errors})
    # 每项平均值忽略失败的单项；全部失败时返回 null 而不是伪造 0 分。
    summary = {}
    for name in metrics:
        values = [case["scores"][name] for case in case_results if case["scores"][name] is not None]
        summary[name] = sum(values) / len(values) if values else None
    return {
        "framework": "RAGAS",
        "case_count": len(case_results),
        "configured_case_limit": RAGAS_MAX_CASES,
        "summary": summary,
        "cases": case_results,
        "elapsed_ms": (perf_counter() - started) * 1000,
    }
