"""对比原始 Qdrant、Cross-Encoder 和 Query Rewrite 完整管线。"""

# 导入 json，用于读取评估问题和输出命令行结果。
import json
# 导入 dataclass，用不可变数据类表示单条评估问题。
from dataclasses import dataclass
# 导入 Path，用于定位 eval/questions.json。
from pathlib import Path
# 导入 log2，用于计算按排名位置折扣的 nDCG@5。
from math import log2
# 导入高精度计时器，用于测量检索和重排耗时。
from time import perf_counter
# 导入 Any，用于描述包含不同字段类型的结果字典。
from typing import Any

# 导入 Document 类型，用于声明 Qdrant 候选的数据结构。
from langchain_core.documents import Document

# 导入检索结果类型、改写、重排和 Qdrant 候选召回函数。
from rag import SearchHit, generate, rerank_candidates, retrieve_candidates, retrieve_dense_candidates, rewrite_query


# 定位项目默认评估问题文件。
DEFAULT_CASES_PATH = Path(__file__).with_name("eval") / "questions.json"


# 将评估问题定义成不可变数据类，防止运行中意外修改标准答案。
@dataclass(frozen=True)
# 定义一条评估用例的数据结构。
class EvaluationCase:
    # 要发送给检索系统的问题。
    question: str
    # 预期应当命中的知识来源文件。
    expected_source: str
    # 可选知识分类，用于同时评估 Metadata Filter。
    category: str = "全部"
    # 最终回答中应出现的关键答案，用于轻量、可重复的生成正确性评估。
    expected_answer: str = ""


# 定义评估问题加载函数，并允许调用方传入其他 JSON 文件。
def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[EvaluationCase]:
    # 读取 UTF-8 JSON 文本并解析成 Python 列表。
    data = json.loads(path.read_text(encoding="utf-8"))
    # 把每个字典转换成 EvaluationCase。
    return [EvaluationCase(**item) for item in data]


# 定义正确来源排名计算函数。
def source_rank(ranking: list[dict[str, Any]], expected_source: str) -> int | None:
    """返回正确来源第一次出现的排名；未命中时返回 None。"""
    # 使用 next 取得第一个满足条件的排名，没有匹配时使用 None。
    return next(
        # 生成所有 source 等于 expected_source 的一基排名。
        (
            # 返回从 1 开始的排名索引。
            index
            # enumerate 为排名列表中的每一项附加排名。
            for index, item in enumerate(ranking, start=1)
            # 只保留来源文件与标准答案一致的项。
            if item["source"] == expected_source
        ),
        # 生成器没有结果时返回 None。
        None,
    )


# 根据一个正确来源排名计算单题指标。
def ranking_metrics(rank: int | None) -> dict[str, float]:
    # 返回 Hit@1、Hit@3 和倒数排名。
    return {
        # 只有正确来源排第 1 时记为 1，否则为 0。
        "hit_at_1": float(rank == 1),
        # 正确来源出现在前三名时记为 1，否则为 0。
        "hit_at_3": float(rank is not None and rank <= 3),
        # Hit@5 衡量正确来源是否进入更宽的候选集合。
        "hit_at_5": float(rank is not None and rank <= 5),
        # 未命中记为 0；命中时使用 1/rank。
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        # 单一相关来源时，nDCG@5 按排名位置给予对数折扣。
        "ndcg_at_5": 0.0 if rank is None or rank > 5 else 1.0 / log2(rank + 1),
    }


# 将 Qdrant 原始候选转换成可序列化的排名列表。
def qdrant_ranking(candidates: list[tuple[Document, float]]) -> list[dict[str, Any]]:
    # 返回保持 Qdrant 原始顺序的结果字典列表。
    return [
        # 为每个候选创建一个前台可展示的字典。
        {
            # 从 metadata 获取来源文件。
            "source": str(document.metadata.get("source", "unknown")),
            # 从 metadata 获取切片编号。
            "chunk_index": int(document.metadata.get("chunk_index", 0)),
            # 保存 Qdrant 的向量分数。
            "vector_score": float(vector_score),
        }
        # 逐个解包 Document 和 vector_score。
        for document, vector_score in candidates
    ]


# 将重排后的 SearchHit 转换成可序列化的排名列表。
def reranker_ranking(hits: list[SearchHit]) -> list[dict[str, Any]]:
    # 返回保持 Reranker 新顺序的结果字典列表。
    return [
        # 为每个重排结果创建一个前台可展示的字典。
        {
            # 保存来源文件。
            "source": hit.source,
            # 保存切片编号。
            "chunk_index": hit.chunk_index,
            # 同时保留原始向量分数，便于对照。
            "vector_score": hit.vector_score,
            # 保存 Cross-Encoder 重排分数。
            "rerank_score": hit.rerank_score,
        }
        # 遍历重排后的全部 SearchHit。
        for hit in hits
    ]


# 汇总所有问题在某一种检索方法下的平均指标。
def summarize(results: list[dict[str, Any]], method: str) -> dict[str, float]:
    # 取得评估问题总数。
    count = len(results)
    # 空问题集无法计算平均值，因此返回全零指标。
    if count == 0:
        # 返回与正常摘要相同的字段结构。
        return {"hit_at_1": 0.0, "hit_at_3": 0.0, "hit_at_5": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0, "avg_latency_ms": 0.0}
    # 计算各项指标的算术平均值。
    return {
        # 汇总每题 Hit@1 后除以问题总数。
        "hit_at_1": sum(item[method]["hit_at_1"] for item in results) / count,
        # 汇总每题 Hit@3 后除以问题总数。
        "hit_at_3": sum(item[method]["hit_at_3"] for item in results) / count,
        "hit_at_5": sum(item[method]["hit_at_5"] for item in results) / count,
        # MRR 是每题 reciprocal_rank 的平均值。
        "mrr": sum(item[method]["reciprocal_rank"] for item in results) / count,
        "ndcg_at_5": sum(item[method]["ndcg_at_5"] for item in results) / count,
        # 计算该方法处理单个问题的平均毫秒耗时。
        "avg_latency_ms": sum(item[method]["latency_ms"] for item in results) / count,
    }


# 定义完整评估入口；不传 cases 时读取默认 JSON。
def evaluate(cases: list[EvaluationCase] | None = None) -> dict[str, Any]:
    """运行全部问题，返回逐题排名和汇总指标。"""
    # 优先使用调用方传入的用例，否则加载默认问题集。
    selected_cases = load_cases() if cases is None else cases
    # 创建空列表，用于保存每个问题的详细评估结果。
    results: list[dict[str, Any]] = []

    # 逐个运行评估问题。
    for case in selected_cases:
        # 记录 Dense 基线检索开始时间。
        qdrant_started = perf_counter()
        # 只使用 bge-m3 Dense 向量执行基线召回。
        dense_candidates = retrieve_dense_candidates(case.question, case.category)
        # 计算 Qdrant 阶段耗时并转换成毫秒。
        qdrant_latency_ms = (perf_counter() - qdrant_started) * 1000
        # 把原始候选转换成前台需要的排名结构。
        raw_ranking = qdrant_ranking(dense_candidates)

        # 运行 Dense + BM25 + RRF Hybrid Search。
        hybrid_started = perf_counter()
        candidates = retrieve_candidates(case.question, case.category)
        hybrid_latency_ms = (perf_counter() - hybrid_started) * 1000
        hybrid_ranking = qdrant_ranking(candidates)

        # 记录 Reranker 阶段开始时间。
        rerank_started = perf_counter()
        # 对全部候选重排；limit=None 保留完整排名用于 MRR。
        reranked_hits = rerank_candidates(case.question, candidates, limit=None)
        # 计算纯 Reranker 阶段耗时并转换成毫秒。
        rerank_only_ms = (perf_counter() - rerank_started) * 1000
        # 把重排结果转换成前台需要的排名结构。
        reranked_ranking = reranker_ranking(reranked_hits)

        # 记录 Query Rewrite 完整管线开始时间。
        rewrite_pipeline_started = perf_counter()
        # 把原问题改写成更适合向量检索的独立查询。
        rewritten_query = rewrite_query(case.question)
        # 使用改写查询重新从 Qdrant 召回一组候选。
        rewritten_candidates = retrieve_candidates(rewritten_query, case.category)
        # 使用原始问题对改写查询召回的候选进行 Cross-Encoder 重排。
        rewritten_hits = rerank_candidates(case.question, rewritten_candidates, limit=None)
        # 统计 Query Rewrite、Qdrant 和 Reranker 三个阶段的总耗时。
        rewrite_pipeline_latency_ms = (perf_counter() - rewrite_pipeline_started) * 1000
        # 把第三条管线结果转换成可序列化排名。
        rewritten_ranking = reranker_ranking(rewritten_hits)

        # 使用完整管线的前三个结果生成最终回答，评估端到端 RAG。
        generation_started = perf_counter()
        generated_answer = generate(case.question, rewritten_hits[:3]) if rewritten_hits else ""
        generation_latency_ms = (perf_counter() - generation_started) * 1000
        # 关键答案匹配是无需额外裁判模型的确定性正确性指标。
        answer_match = float(bool(case.expected_answer) and case.expected_answer in generated_answer)
        # 引用覆盖检查回答是否按照 Prompt 输出了至少一个 Reference 标记。
        citation_present = float("[Reference" in generated_answer)

        # 查找正确来源在 Qdrant 原始结果中的排名。
        qdrant_rank = source_rank(raw_ranking, case.expected_source)
        # 查找正确来源在 Hybrid Search 中的排名。
        hybrid_rank = source_rank(hybrid_ranking, case.expected_source)
        # 查找正确来源在 Reranker 结果中的新排名。
        reranker_rank = source_rank(reranked_ranking, case.expected_source)
        # 查找正确来源在 Query Rewrite 完整管线中的排名。
        rewrite_reranker_rank = source_rank(rewritten_ranking, case.expected_source)
        # 保存当前问题的标准答案、三种排名、指标和耗时。
        results.append(
            # 结果字典会直接由 FastAPI 序列化成 JSON。
            {
                # 保存评估问题。
                "question": case.question,
                # 保存正确来源，供前台对照。
                "expected_source": case.expected_source,
                # 保存评估使用的知识分类。
                "category": case.category,
                # 保存 Qdrant 原始检索的详细结果。
                "qdrant": {
                    # 正确来源的原始排名，未命中时为 None。
                    "rank": qdrant_rank,
                    # 仅包含向量检索阶段的耗时。
                    "latency_ms": qdrant_latency_ms,
                    # 保存完整原始排名。
                    "ranking": raw_ranking,
                    # 展开当前排名对应的 Hit@1、Hit@3 和倒数排名。
                    **ranking_metrics(qdrant_rank),
                },
                # 保存未经 Reranker 的 Hybrid Search 结果。
                "hybrid": {
                    "rank": hybrid_rank,
                    "latency_ms": hybrid_latency_ms,
                    "ranking": hybrid_ranking,
                    **ranking_metrics(hybrid_rank),
                },
                # 保存加入 Cross-Encoder 后的详细结果。
                "reranker": {
                    # 正确来源在重排后的排名。
                    "rank": reranker_rank,
                    # 完整重排管线耗时等于 Qdrant 耗时加纯重排耗时。
                    "latency_ms": hybrid_latency_ms + rerank_only_ms,
                    # 单独保留纯 Reranker 耗时，便于性能分析。
                    "rerank_only_ms": rerank_only_ms,
                    # 保存完整重排结果。
                    "ranking": reranked_ranking,
                    # 展开重排后排名对应的指标。
                    **ranking_metrics(reranker_rank),
                },
                # 保存 Query Rewrite、Qdrant 和 Cross-Encoder 完整管线结果。
                "rewrite_reranker": {
                    # 保存 Qwen 实际生成的改写查询，方便人工判断质量。
                    "rewritten_query": rewritten_query,
                    # 保存正确来源在第三条管线中的排名。
                    "rank": rewrite_reranker_rank,
                    # 耗时包含改写、重新向量召回和重排。
                    "latency_ms": rewrite_pipeline_latency_ms,
                    # 保存第三条管线的完整排名结果。
                    "ranking": rewritten_ranking,
                    # 展开 Hit@1、Hit@3 和倒数排名。
                    **ranking_metrics(rewrite_reranker_rank),
                },
                # 保存最终生成质量与端到端耗时。
                "generation": {
                    "expected_answer": case.expected_answer,
                    "answer": generated_answer,
                    "answer_match": answer_match,
                    "citation_present": citation_present,
                    "generation_latency_ms": generation_latency_ms,
                    "end_to_end_latency_ms": rewrite_pipeline_latency_ms + generation_latency_ms,
                },
            }
        )

    # 返回问题数量、三种方法的摘要以及所有逐题结果。
    # 汇总最终回答的关键答案命中率、引用率和端到端耗时。
    generation_summary = {
        "answer_match_rate": sum(item["generation"]["answer_match"] for item in results) / len(results) if results else 0.0,
        "citation_rate": sum(item["generation"]["citation_present"] for item in results) / len(results) if results else 0.0,
        "avg_end_to_end_latency_ms": sum(item["generation"]["end_to_end_latency_ms"] for item in results) / len(results) if results else 0.0,
    }
    return {
        # 保存参与评估的问题总数。
        "case_count": len(selected_cases),
        # 分别汇总三条管线的平均表现。
        "summary": {
            # 计算 Qdrant 原始排名的汇总指标。
            "qdrant": summarize(results, "qdrant"),
            # 计算 Dense + BM25 融合召回的汇总指标。
            "hybrid": summarize(results, "hybrid"),
            # 计算完整重排管线的汇总指标。
            "reranker": summarize(results, "reranker"),
            # 计算 Query Rewrite + Qdrant + Reranker 的汇总指标。
            "rewrite_reranker": summarize(results, "rewrite_reranker"),
            # 汇总最终回答的确定性质量代理指标。
            "generation": generation_summary,
        },
        # 保存前台逐题表格所需的详细结果。
        "cases": results,
    }


# 定义命令行入口函数。
def main() -> None:
    # 执行评估，并用可读的 UTF-8 JSON 打印完整结果。
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))


# 只有直接运行 evaluation.py 时才进入命令行入口。
if __name__ == "__main__":
    # 启动评估流程。
    main()
