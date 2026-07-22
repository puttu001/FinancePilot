"""
Evaluator — orchestrates the offline/batch evaluation loop against the golden dataset.

ARCHITECTURE (where this fits):
    golden_qa.json --> RetrievalMetrics / LLMJudge (future) --> Evaluator (this file) --> report

WHAT THIS FILE DOES:
    Ties the individual metric modules together into one entry point that runs
    the golden set and produces a report. This is the offline/batch half of
    evaluation - see confidence_score.py for the separate real-time, per-request
    check that runs on live traffic instead of a fixed golden set.
"""

from typing import Any, Dict

from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.evaluation.retrieval_metrics import RetrievalMetrics
from utils.logging import set_logger

logger = set_logger(__name__)


class Evaluator:
    """
    Runs the offline evaluation suite against the golden dataset.

    Usage:
        evaluator = await Evaluator.create()
        report = await evaluator.run_retrieval_eval()
    """

    def __init__(self, retriever: Retriever, reranker: Reranker):
        self._retriever = retriever
        self._reranker = reranker
        self._retrieval_metrics = RetrievalMetrics(retriever, reranker)

    @classmethod
    async def create(cls) -> "Evaluator":
        retriever = await Retriever.create()
        reranker = Reranker()
        logger.info("Evaluator initialized")
        return cls(retriever, reranker)

    async def run_retrieval_eval(self) -> Dict[str, Any]:
        """
        Run the known-item recall check (retrieval + reranking stages) across
        the full golden set and return a summary + per-entry report.

        NOTE: generation metrics (Faithfulness, Answer Relevancy, Answer
        Correctness via RAGAS) and the real-time confidence gate are not
        wired in yet - see llm_judge.py and confidence_score.py.
        """
        results = await self._retrieval_metrics.run()
        summary = self._retrieval_metrics.summarize(results)

        return {
            "summary": summary,
            "per_entry": [vars(r) for r in results],
        }