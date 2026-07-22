"""
Retrieval Metrics — known-item recall checks for the retrieval and reranking stages.

ARCHITECTURE (where this fits):
    data/eval_datasets/golden_qa.json --> RetrievalMetrics (this file) --> hit-rate report

WHY TWO SEPARATE CHECKS (retrieval vs. reranking)?
    A known fact can go missing for two different reasons:
    - Retriever.retrieve() never found it at all (embedding/BM25/hybrid-search problem)
    - Retriever found it, but Reranker.rerank() dropped it via its score threshold
      or top_n cutoff (reranker-tuning problem)
    Checking both stages independently on the SAME golden entry tells you which
    one actually broke, instead of one aggregate "retrieval is bad" signal.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from src.retrieval.retriever import Retriever, RetrievedChunk
from src.retrieval.reranker import Reranker, RankedChunk
from utils.logging import set_logger

logger = set_logger(__name__)

DEFAULT_GOLDEN_SET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "eval_datasets" / "golden_qa.json"
)


@dataclass
class KnownItemResult:
    id: str
    query_type: str
    file_id: str
    retrieval_hit: bool
    rerank_hit: bool
    missing_at_retrieval: List[str] = field(default_factory=list)
    missing_at_rerank: List[str] = field(default_factory=list)


class RetrievalMetrics:
    """
    Runs known-item recall checks from golden_qa.json against the live pipeline.

    Usage:
        retriever = await Retriever.create()
        reranker = Reranker()
        metrics = RetrievalMetrics(retriever, reranker)

        results = await metrics.run()
        summary = metrics.summarize(results)
    """

    def __init__(self, retriever: Retriever, reranker: Reranker):
        self._retriever = retriever
        self._reranker = reranker

    @staticmethod
    def load_golden_set(path: Path = DEFAULT_GOLDEN_SET_PATH) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _expected_snippets(entry: Dict[str, Any]) -> List[str]:
        """
        Normalize the golden-set schema into a flat list of snippets.

        qa entries use the singular expected_context_snippet; calculation/comparison
        entries use the plural expected_context_snippets, since those query types
        need multiple facts from potentially different pages (see ril-021/ril-022).
        """
        if "expected_context_snippets" in entry:
            return entry["expected_context_snippets"]
        snippet = entry.get("expected_context_snippet")
        return [snippet] if snippet else []

    @staticmethod
    def _snippet_found(snippet: str, chunks: List) -> bool:
        return any(snippet in chunk.document.page_content for chunk in chunks)

    async def _check_entry(self, entry: Dict[str, Any]) -> KnownItemResult:
        """Run one golden entry through retrieval and reranking, checking each stage independently."""
        snippets = self._expected_snippets(entry)

        retrieved_chunks: List[RetrievedChunk] = await self._retriever.retrieve(
            entry["query"], file_id=entry["file_id"]
        )
        missing_at_retrieval = [
            s for s in snippets if not self._snippet_found(s, retrieved_chunks)
        ]

        ranked_chunks: List[RankedChunk] = await self._reranker.arerank(
            entry["query"], retrieved_chunks
        )
        missing_at_rerank = [
            s for s in snippets if not self._snippet_found(s, ranked_chunks)
        ]

        result = KnownItemResult(
            id=entry["id"],
            query_type=entry.get("query_type", "qa"),
            file_id=entry["file_id"],
            retrieval_hit=len(missing_at_retrieval) == 0,
            rerank_hit=len(missing_at_rerank) == 0,
            missing_at_retrieval=missing_at_retrieval,
            missing_at_rerank=missing_at_rerank,
        )

        logger.info(
            "[%s] retrieval_hit=%s rerank_hit=%s",
            result.id, result.retrieval_hit, result.rerank_hit,
        )
        return result

    async def run(
        self, golden_set: Optional[List[Dict[str, Any]]] = None
    ) -> List[KnownItemResult]:
        """Run known-item recall checks across the full golden set, one entry at a time."""
        golden_set = golden_set if golden_set is not None else self.load_golden_set()
        return [await self._check_entry(entry) for entry in golden_set]

    @staticmethod
    def summarize(results: List[KnownItemResult]) -> Dict[str, Any]:
        """
        Aggregate hit rates and rerank drop-off across all checked entries.

        rerank_dropoff = retrieval_hit_rate - rerank_hit_rate
        A positive drop-off means facts were found by retrieval but lost during
        reranking (score threshold or top_n cutoff) — a reranker-tuning problem,
        isolated from retrieval quality.
        """
        total = len(results)
        if total == 0:
            return {}

        retrieval_hit_rate = sum(r.retrieval_hit for r in results) / total
        rerank_hit_rate = sum(r.rerank_hit for r in results) / total

        dropped_at_rerank = [
            r.id for r in results if r.retrieval_hit and not r.rerank_hit
        ]

        return {
            "total_entries": total,
            "retrieval_hit_rate": round(retrieval_hit_rate, 3),
            "rerank_hit_rate": round(rerank_hit_rate, 3),
            "rerank_dropoff": round(retrieval_hit_rate - rerank_hit_rate, 3),
            "dropped_at_rerank_ids": dropped_at_rerank,
        }