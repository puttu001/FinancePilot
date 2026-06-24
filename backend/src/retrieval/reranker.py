"""
Reranker module — re-scores retrieved chunks for higher accuracy.

ARCHITECTURE (where this fits):
    QueryEnhancer → Retriever → Reranker (this file) → Augmentation

WHY RERANKING?
    The retriever uses embedding similarity (cosine + BM25 via RRF) to find
    candidate chunks. This is fast but approximate — embeddings compress a
    whole chunk into a single vector, losing fine-grained meaning.

    A cross-encoder reranker is MORE ACCURATE because:
    - It reads the FULL (query, chunk) pair together with cross-attention
    - It understands the relationship between query and chunk, not just
      their independent embeddings
    - It can catch nuances like "revenue in 2023" vs "revenue in 2022"

    The tradeoff: cross-encoders are SLOWER (can't be pre-computed), so we
    only run them on the small candidate set from the retriever (10-20 chunks),
    not the entire vector DB (thousands of chunks).

    RETRIEVER (fast, approximate) → top-k candidates
        → RERANKER (slow, precise) → top-n final results

WHAT THIS FILE DOES:
    1. Takes retrieved chunks from Retriever
    2. Re-scores each (query, chunk) pair using a cross-encoder model
    3. Filters out low-relevance chunks below a score threshold
    4. Returns the final top-n chunks sorted by reranker score
"""

import asyncio
from typing import List, Optional
from dataclasses import dataclass
from sentence_transformers import CrossEncoder

from langchain_core.documents import Document
from src.retrieval.retriever import RetrievedChunk
from utils.logging import set_logger
from utils.config import get_config

logger = set_logger(__name__)


# ---------------------------------------------------------------------------
# Data class for reranked results
# ---------------------------------------------------------------------------
# WHY a separate dataclass instead of reusing RetrievedChunk?
#   After reranking, each chunk has TWO scores:
#   - retrieval_score: the original RRF score from hybrid search
#   - rerank_score:    the cross-encoder's relevance score
#   Keeping both lets you debug and compare the two ranking stages.
#   e.g., "this chunk was ranked #8 by retriever but #1 by reranker"
# ---------------------------------------------------------------------------
@dataclass
class RankedChunk:
    document: Document           # the LangChain Document (page_content + metadata)
    rerank_score: float         # cross-encoder score (higher = more relevant)
    retrieval_score: float      # original RRF score from retriever
    source_query: str = ""      # which query produced this chunk


# ---------------------------------------------------------------------------
# Reranker class
# ---------------------------------------------------------------------------
class Reranker:
    """
    Re-scores retrieved chunks using a cross-encoder model.

    Usage:
        reranker = Reranker()
        ranked = reranker.rerank(query="What is IHCL's revenue?", chunks=retrieved_chunks)

    HOW CROSS-ENCODERS DIFFER FROM BI-ENCODERS:
        Bi-encoder (what the retriever uses):
            - Encodes query and chunk SEPARATELY into vectors
            - Compares them with cosine similarity
            - Fast: chunk vectors are pre-computed at ingestion time
            - But: loses interaction between query and chunk tokens

        Cross-encoder (what the reranker uses):
            - Feeds query AND chunk TOGETHER into the model
            - Full cross-attention between all tokens
            - Slow: must run the model for every (query, chunk) pair
            - But: much more accurate relevance scoring

        Think of it like:
            Bi-encoder  = reading a book summary to decide if it's relevant
            Cross-encoder = actually reading the book chapter to check
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the cross-encoder model.

        MODEL CHOICE:
            - cross-encoder/ms-marco-MiniLM-L-6-v2: fast, good general quality
            - BAAI/bge-reranker-v2-m3: multilingual, higher accuracy, slower
            - cross-encoder/ms-marco-MiniLM-L-12-v2: balance of speed and quality

        WHY LAZY LOADING?
            The model is loaded into memory on first use.
            CrossEncoder downloads and caches the model from HuggingFace
            on first run (~80MB for MiniLM-L-6). Subsequent loads are instant.
        """
        reranker_config = get_config("reranker", {})
        self._model_name = model_name or reranker_config.get("model")
        self._top_n = reranker_config.get("top_n", 5)
        self._score_threshold = reranker_config.get("score_threshold", 0.01)

        self._model = None

    def _load_model(self) -> CrossEncoder:
        """
        Lazy-load the cross-encoder model.

        WHY LAZY?
            Loading the model takes ~1-2 seconds and ~80-300MB of RAM.
            We only load it when rerank() is actually called, not at import time.
            This keeps app startup fast and avoids loading the model for
            code paths that don't need reranking (e.g., ingestion).
        """
        if self._model is None:
            logger.info("Loading cross-encoder model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.info("Cross-encoder model loaded successfully")
        return self._model

    # -------------------------------------------------------------------
    # CORE: Score each (query, chunk) pair
    # -------------------------------------------------------------------
    def _score_pairs(
        self, query: str, chunks: List[RetrievedChunk]
    ) -> List[RankedChunk]:
        """
        Run the cross-encoder on all (query, chunk) pairs.

        HOW CROSS-ENCODER SCORING WORKS:
            1. For each chunk, create a pair: [query, chunk_text]
            2. Feed ALL pairs to the model in one batch
               (batching is important — one-by-one would be much slower)
            3. Model outputs a relevance score for each pair
               - Higher score = more relevant to the query
               - Score range depends on model, typically -10 to +10
               - NOT a probability — it's a logit/raw score

        EXAMPLE:
            query = "What is IHCL's revenue?"
            pairs = [
                ["What is IHCL's revenue?", "IHCL reported revenue of ₹6,000 Cr..."],
                ["What is IHCL's revenue?", "The board of directors met on..."],
            ]
            scores = [8.2, -3.1]
            → First chunk is highly relevant, second is not

        WHY BATCH ALL AT ONCE?
            The model runs on GPU/CPU with fixed overhead per batch.
            Scoring 10 pairs at once is nearly as fast as scoring 1 pair.
            Individual calls would add 10x overhead from model loading/unloading.
        """
        model = self._load_model()

        pairs = [[query, chunk.document.page_content] for chunk in chunks]

        scores = model.predict(pairs)

        ranked = []
        for chunk, rerank_score in zip(chunks, scores):
            ranked.append(
                RankedChunk(
                    document=chunk.document,
                    rerank_score=float(rerank_score),
                    retrieval_score=chunk.score,
                    source_query=chunk.source_query,
                )
            )

        return ranked

    # -------------------------------------------------------------------
    # FILTER: Remove chunks below the relevance threshold
    # -------------------------------------------------------------------
    def _filter_by_threshold(
        self, chunks: List[RankedChunk], threshold: Optional[float] = None
    ) -> List[RankedChunk]:
        """
        Remove chunks with rerank scores below the threshold.

        WHY THRESHOLD FILTERING?
            Not all retrieved chunks are relevant. Even the retriever's top-10
            might include chunks that matched on keywords but aren't actually
            useful. The cross-encoder can identify these — a low rerank score
            means the model read the full text and decided it's not relevant.

            Sending irrelevant chunks to the LLM:
            - Wastes context window (tokens cost money)
            - Can MISLEAD the LLM into generating wrong answers
            - Dilutes the signal from truly relevant chunks

        THRESHOLD TUNING:
            - Too high → drops relevant chunks, LLM lacks context
            - Too low  → lets irrelevant chunks through, LLM gets confused
            - Start with 0.01, tune based on your evaluation metrics
            - Different models have different score ranges:
              ms-marco-MiniLM: scores roughly -10 to +10, threshold ~0.01
              bge-reranker: scores 0 to 1 (sigmoid), threshold ~0.3
        """
        cutoff = threshold if threshold is not None else self._score_threshold

        filtered = [c for c in chunks if c.rerank_score >= cutoff]

        dropped = len(chunks) - len(filtered)
        if dropped > 0:
            logger.info(
                "Threshold filter: kept %d, dropped %d (threshold=%.3f)",
                len(filtered), dropped, cutoff,
            )

        return filtered

    # -------------------------------------------------------------------
    # PUBLIC API: Main entry point
    # -------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RankedChunk]:
        """
        Main reranking method — scores, filters, and returns top results.

        FLOW:
            1. Score all (query, chunk) pairs using cross-encoder
            2. Sort by rerank_score descending
            3. Filter out chunks below score threshold
            4. Return top-n results

        PARAMETERS:
            query:           the original user query (not enhanced — we want
                             the reranker to judge against what the user actually asked)
            chunks:          List[RetrievedChunk] from the retriever
            top_n:           how many final results to return (default from config)
            score_threshold: minimum rerank score to keep (default from config)

        RETURNS:
            List[RankedChunk] — final chunks ready for augmentation/generation

        WHY USE ORIGINAL QUERY, NOT ENHANCED?
            The retriever uses the enhanced query ("profit" → "profit PAT net income...")
            to cast a wide net. But the reranker should judge relevance against
            what the user ACTUALLY asked. "What is the profit?" should be judged
            as-is, not with the expanded terms that were only meant to help retrieval.

        EDGE CASES:
            - Empty chunks list → returns empty list (nothing to rerank)
            - Single chunk → still runs through scoring (threshold may drop it)
            - All chunks below threshold → returns empty list (better to say
              "I don't have enough information" than to hallucinate)
        """
        n = top_n or self._top_n

        if not chunks:
            logger.warning("No chunks to rerank")
            return []

        logger.info("Reranking %d chunks for query: '%s'", len(chunks), query)

        # Step 1: Score all pairs
        ranked = self._score_pairs(query, chunks)

        # Step 2: Sort by rerank score (highest first)
        ranked.sort(key=lambda c: c.rerank_score, reverse=True)

        # Step 3: Filter by threshold
        ranked = self._filter_by_threshold(ranked, threshold=score_threshold)

        # Step 4: Take top-n
        result = ranked[:n]

        if result:
            logger.info(
                "Rerank complete: %d → %d chunks, scores: [%.3f ... %.3f]",
                len(chunks), len(result),
                result[0].rerank_score, result[-1].rerank_score,
            )
        else:
            logger.warning("All chunks filtered out after reranking")

        return result

    async def arerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[RankedChunk]:
        """
        Async wrapper around rerank() — runs the CPU-bound scoring in a thread pool.

        WHY THIS EXISTS:
            rerank() is synchronous because CrossEncoder.predict() is a CPU-bound
            PyTorch operation with no async variant. But your retriever and API
            routes are async. Calling sync rerank() directly from async code
            would BLOCK the event loop — no other requests can be served while
            the model is computing.

            asyncio.to_thread() solves this by running rerank() in a separate
            thread from Python's default ThreadPoolExecutor. The event loop
            stays free to handle other requests while the reranker computes.

        SAME PATTERN AS:
            vectordb.py line 97 — uses asyncio.to_thread() for sync LangChain calls

        WHEN TO USE WHICH:
            rerank()  → from sync code (scripts, tests, notebooks)
            arerank() → from async code (FastAPI routes, async services)
        """
        return await asyncio.to_thread(
            self.rerank, query, chunks, top_n, score_threshold
        )
