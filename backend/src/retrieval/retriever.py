"""
Retriever module — searches the vector DB and returns relevant chunks.

ARCHITECTURE:
    QueryEnhancer (queryprocessor.py)
        → Retriever (this file)  — searches vector DB, merges multi-query results
            → Reranker (reranker.py) — re-scores and filters

WHAT THIS FILE DOES:
    1. Takes a raw or enhanced query
    2. Builds a hybrid vector store (dense + sparse) using existing core modules
    3. Searches Qdrant using similarity_search_with_score
    4. Supports metadata filtering (by file, user, etc.)
    5. Handles multi-query: when QueryEnhancer decomposes a complex query into
       sub-queries, this retriever searches each and merges + deduplicates results

"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from qdrant_client import models
from langchain_core.documents import Document

from core.database import sync_qdrant_client, get_collection_name
from core.embeddings import build_hybrid_vector_store, get_embedding_models, dense_model, sparse_model
from src.retrieval.queryprocessor import QueryEnhancer
from utils.logging import set_logger
from utils.config import get_config

logger = set_logger(__name__)


# ---------------------------------------------------------------------------
# Data class to hold a retrieved result with its score and origin query
# ---------------------------------------------------------------------------
# WHY a dataclass?
#   similarity_search_with_score returns (Document, float) tuples.
#   Wrapping them gives us a place to attach metadata like which query
#   produced this result — useful for debugging and for the reranker.
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    document: Document
    score: float
    source_query: str = ""


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------
class Retriever:
    """
    Searches the Qdrant hybrid vector store and returns relevant chunks.

    Usage:
        retriever = await Retriever.create()
        results = await retriever.retrieve("What is IHCL's revenue?")

    WHY async create() instead of __init__?
        Getting the collection name requires an async call to Qdrant.
        Python __init__ cannot be async, so we use a factory pattern.
    """

    def __init__(self, vector_store, query_enhancer: QueryEnhancer, collection_name: str):
        self._vector_store = vector_store
        self._query_enhancer = query_enhancer
        self._collection_name = collection_name

        retrieval_config = get_config("retrieval", {})
        self._default_k = retrieval_config.get("default_k", 10)
        self._multi_query_k = retrieval_config.get("multi_query_k", 5)

    @classmethod
    async def create(cls) -> "Retriever":
        """
        Factory method to create a Retriever instance.

        WHY factory pattern?
            We need to await get_collection_name() which talks to Qdrant.
            Python doesn't allow `await` inside __init__, so we use this
            classmethod that CAN be async.

        WHAT HAPPENS HERE:
            1. Get the sync Qdrant client (for LangChain compatibility)
            2. Resolve the collection name from Qdrant cloud
            3. Build the hybrid vector store (dense + sparse embeddings configured)
            4. Return a fully initialized Retriever
        """
        sync_client = sync_qdrant_client()
        collection_name = await get_collection_name()
        dense, sparse = get_embedding_models(dense_model, sparse_model)

        vector_store = build_hybrid_vector_store(
            client=sync_client,
            collection_name=collection_name,
            dense=dense,
            sparse=sparse,
        )

        query_enhancer = QueryEnhancer()
        return cls(vector_store, query_enhancer, collection_name)

    # Single query search
    async def _search(
        self,
        query: str,
        k: int = 10,
        qdrant_filter: Optional[models.Filter] = None,
    ) -> List[RetrievedChunk]:
        """
        Execute a single hybrid search against the vector store.

        HOW IT WORKS:
            1. You pass a query string
            2. QdrantVectorStore INTERNALLY embeds it using:
               - OpenAI text-embedding-3-large → dense vector (3072 dims)
               - BM25 → sparse vector
            3. Qdrant runs BOTH searches in parallel
            4. Results are fused using Reciprocal Rank Fusion (RRF):
               RRF_score(doc) = 1/(k + rank_dense) + 1/(k + rank_sparse)
            5. Top-k results returned sorted by fused score (highest first)

        PARAMETERS:
            query: the search query (plain text, not an embedding)
            k: how many results to return
            qdrant_filter: optional Qdrant filter to narrow search scope

        RETURNS:
            List of RetrievedChunk with document, score, and source query
        """
        search_kwargs = {"k": k}
        if qdrant_filter:
            search_kwargs["filter"] = qdrant_filter

        # similarity_search_with_score returns List[Tuple[Document, float]]
        # The embedding happens INSIDE this call — we never touch vectors directly
        results: List[Tuple[Document, float]] = (
            await self._vector_store.asimilarity_search_with_score(
                query, **search_kwargs
            )
        )

        return [
            RetrievedChunk(document=doc, score=score, source_query=query)
            for doc, score in results
        ]

    # -------------------------------------------------------------------
    # BUILD FILTERS: Create Qdrant filter objects from simple parameters
    # -------------------------------------------------------------------
    @staticmethod
    def build_filter(
        file_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[models.Filter]:
        """
        Build a Qdrant filter from common parameters.

        WHY FILTERING?
            Without filters, search scans ALL chunks in the collection.
            If a user uploaded 10 reports but asks about one specific company,
            we should only search that file's chunks.

        HOW QDRANT FILTERS WORK:
            - Filters are applied BEFORE vector search (pre-filtering)
            - This means they don't hurt search quality — they just narrow scope
            - Each FieldCondition matches against stored payload metadata
            - 'must' = AND logic: all conditions must match

        These field names match what ingestion stores in vectordb.py:
            metadata.file_id  → the uploaded filename
            metadata.user_id  → the user who uploaded it
        """
        conditions = []

        if file_id:
            conditions.append(
                models.FieldCondition(
                    key="metadata.file_id",
                    match=models.MatchValue(value=file_id),
                )
            )

        if user_id:
            conditions.append(
                models.FieldCondition(
                    key="metadata.user_id",
                    match=models.MatchValue(value=user_id),
                )
            )

        if not conditions:
            return None

        return models.Filter(must=conditions)

    # MULTI-QUERY RETRIEVAL: Search with decomposed sub-queries
    async def _multi_query_retrieve(
        self,
        sub_queries: List[str],
        k_per_query: int = 5,
        qdrant_filter: Optional[models.Filter] = None,
    ) -> List[RetrievedChunk]:
        """
        Search for each sub-query separately and merge results.

        WHY MULTI-QUERY?
            A single query might miss relevant chunks due to vocabulary mismatch.
            Example: "Compare revenue and profit of IHCL"
            → decomposed into: ["revenue of IHCL?", "profit of IHCL?"]
            → each sub-query retrieves chunks its phrasing matches best
            → merged results give broader coverage

        HOW DEDUPLICATION WORKS:
            Multiple sub-queries often retrieve the same chunk.
            We deduplicate by page_content, keeping the highest-scored version.
            This avoids sending duplicate context to the LLM.

        PARAMETERS:
            sub_queries: list of decomposed queries from QueryEnhancer
            k_per_query: results per sub-query (lower than single-query k
                         because we're merging multiple result sets)
            qdrant_filter: optional filter applied to all sub-queries
        """
        all_chunks: List[RetrievedChunk] = []

        for sub_query in sub_queries:
            chunks = await self._search(sub_query, k=k_per_query, qdrant_filter=qdrant_filter)
            all_chunks.extend(chunks)

        return self._deduplicate(all_chunks)

    # -------------------------------------------------------------------
    # DEDUPLICATION: Remove duplicate chunks, keep highest score
    # -------------------------------------------------------------------
    @staticmethod
    def _deduplicate(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Remove duplicate chunks, keeping the one with the highest score.

        WHY?
            When running multi-query retrieval, the same chunk can appear
            in results from different sub-queries. Sending duplicates to
            the LLM wastes context window and adds no value.

        HOW?
            We use page_content as the dedup key. If two chunks have
            identical text, we keep the one with the higher relevance score.
            Result is sorted by score descending.
        """
        seen: Dict[str, RetrievedChunk] = {}

        for chunk in chunks:
            content_key = chunk.document.page_content
            if content_key not in seen or chunk.score > seen[content_key].score:
                seen[content_key] = chunk

        deduped = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        return deduped

    # TARGETED RETRIEVAL: For calculation queries needing specific inputs
    async def _retrieve_for_calculation(
        self,
        required_inputs: List[str],
        base_query: str,
        qdrant_filter: Optional[models.Filter] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve chunks specifically targeting each required input for a calculation.

        WHY TARGETED RETRIEVAL?
            When QueryEnhancer detects a calculation request like "Calculate ROA",
            it tells us we need specific inputs: ["PAT", "Total Assets"].
            A generic search for "Calculate ROA" might not find the exact
            numbers we need. Instead, we search for each input specifically.

        EXAMPLE:
            Query: "Calculate ROA for IHCL"
            → detect_calculation_request returns: needs ["PAT", "Total Assets"]
            → We search for: "PAT Calculate ROA for IHCL", "Total Assets Calculate ROA for IHCL"
            → Each targeted search finds the exact chunk with that number
        """
        targeted_queries = [f"{input_metric} {base_query}" for input_metric in required_inputs]
        return await self._multi_query_retrieve(
            targeted_queries, k_per_query=3, qdrant_filter=qdrant_filter
        )

    # PUBLIC API: Main entry point
    async def retrieve(
        self,
        query: str,
        file_id: Optional[str] = None,
        user_id: Optional[str] = None,
        k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Main retrieval method — orchestrates the full retrieval flow.

        FLOW:
            1. Build metadata filter (if file_id or user_id provided)
            2. Enhance the query (expand financial abbreviations)
            3. Check if it's a calculation request → targeted retrieval
            4. Check if it's a multi-part question → multi-query retrieval
            5. Otherwise → single enhanced query search
            6. Return deduplicated, score-sorted chunks for the reranker

        PARAMETERS:
            query:   raw user query
            file_id: optional — restrict search to one document
            user_id: optional — restrict search to one user's documents
            k:       optional — override default number of results

        RETURNS:
            List[RetrievedChunk] sorted by score descending, ready for reranking
        """
        qdrant_filter = self.build_filter(file_id=file_id, user_id=user_id)

        # Step 1: Enhance the query with financial context
        enhanced_query = self._query_enhancer.enhance_query(query)
        logger.info("Enhanced query: '%s' → '%s'", query, enhanced_query)

        # Step 2: Check if this is a calculation request
        is_calc, metric_name, required_inputs = self._query_enhancer.detect_calculation_request(query)
        if is_calc and required_inputs:
            logger.info("Calculation detected: %s, needs: %s", metric_name, required_inputs)
            calc_chunks = await self._retrieve_for_calculation(
                required_inputs=required_inputs,
                base_query=query,
                qdrant_filter=qdrant_filter,
            )
            general_chunks = await self._search(
                enhanced_query, k=k or self._default_k, qdrant_filter=qdrant_filter
            )
            combined = calc_chunks + general_chunks
            return self._deduplicate(combined)

        # Step 3: Check if this is a multi-part/comparison query
        sub_queries = self._query_enhancer.decompose_complex_query(enhanced_query)
        if len(sub_queries) > 1:
            logger.info("Multi-query decomposition: %s", sub_queries)
            return await self._multi_query_retrieve(
                sub_queries, k_per_query=k or self._multi_query_k, qdrant_filter=qdrant_filter
            )

        # Step 4: Simple single query search
        return await self._search(enhanced_query, k=k or self._default_k, qdrant_filter=qdrant_filter)
