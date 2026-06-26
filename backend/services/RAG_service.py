"""
RAG Service -- orchestrates the full RAG pipeline.

LANGSMITH TRACING:
    Each pipeline step is a traced method so LangSmith shows:
        rag_pipeline (parent)
        |-- retrieval         (query -> chunks with scores)
        |-- reranking         (chunks -> reranked chunks with scores)
        |-- context_assembly  (chunks -> formatted prompt)
        |-- generation        (prompt -> answer)
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from src.retrieval.retriever import Retriever, RetrievedChunk
from src.retrieval.reranker import Reranker, RankedChunk
from src.augmentation.contextassembler import ContextAssembler
from src.generation.llm_client import LLMClient, GenerationResult
from utils.tracing import trace_step, init_tracing
from utils.logging import set_logger

logger = set_logger(__name__)


@dataclass
class RAGResponse:
    answer: str
    model: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    is_valid: bool = True
    validation_errors: list = field(default_factory=list)
    query_type: str = "qa"
    chunks_retrieved: int = 0
    chunks_after_rerank: int = 0
    sources: list = field(default_factory=list)


class RAGService:

    def __init__(self, retriever: Retriever, reranker: Reranker, assembler: ContextAssembler, llm_client: LLMClient):
        self._retriever = retriever
        self._reranker = reranker
        self._assembler = assembler
        self._llm_client = llm_client

    @classmethod
    async def create(cls) -> "RAGService":
        init_tracing()

        retriever = await Retriever.create()
        reranker = Reranker()
        assembler = ContextAssembler()
        llm_client = LLMClient()

        logger.info("RAG Service initialized")
        return cls(retriever, reranker, assembler, llm_client)

    # -------------------------------------------------------------------
    # TRACED PIPELINE STEPS
    # -------------------------------------------------------------------
    # Each method does the ACTUAL work AND returns a serializable dict
    # for LangSmith. The dict is what you see in the trace output panel.
    # The actual objects (chunks, messages) are stored as instance state
    # between steps via return values in query().
    # -------------------------------------------------------------------

    @trace_step(name="retrieval", run_type="retriever")
    async def _step_retrieve(
        self, query: str, file_id: Optional[str], user_id: Optional[str]
    ) -> List[RetrievedChunk]:
        chunks = await self._retriever.retrieve(query=query, file_id=file_id, user_id=user_id)
        logger.info("Retrieved %d chunks", len(chunks))
        return chunks

    @trace_step(name="reranking", run_type="chain")
    def _step_rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RankedChunk]:
        ranked = self._reranker.rerank(query, chunks)
        logger.info("Reranked: %d -> %d chunks", len(chunks), len(ranked))
        return ranked

    @trace_step(name="context_assembly", run_type="chain")
    def _step_assemble(
        self, query: str, ranked_chunks: List[RankedChunk], sector: Optional[str]
    ) -> Tuple[str, str]:
        system_msg, user_msg = self._assembler.assemble(
            query=query, chunks=ranked_chunks, sector=sector
        )
        return system_msg, user_msg

    @trace_step(name="generation", run_type="llm")
    async def _step_generate(self, system_msg: str, user_msg: str) -> GenerationResult:
        return await self._llm_client.generate(system_msg, user_msg)

    @staticmethod
    def _extract_sources(chunks: list[RankedChunk]) -> list[dict]:
        seen = set()
        sources = []
        for chunk in chunks:
            metadata = chunk.document.metadata or {}
            file_id = metadata.get("file_id", "Unknown")
            page = metadata.get("page", metadata.get("page_number", "N/A"))
            key = f"{file_id}:{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": file_id,
                    "page": page,
                    "relevance_score": round(chunk.rerank_score, 4),
                })
        return sources

    # -------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------
    @trace_step(name="rag_pipeline", run_type="chain")
    async def query(
        self,
        query: str,
        file_id: Optional[str] = None,
        user_id: Optional[str] = None,
        sector: Optional[str] = None,
    ) -> RAGResponse:
        logger.info("Processing query: '%s'", query)

        # Each step is individually traced in LangSmith
        retrieved_chunks = await self._step_retrieve(query, file_id, user_id)
        ranked_chunks = self._step_rerank(query, retrieved_chunks)
        system_msg, user_msg = self._step_assemble(query, ranked_chunks, sector)
        generation_result = await self._step_generate(system_msg, user_msg)

        sources = self._extract_sources(ranked_chunks)
        query_type, _ = self._assembler._detect_query_type(query, ranked_chunks)

        response = RAGResponse(
            answer=generation_result.answer,
            model=generation_result.model,
            latency_ms=generation_result.latency_ms,
            input_tokens=generation_result.input_tokens,
            output_tokens=generation_result.output_tokens,
            is_valid=generation_result.is_valid,
            validation_errors=generation_result.validation_errors,
            query_type=query_type,
            chunks_retrieved=len(retrieved_chunks),
            chunks_after_rerank=len(ranked_chunks),
            sources=sources,
        )

        logger.info(
            "Query complete: type=%s, chunks=%d->%d, model=%s, latency=%.0fms",
            query_type, len(retrieved_chunks), len(ranked_chunks),
            generation_result.model, generation_result.latency_ms,
        )

        return response
