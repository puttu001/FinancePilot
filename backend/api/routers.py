from fastapi import APIRouter, HTTPException
from api.models import QueryRequest, QueryResponse, SourceInfo, ErrorResponse
from services.RAG_service import RAGService
from utils.logging import set_logger

logger = set_logger(__name__)

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# SERVICE SINGLETON
# ---------------------------------------------------------------------------
# WHY a module-level variable?
#   RAGService.create() is async and initializes Qdrant connection,
#   embedding models, cross-encoder, etc. We don't want to do this
#   on every request. Instead, we initialize once and reuse.
#
# WHY not initialize here directly?
#   create() is async, and module-level code can't use await.
#   So we set it to None and initialize on first request or via lifespan.
# ---------------------------------------------------------------------------
_rag_service: RAGService | None = None


async def get_rag_service() -> RAGService:
    """
    Get or create the RAG service singleton.

    First call initializes the service (Qdrant connection, models, tracing).
    Subsequent calls return the same instance.
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = await RAGService.create()
    return _rag_service


# ---------------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------------
@router.get("/health")
def health_check():
    logger.info("Health status fetched.")
    return {"FinancePilot": "A financial intelligent system", "status": "ok"}


# ---------------------------------------------------------------------------
# QUERY ENDPOINT
# ---------------------------------------------------------------------------
@router.post(
    "/query",
    response_model=QueryResponse,
    responses={500: {"model": ErrorResponse}},
)
async def query(request: QueryRequest):
    """
    Ask a question about your financial documents.

    This endpoint runs the full RAG pipeline:
    query enhancement -> hybrid search -> reranking -> context assembly -> LLM generation

    Returns the answer with source citations and performance metadata.
    """
    try:
        service = await get_rag_service()

        result = await service.query(
            query=request.query,
            file_id=request.file_id,
            user_id=request.user_id,
            sector=request.sector,
        )

        sources = [
            SourceInfo(
                file=s["file"],
                page=s["page"],
                relevance_score=s["relevance_score"],
            )
            for s in result.sources
        ]

        return QueryResponse(
            answer=result.answer,
            query_type=result.query_type,
            model=result.model,
            sources=sources,
            metadata={
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "chunks_retrieved": result.chunks_retrieved,
                "chunks_after_rerank": result.chunks_after_rerank,
                "is_valid": result.is_valid,
                "validation_errors": result.validation_errors,
            },
        )

    except Exception as e:
        logger.error("Query failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
