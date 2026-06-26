"""
API request/response models -- defines the shape of data in and out.

WHY PYDANTIC MODELS?
    - Automatic request validation (wrong type? missing field? -> 422 error)
    - Auto-generated API docs (Swagger UI at /docs)
    - Type safety -- your IDE catches errors before runtime
    - Serialization -- converts dataclasses/dicts to JSON automatically
"""

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# REQUEST MODELS -- what the client sends
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    """
    Request body for the /api/query endpoint.

    EXAMPLE REQUEST:
        POST /api/query
        {
            "query": "What is IHCL's total revenue?",
            "file_id": "ihcl.pdf",
            "sector": "Real Estate & Hospitality"
        }
    """
    query: str = Field(
        ...,
        min_length=3,
        description="The user's question about financial documents",
        examples=["What is the total revenue?"],
    )
    file_id: Optional[str] = Field(
        default=None,
        description="Filter search to a specific document. Must match the filename used during ingestion",
        examples=["ihcl.pdf"],
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Filter search to a specific user's documents",
    )
    sector: Optional[str] = Field(
        default=None,
        description="Inject sector-specific financial knowledge (e.g., Banking & NBFC, IT & Technology)",
        examples=["Banking & NBFC", "Real Estate & Hospitality"],
    )


# ---------------------------------------------------------------------------
# RESPONSE MODELS -- what the client receives
# ---------------------------------------------------------------------------
class SourceInfo(BaseModel):
    """A single source citation from the retrieved chunks."""
    file: str
    page: str | int
    relevance_score: float


class QueryResponse(BaseModel):
    """
    Response body for the /api/query endpoint.

    EXAMPLE RESPONSE:
        {
            "answer": "IHCL reported total revenue of ...",
            "query_type": "qa",
            "model": "gpt-4o-mini",
            "sources": [{"file": "ihcl.pdf", "page": 45, "relevance_score": 0.92}],
            "metadata": {
                "latency_ms": 2340.5,
                "input_tokens": 890,
                "output_tokens": 245,
                "chunks_retrieved": 10,
                "chunks_after_rerank": 5,
                "is_valid": true
            }
        }

    WHY NESTED METADATA?
        The client primarily cares about `answer` and `sources`.
        Performance/debug info (latency, tokens, validation) is secondary,
        so it's nested under `metadata` to keep the top level clean.
    """
    answer: str
    query_type: str
    model: str
    sources: list[SourceInfo]
    metadata: dict


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
