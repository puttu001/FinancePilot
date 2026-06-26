"""
LangSmith Tracing -- instruments the RAG pipeline for monitoring and debugging.

WHAT IS LANGSMITH?
    LangSmith is an observability platform for LLM applications. It records
    every step of your pipeline (retrieval, reranking, generation) so you can:
    - See exactly what happened for each query (full trace)
    - Debug why a specific answer was wrong
    - Monitor latency, token usage, and costs over time
    - Compare experiments (e.g., different prompts, models, chunk sizes)
    - Build evaluation datasets from real user queries

HOW TRACING WORKS IN LANGCHAIN:
    LangChain has AUTOMATIC tracing for its components. When you set these
    environment variables, ALL LangChain calls get traced with zero code changes:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=lsv2_pt_xxxxx
        LANGCHAIN_PROJECT=FinancePilot

    This means your ChatOpenAI and QdrantVectorStore calls are already traced.
    This module adds CUSTOM traces on top for your own pipeline stages
    (query enhancement, reranking, context assembly) that LangChain doesn't
    know about.

TWO LEVELS OF TRACING:
    1. Automatic (just env vars) - LLM calls, vector store searches
       -> Already works with your existing code, no changes needed

    2. Custom (this module) - your pipeline stages as named spans
       -> Adds visibility into retrieval timing, rerank scores, etc.
       -> Uses @traceable decorator or langsmith.trace() context manager

WHAT YOU'LL SEE IN LANGSMITH DASHBOARD:
    RAG Pipeline (parent trace)
    |-- Query Enhancement     (input query -> enhanced query)
    |-- Retrieval             (query -> chunks with scores)
    |-- Reranking             (chunks -> reranked chunks with scores)
    |-- Context Assembly      (chunks -> formatted prompt)
    |-- LLM Generation        (prompt -> answer)  [auto-traced by LangChain]
"""

import os
from typing import Optional
from functools import wraps
from dotenv import load_dotenv

from utils.logging import set_logger
from utils.config import get_config

logger = set_logger(__name__)


def init_tracing():
    """
    Initialize LangSmith tracing by setting environment variables.

    WHY SET ENV VARS IN CODE?
        LangChain checks environment variables to decide whether to trace.
        Your .env file has the API key, but LANGCHAIN_TRACING_V2 and
        LANGCHAIN_PROJECT need to be set too. This function ensures
        everything is configured correctly on app startup.

    WHEN TO CALL:
        Call once at application startup (e.g., in main.py or FastAPI lifespan).
        After this, all LangChain calls are automatically traced.

    RETURNS:
        True if tracing is enabled and configured, False otherwise.
    """
    load_dotenv()

    langsmith_config = get_config("langsmith", {})
    tracing_enabled = langsmith_config.get("tracing_enabled", False)
    project_name = langsmith_config.get("project_name", "FinancePilot")

    if not tracing_enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing is DISABLED (set tracing_enabled: true in config.yaml)")
        return False

    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.warning(
            "LangSmith tracing enabled in config but LANGCHAIN_API_KEY not found in .env. "
            "Tracing will not work. Add LANGCHAIN_API_KEY=lsv2_pt_xxxxx to your .env file."
        )
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name

    logger.info("LangSmith tracing ENABLED for project: %s", project_name)
    return True


def get_langsmith_client():
    """
    Get a LangSmith client instance for advanced operations.

    WHEN YOU NEED THIS:
        - Creating evaluation datasets from traced runs
        - Fetching run results programmatically
        - Managing projects/datasets via API

    FOR BASIC TRACING:
        You DON'T need this. The @traceable decorator and LangChain's
        auto-tracing work via environment variables alone.
    """
    try:
        from langsmith import Client
        return Client()
    except Exception as e:
        logger.warning("Failed to create LangSmith client: %s", str(e))
        return None


# ---------------------------------------------------------------------------
# TRACEABLE DECORATOR -- wraps your functions as named spans in LangSmith
# ---------------------------------------------------------------------------
# WHY?
#   LangChain auto-traces its own components (ChatOpenAI, VectorStore).
#   But your custom code (QueryEnhancer, Reranker, ContextAssembler) is
#   invisible to LangSmith. This decorator makes them visible as named
#   spans in the trace, so you see the full pipeline breakdown.
#
# HOW IT WORKS:
#   Uses langsmith's @traceable decorator when tracing is enabled.
#   When disabled, returns the original function unchanged (no overhead).
#
# USAGE:
#   @trace_step(name="query_enhancement", run_type="chain")
#   def enhance_query(query: str) -> str:
#       ...
#
# RUN TYPES:
#   "chain"     -> generic pipeline step (query enhancement, context assembly)
#   "retriever" -> search/retrieval operations
#   "llm"       -> LLM calls (auto-handled by LangChain, rarely needed manually)
#   "tool"      -> tool/function calls
# ---------------------------------------------------------------------------
def trace_step(name: str, run_type: str = "chain", metadata: Optional[dict] = None):
    """
    Decorator to trace a function as a named span in LangSmith.

    PARAMETERS:
        name:     display name in the LangSmith trace tree
        run_type: categorizes the span ("chain", "retriever", "llm", "tool")
        metadata: optional static metadata attached to every run of this function

    EXAMPLE:
        @trace_step(name="reranking", run_type="chain")
        def rerank(query, chunks):
            ...

        In LangSmith, this shows up as:
            RAG Pipeline
            |-- reranking (inputs: query, chunks -> output: ranked_chunks)
    """
    langsmith_config = get_config("langsmith", {})
    tracing_enabled = langsmith_config.get("tracing_enabled", False)

    def decorator(func):
        if not tracing_enabled:
            return func

        try:
            from langsmith import traceable

            @traceable(name=name, run_type=run_type, metadata=metadata or {})
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            @traceable(name=name, run_type=run_type, metadata=metadata or {})
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        except ImportError:
            logger.warning("langsmith not installed, tracing disabled for %s", name)
            return func

    return decorator


# ---------------------------------------------------------------------------
# TRACE METADATA HELPERS -- attach runtime info to traces
# ---------------------------------------------------------------------------
def build_trace_metadata(
    query: str,
    file_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **extra,
) -> dict:
    """
    Build metadata dict to attach to a trace run.

    WHY METADATA?
        Metadata makes traces searchable and filterable in LangSmith.
        You can filter by file_id to see all queries about a specific document,
        or by user_id to see a specific user's query history.

    WHAT GETS ATTACHED:
        - query: the original user question
        - file_id: which document was queried
        - user_id: who asked
        - any extra key-value pairs you pass

    HOW TO USE:
        metadata = build_trace_metadata(query="What is ROA?", file_id="ihcl.pdf")
        # Pass to langsmith.trace() or as run metadata
    """
    meta = {"query": query}
    if file_id:
        meta["file_id"] = file_id
    if user_id:
        meta["user_id"] = user_id
    meta.update(extra)
    return meta


def build_retrieval_feedback(
    retrieved_count: int,
    reranked_count: int,
    top_retrieval_score: float = 0.0,
    top_rerank_score: float = 0.0,
) -> dict:
    """
    Build a feedback dict summarizing retrieval quality for a trace.

    WHY?
        Over time, you can aggregate this to answer questions like:
        - "What % of queries return fewer than 3 chunks after reranking?"
        - "What's the average top retrieval score?"
        - "How much does reranking change the ranking?"

    This data feeds into your evaluation module later.
    """
    return {
        "retrieved_count": retrieved_count,
        "reranked_count": reranked_count,
        "chunks_filtered": retrieved_count - reranked_count,
        "top_retrieval_score": round(top_retrieval_score, 4),
        "top_rerank_score": round(top_rerank_score, 4),
    }
