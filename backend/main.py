from fastapi import FastAPI
from contextlib import asynccontextmanager
from api.routers import router, get_rag_service
from utils.logging import set_logger
from utils.config import load_config
from utils.tracing import init_tracing

logger = set_logger(__name__)
config = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on app startup and shutdown.

    Startup:
        - Initialize LangSmith tracing (sets env vars before any LangChain call)
        - Pre-initialize RAG service (Qdrant connection, models loaded)
        This way the first user request doesn't pay the initialization cost.

    Shutdown:
        - Cleanup if needed (currently nothing to clean up)
    """
    init_tracing()
    await get_rag_service()
    logger.info("FinancePilot API ready")
    yield
    logger.info("FinancePilot API shutting down")


app = FastAPI(title="FinancePilot API", lifespan=lifespan)
app.include_router(router)


@app.get("/")
def root() -> dict:
    logger.info("Root endpoint accessed")
    return {"message": "FinancePilot backend running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
