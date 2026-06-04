from src.ingestion.doc_loader import DocumentLoader
from src.ingestion.splitter import DocumentSplitter
from src.ingestion.vectordb import upload_to_qdrant
from utils.logging import set_logger
logger = set_logger(__name__)


async def ingestion_pipeline_flow(file, user_id="Anonymous"):
    try:

        loader = DocumentLoader()
        splitter = DocumentSplitter()

        parsed_results = await loader.load_and_parse(file)
        chunks = splitter.parsed_data_to_chunks(parsed_results, file, user_id=user_id)
        count = await upload_to_qdrant(chunks,file)
        return count
    except Exception as exc:
        logger.exception("Ingestion failed for %s", file)
        raise

