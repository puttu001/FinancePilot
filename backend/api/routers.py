from fastapi import APIRouter
from fastapi.responses import JSONResponse
from src.ingestion.doc_loader import DocumentLoader
from src.ingestion.splitter import DocumentSplitter
from utils.logging import set_logger
logger = set_logger(__name__)

router = APIRouter(prefix="/api")




@router.get('/health')
def health_check():
    logger.info('Health Status fetched.')
    return {'FinancePilot':'A financial intelligent system','status':'ok'}

@router.get('/data-ingestion')
def data_ingestion()