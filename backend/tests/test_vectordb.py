from src.ingestion.splitter import DocumentSplitter
import asyncio
from src.ingestion.doc_loader import DocumentLoader
from src.ingestion.vectordb import upload_to_qdrant
async def main():
    loader = DocumentLoader()
    splitter=DocumentSplitter()
    document = r"data/raw/ihcl.pdf"
    parsed_results = await loader.load_and_parse(document)
    chunks = splitter.parsed_data_to_chunks(parsed_results,document)
    uploaded = await upload_to_qdrant(chunks,document)
    print(f"Uploaded {uploaded} chunks")


if __name__ == "__main__":
    asyncio.run(main())


