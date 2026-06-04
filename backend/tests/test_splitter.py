from src.ingestion.splitter import DocumentSplitter
import asyncio
from src.ingestion.doc_loader import DocumentLoader
async def main():
    loader = DocumentLoader()
    splitter=DocumentSplitter()
    document = r"data/raw/ihcl.pdf"
    parsed_results = await loader.load_and_parse(document)
    result = splitter.parsed_data_to_chunks(parsed_results,document)
    print(len(result))
    print(type(result))

asyncio.run(main())


