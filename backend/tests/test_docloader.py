import asyncio
from src.ingestion.doc_loader import DocumentLoader
async def main():
    loader = DocumentLoader()
    document = r"data/raw/ihcl.pdf"
    result = await loader.load_and_parse(document)
    print(result)

asyncio.run(main())


