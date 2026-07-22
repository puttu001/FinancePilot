import asyncio
from src.ingestion.ingestion_pipeline import ingestion_pipeline_flow

if __name__ == "__main__":
    file = r"data/raw/ril.pdf"
    print("File Loaded")
    result = asyncio.run(ingestion_pipeline_flow(file))
    print(result)
