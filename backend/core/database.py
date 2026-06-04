import asyncio
from qdrant_client import AsyncQdrantClient, QdrantClient
from utils.config import get_config
from utils.logging import set_logger

logger = set_logger(__name__)

# We store the client in a variable at the module level (Singleton pattern)
_async_client = None
_sync_client = None

def sync_qdrant_client():
    global _sync_client

    if _sync_client is None:
        db_config = get_config("vector_db")

        _sync_client = QdrantClient(
            url=db_config["qdrant_url"],
            api_key=db_config["qdrant_api_key"],
            timeout=60,
        )
    return _sync_client

def get_qdrant_client():
    global _async_client

    if _async_client is None:
        db_config = get_config("vector_db")
        url = db_config.get("qdrant_url")
        api_key = db_config.get("qdrant_api_key")

        if not url:
            raise ValueError("QDRANT_URL is missing in configuration/environment.")

        _async_client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            timeout=60,
        )
        logger.info("Async Qdrant client initialized for url: %s", url)

    return _async_client



async def get_collection_name():
    db_config = get_config("vector_db")
    configured_name = db_config.get("collection_name")

    try:
        client = get_qdrant_client()
        collections_response = await client.get_collections()
        cloud_names = [c.name for c in collections_response.collections]

        if configured_name in cloud_names:
            logger.info("Using existing Qdrant collection from cloud: %s", configured_name)
            return configured_name

        if cloud_names:
            fallback_name = cloud_names[0]
            logger.warning(
                "Configured collection '%s' not found. Falling back to cloud collection '%s'.",
                configured_name,
                fallback_name,
            )
            return fallback_name
    except Exception:
        logger.warning(
            "Could not fetch collections from Qdrant cloud. Using configured collection '%s'.",
            configured_name,
            exc_info=True,
        )

    return configured_name

async def main():
    client = sync_qdrant_client()

    collections = client.get_collections()
    print(collections)

    collection_name =  await get_collection_name()
    print(collection_name)

if __name__ == "__main__":
    asyncio.run(main())