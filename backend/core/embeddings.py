from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from functools import lru_cache
from utils.config import get_config

embed_config = get_config("embeddings")
dense_model = embed_config.get("DENSE_MODEL")
sparse_model = embed_config.get("SPARSE_MODEL")
dense_vector_size = embed_config.get("DENSE_VECTOR_SIZE")
default_vector_name = embed_config.get("VECTOR_NAME", "text-dense")
default_sparse_vector_name = embed_config.get("SPARSE_VECTOR_NAME", "text-sparse")


@lru_cache(maxsize=16)
def get_embedding_models(dense_model: str, sparse_model: str):
    dense = OpenAIEmbeddings(model=dense_model)
    sparse = FastEmbedSparse(model=sparse_model)
    return dense, sparse


def build_hybrid_vector_store(
    client,
    collection_name,
    dense=None,
    sparse=None,
    vector_name: str | None = None,
    sparse_vector_name: str | None = None,
):
    """Build a Qdrant hybrid vector store using configured embedding models."""
    if dense is None or sparse is None:
        dense, sparse = get_embedding_models(dense_model, sparse_model)

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense,
        sparse_embedding=sparse,
        vector_name=vector_name or default_vector_name,
        sparse_vector_name=sparse_vector_name or default_sparse_vector_name,
        retrieval_mode=RetrievalMode.HYBRID,
    )





