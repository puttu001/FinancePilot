import asyncio
import hashlib
import inspect
import time
from qdrant_client import models
from core.database import get_qdrant_client, get_collection_name, sync_qdrant_client
from utils.logging import set_logger
from core.embeddings import (
	build_hybrid_vector_store,
	get_embedding_models,
	dense_model,
	sparse_model,
	default_vector_name,
	default_sparse_vector_name,
	dense_vector_size
	
)

logger = set_logger(__name__)

# Per-user cache: { user_id: {"data": [...], "timestamp": 0} }
_cached_files = {}


async def get_resources():
	"""Resolve all resources needed for Qdrant operations lazily.

	Returns a tuple of (sync_client, async_client, collection_name, dense, sparse).
	"""
	async_client = get_qdrant_client()
	sync_client = sync_qdrant_client()
	collection_name = await get_collection_name()
	dense, sparse = get_embedding_models(dense_model, sparse_model)
	return sync_client, async_client, collection_name, dense, sparse


async def _ensure_collection_and_indexes(async_client, collection_name: str):
	"""Create collection and payload indexes when missing using the async client."""
	if not await async_client.collection_exists(collection_name):
		await async_client.create_collection(
			collection_name=collection_name,
			vectors_config={
				default_vector_name: models.VectorParams(
					size=dense_vector_size,
					distance=models.Distance.COSINE,
				)
			},
			sparse_vectors_config={
				default_sparse_vector_name: models.SparseVectorParams()
			},
		)
		logger.info("Created Qdrant collection '%s'.", collection_name)

	# Safe to call repeatedly; errors are ignored if index already exists.
	for field_name in ("metadata.file_id", "metadata.user_id"):
		try:
			await async_client.create_payload_index(
				collection_name=collection_name,
				field_name=field_name,
				field_schema="keyword",
			)
		except Exception:
			logger.debug("Payload index already exists or could not be created: %s", field_name)


async def upload_to_qdrant(processed_docs, file_id, batch_size=50, progress_callback=None):
	"""Upload document chunks to Qdrant in batches."""
	if not processed_docs:
		logger.warning("No documents received for upload.")
		return 0

	sync_client, async_client, collection_name, dense, sparse = await get_resources()
	await _ensure_collection_and_indexes(async_client, collection_name)

	vectorstore = build_hybrid_vector_store(
		client=sync_client,
		collection_name=collection_name,
		dense=dense,
		sparse=sparse,
	)

	all_ids = [
		hashlib.md5(f"{file_id}_{d.page_content}".encode("utf-8")).hexdigest()
		for d in processed_docs
	]

	total_chunks = len(processed_docs)
	for i in range(0, total_chunks, batch_size):
		batch_docs = processed_docs[i : i + batch_size]
		batch_ids = all_ids[i : i + batch_size]

		if hasattr(vectorstore, "aadd_documents"):
			await vectorstore.aadd_documents(documents=batch_docs, ids=batch_ids)
		else:
			await asyncio.to_thread(
				vectorstore.add_documents,
				documents=batch_docs,
				ids=batch_ids,
			)

		if progress_callback:
			current = min(i + batch_size, total_chunks)
			callback_result = progress_callback(current, total_chunks)
			if inspect.isawaitable(callback_result):
				await callback_result

	invalidate_file_cache()
	logger.info("Uploaded %s chunks to collection '%s'.", total_chunks, collection_name)
	return total_chunks


async def delete_indexed_file(file_id: str, user_id: str = None):
	"""Delete all indexed chunks belonging to a file, scoped by optional user_id."""
	sync_client, async_client, collection_name, _, _ = await get_resources()

	if not await async_client.collection_exists(collection_name):
		return False

	must_conditions = [
		models.FieldCondition(
			key="metadata.file_id",
			match=models.MatchValue(value=file_id),
		)
	]

	if user_id:
		must_conditions.append(
			models.FieldCondition(
				key="metadata.user_id",
				match=models.MatchValue(value=user_id),
			)
		)

	await async_client.delete(
		collection_name=collection_name,
		points_selector=models.Filter(must=must_conditions),
	)

	invalidate_file_cache(user_id=user_id)
	return True


def invalidate_file_cache(user_id: str = None):
	"""Invalidate the cached document list for one user or all users."""
	if user_id and user_id in _cached_files:
		del _cached_files[user_id]
	elif user_id is None:
		_cached_files.clear()


async def get_indexed_files(cache_ttl=600, user_id: str = None):
	"""List indexed file IDs with per-user filtering and cache support."""
	current_time = time.time()
	cache_key = user_id or "__global__"

	entry = _cached_files.get(cache_key)
	if entry and current_time - entry["timestamp"] < cache_ttl:
		return entry["data"]

	sync_client, async_client, collection_name, _, _ = await get_resources()
	if not await async_client.collection_exists(collection_name):
		return []

	scroll_filter = None
	if user_id:
		scroll_filter = models.Filter(
			must=[
				models.FieldCondition(
					key="metadata.user_id",
					match=models.MatchValue(value=user_id),
				)
			]
		)

	files = set()
	offset = None
	while True:
		results, next_offset = await async_client.scroll(
			collection_name=collection_name,
			limit=500,
			offset=offset,
			with_payload=["metadata.file_id", "metadata.user_id"],
			with_vectors=False,
			scroll_filter=scroll_filter,
		)

		for point in results:
			file_name = point.payload.get("metadata", {}).get("file_id")
			if file_name:
				files.add(file_name)

		if next_offset is None:
			break
		offset = next_offset

	file_list = sorted(list(files))
	_cached_files[cache_key] = {"data": file_list, "timestamp": current_time}
	return file_list

