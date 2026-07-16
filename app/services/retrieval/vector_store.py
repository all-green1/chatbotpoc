"""Async vector store service with search and storage operations (POC refactor).

This refactor removes book-specific concepts (slug, metadata collection, TOC/page-based top_k logic)
and standardizes filtering on a single `article_id`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING, Union
from uuid import uuid4

from loguru import logger
import sentry_sdk
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Batch,
    FieldCondition,
    Filter,
    MatchExcept,
    MatchValue,
    ScoredPoint,
    VectorParams,
)

from app.core.embedder import GTETextEmbedder

if TYPE_CHECKING:
    import numpy as np


class VectorStoreService:
    """Manages vector storage and search operations (article-focused POC)."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        embedder: GTETextEmbedder,
        distance_metric: str,
    ) -> None:
        logger.info("Initializing VectorStoreService")
        self.embedder: GTETextEmbedder = embedder
        self.distance_metric: str = distance_metric
        self.vector_store: AsyncQdrantClient = client
        logger.success("VectorStoreService initialized")

    async def embed(self, docs: Sequence[str], is_query: bool = False) -> List["np.ndarray"]:
        """Embeds texts with prefixing."""
        logger.debug(f"Embedding {len(docs)} documents (is_query={is_query})")
        try:
            processed_docs = (
                [self.embedder.prepare_query(doc) for doc in docs]
                if is_query
                else [self.embedder.prepare_document(doc) for doc in docs]
            )
            result = await asyncio.to_thread(self.embedder.embed, processed_docs)
            logger.debug("Embedded successfully")
            return result
        except Exception as e:
            logger.error("Failed to embed")
            sentry_sdk.capture_exception(e)
            raise

    async def create_collection(self, collection_name: str) -> Any:
        """Creates a collection."""
        logger.info(f"Creating {collection_name}")
        try:
            result = await self.vector_store.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.vector_size,
                    distance=self.distance_metric,
                ),
            )
            logger.success(f"Created {collection_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to create {collection_name}")
            sentry_sdk.capture_exception(e)
            raise

    async def delete_collection(self, collection_name: str) -> Any:
        """Deletes a collection."""
        logger.info(f"Deleting {collection_name}")
        try:
            result = await self.vector_store.delete_collection(collection_name)
            logger.success(f"Deleted {collection_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to delete {collection_name}")
            sentry_sdk.capture_exception(e)
            raise

    async def collection_exists(self, collection_name: str) -> bool:
        """Checks collection existence."""
        logger.debug(f"Checking existence of {collection_name}")
        try:
            result = await self.vector_store.collection_exists(collection_name)
            logger.debug(f"Exists: {result}")
            return bool(result)
        except Exception as e:
            logger.error("Failed to check existence")
            sentry_sdk.capture_exception(e)
            raise

    async def get_collection_size(self, collection_name: str) -> int:
        """Gets point count."""
        logger.debug(f"Getting size of {collection_name}")
        try:
            result = await self.vector_store.count(collection_name, exact=True)
            logger.debug(f"Size: {result.count}")
            return int(result.count)
        except Exception as e:
            logger.error("Failed to get size")
            sentry_sdk.capture_exception(e)
            raise

    async def upload_documents(
        self,
        collection_name: str,
        docs: Sequence[Dict[str, Any]],
        *,
        embedding_field: str = "content",
        ids: Optional[Sequence[Union[str, int]]] = None,
    ) -> Any:
        """Upserts payload documents and embeds `embedding_field`."""
        logger.info(f"Uploading {len(docs)} to {collection_name}")
        try:
            if not docs:
                return {"status": "ok", "uploaded": 0}

            point_ids: List[Union[str, int]] = (
                list(ids) if ids is not None else [str(uuid4()) for _ in docs]
            )

            texts: List[str] = []
            for i, doc in enumerate(docs):
                if embedding_field not in doc:
                    raise KeyError(
                        f"Doc at index {i} missing embedding_field={embedding_field!r}"
                    )
                texts.append(str(doc[embedding_field] or ""))

            vectors = await self.embed(texts, is_query=False)

            result = await self.vector_store.upsert(
                collection_name=collection_name,
                points=Batch(ids=point_ids, vectors=vectors, payloads=list(docs)),
            )
            logger.success(f"Uploaded {len(docs)}")
            return result
        except Exception as e:
            logger.error("Failed to upload")
            sentry_sdk.capture_exception(e)
            raise

    async def search_documents(
        self,
        *,
        collection_name: str,
        query: str,
        article_id: str,
        top_k: Optional[int] = None,
        exclude_ids: Optional[Sequence[Union[str, int]]] = None,
    ) -> List[ScoredPoint]:
        """Searches for nearest neighbors restricted to a single `article_id`.

        This expects each chunk payload to include:
          - article_id: str
          - content: str (or whatever you embedded)
          - optional: source_type, tags, etc.

        Filtering is performed using payload field `article_id`.
        """
        logger.debug(f"Searching in {collection_name} with query: {query[:50]}...")
        try:
            aid = (article_id or "").strip()
            if not aid:
                raise ValueError("article_id is required")

            embedded = (await self.embed([query], is_query=True))[0]

            query_filter = Filter(must=[], should=[], must_not=[])
            query_filter.must.append(
                FieldCondition(
                    key="article_id",
                    match=MatchValue(value=aid),
                )
            )

            if exclude_ids:
                cleaned_exclude_ids = [x for x in exclude_ids if x is not None and x != ""]
                if cleaned_exclude_ids:
                    query_filter.must_not.append(
                        FieldCondition(
                            key="id",
                            match=MatchExcept(**{"except": cleaned_exclude_ids}),
                        )
                    )

            limit = int(top_k or 20)

            result = await self.vector_store.query_points(
                collection_name=collection_name,
                query=embedded,
                limit=limit,
                query_filter=query_filter,
            )
            points = result.points or []

            logger.info(
                f"[search_documents] collection={collection_name} article_id={aid!r} "
                f"limit={limit} returned_points={len(points)}"
            )
            return points
        except Exception as e:
            logger.error("Failed to search")
            sentry_sdk.capture_exception(e)
            raise

    async def retrieve_by_ids(
        self,
        collection_name: str,
        point_ids: Sequence[Union[str, int]],
    ) -> List[Any]:
        """Retrieves points by IDs."""
        logger.debug(f"Retrieving {len(point_ids)} from {collection_name}")
        try:
            results = await self.vector_store.retrieve(
                collection_name=collection_name,
                ids=list(point_ids),
            )
            logger.debug(f"Retrieved {len(results)}")
            return results
        except Exception as e:
            logger.error("Failed to retrieve")
            sentry_sdk.capture_exception(e)
            raise

    async def list_collections(self) -> List[str]:
        """Lists collection names."""
        logger.debug("Listing collections")
        try:
            response = await self.vector_store.get_collections()
            names = [c.name for c in response.collections]
            logger.debug(f"Found {len(names)}")
            return names
        except Exception as e:
            logger.error("Failed to list")
            sentry_sdk.capture_exception(e)
            raise