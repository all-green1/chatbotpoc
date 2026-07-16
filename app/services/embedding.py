from __future__ import annotations

from functools import lru_cache

from fastapi import Depends
from loguru import logger
import sentry_sdk
from qdrant_client import AsyncQdrantClient

from app.core.settings import Settings
from app.core.embedder import GTETextEmbedder
from app.services.retrieval.vector_store import VectorStoreService


@lru_cache(maxsize=1)
def get_qdrant_client() -> AsyncQdrantClient:
    """
    Initializes and retrieves a cached instance of AsyncQdrantClient for interacting with
    the Qdrant vector database. This function uses an LRU (Least Recently Used) cache
    with a maximum size of 1 to ensure that the same client instance is reused across
    multiple calls, reducing initialization overhead.

    If the initialization of the client fails due to any error, the exception is logged
    and captured using the Sentry SDK before being re-raised to the caller.

    Returns:
        AsyncQdrantClient: A cached instance of the Qdrant client initialized with the
            provided settings.

    Raises:
        Exception: Propagates any initialization errors encountered during the creation
            of the Qdrant client instance.
    """
    logger.info("Initializing Qdrant client")
    try:
        settings = Settings()
        # noinspection HttpUrlsUsage
        client = AsyncQdrantClient(
            f"http://{settings.qdrant_host}:{settings.qdrant_port}"
        )
        logger.success("Qdrant client initialized")
        return client
    except Exception as e:
        logger.error()
        logger.error("Failed to initialize Qdrant client")
        sentry_sdk.capture_exception(e)
        raise


@lru_cache(maxsize=1)
def get_text_embedder() -> GTETextEmbedder:
    """
    Initializes and returns a singleton instance of the TextEmbedder class.

    This function initializes the TextEmbedder with the required configuration settings
    and ensures that only one instance is created and reused throughout the application
    execution by employing the Least Recently Used (LRU) cache mechanism with a maximum
    size of 1.

    Raises:
        Exception: If an error occurs during the initialization of the TextEmbedder.

    Returns:
        TextEmbedder: An instance of the TextEmbedder class, configured with the
            appropriate settings.
    """
    logger.info("Initializing TextEmbedder")
    try:
        settings = Settings()
        embedder = GTETextEmbedder(
            model_path=settings.embedding_model_path,
            vector_size=settings.embedding_vector_size,
            document_prefix=settings.embedding_document_prefix,
            query_prefix=settings.embedding_query_prefix,
            max_length=settings.embedding_max_length,
            batch_size=settings.embedding_batch_size,
        )
        logger.success("TextEmbedder initialized")
        return embedder
    except Exception as e:
        logger.error("Failed to initialize TextEmbedder")
        sentry_sdk.capture_exception(e)
        raise


@lru_cache(maxsize=1)
def get_vector_store_service(
    client: AsyncQdrantClient = Depends(get_qdrant_client),
    embedder: GTETextEmbedder = Depends(get_text_embedder),
) -> VectorStoreService:
    """
    Initializes and provides a singleton instance of the VectorStoreService.

    This function uses dependency injection to obtain an instance of AsyncQdrantClient
    and TextEmbedder. It initializes the VectorStoreService with these dependencies,
    along with configuration settings such as the distance metric. The VectorStoreService
    is responsible for managing vector data operations. A singleton instance is cached to
    ensure that the same service is reused.

    Args:
        client: Asynchronous Qdrant client used to interact with the vector database.
        embedder: Text embedding utility used for generating vector embeddings based
            on textual data.

    Returns:
        VectorStoreService: Initialized service instance.

    Raises:
        Exception: If the service fails to initialize for any reason, logs the error,
            captures the exception with Sentry, and re-raises the exception.
    """
    logger.info("Initializing VectorStoreService")
    try:
        settings = Settings()
        service = VectorStoreService(
            client=client,
            embedder=embedder,
            distance_metric=settings.distance_metric.value,
        )
        logger.success("VectorStoreService initialized")
        return service
    except Exception as e:
        logger.error("Failed to initialize VectorStoreService")
        sentry_sdk.capture_exception(e)
        raise