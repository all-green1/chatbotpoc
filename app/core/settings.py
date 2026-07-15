from __future__ import annotations

from os import path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        qdrant_host: Qdrant server hostname.
        qdrant_port: Qdrant server port.
        embedding_model_path: Path to the embedding model.
        embedding_vector_size: Dimensionality of embedding vectors.
        embedding_document_prefix: Prefix for document texts.
        embedding_query_prefix: Prefix for query texts.
        distance_metric: Vector similarity metric.
    """

    # noinspection PyArgumentList
    model_config = SettingsConfigDict(env_ignore_case=True)

    qdrant_host: str
    qdrant_port: int
    embedding_model_path: str = path.join(
        path.dirname(path.dirname(path.dirname(path.realpath(__file__)))),
        "models/gte-multilingual-base-onnx",
    )
    embedding_vector_size: int = 768
    embedding_document_prefix: str = "search_document: "
    embedding_query_prefix: str = "search_query: "
    distance_metric: DistanceMetric = DistanceMetric.COSINE

    embedding_max_length: int = 2048
    embedding_batch_size: int = 32