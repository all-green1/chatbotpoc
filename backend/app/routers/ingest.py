from typing import Any, List, Annotated
from fastapi import Depends, HTTPException, APIRouter
from pydantic import BaseModel, Field
from loguru import logger
import urllib.request
from backend.app.services.embedding import get_vector_store_service
from backend.app.services.retrieval.vector_store import VectorStoreService
from backend.app.services.chunker import ChunkerService, get_chunker_service

router = APIRouter()

class ArticleUrlIngest(BaseModel):
    article_id: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    tags: List[str] = Field(default_factory=list)


def guess_source_type(url: str, content_type: str) -> str:
    ct = (content_type or "").lower()
    u = (url or "").lower()
    if "application/pdf" in ct or u.endswith(".pdf"):
        return "pdf"
    if "text/markdown" in ct or u.endswith(".md"):
        return "markdown"
    return "text"


def extract_text_from_pdf_bytes(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF support not available: {e}")

    import io
    reader = PdfReader(io.BytesIO(data))
    parts: List[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


@router.post(
    "/article/from-url",
    summary="Ingest an article from a URL (markdown/text, PDF optional)",
    response_model=Any,
)
async def ingest_article_from_url(
    item: ArticleUrlIngest,
    vector_store_service: Annotated[
        VectorStoreService, Depends(get_vector_store_service)
    ],
    chunker_service: Annotated[
        ChunkerService, Depends(get_chunker_service)
    ],
) -> Any:
    logger.info(f"Fetching article url={item.url!r} article_id={item.article_id!r}")
    try:
        req = urllib.request.Request(
            item.url,
            headers={"User-Agent": "rag-poc-ingester/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    source_type = guess_source_type(item.url, content_type)

    if source_type == "pdf":
        text = extract_text_from_pdf_bytes(data)
    else:
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception:
            text = str(data)

    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="No text could be extracted from the provided URL")

    chunks = await chunker_service.chunk_article_text(
        article_id=item.article_id,
        source_type=source_type,
        text=text,
    )

    # document-level tags on each chunk payload
    for c in chunks:
        c["tags"] = item.tags

    collection_name = "article"
    if not await vector_store_service.collection_exists(collection_name):
        logger.info(f"Creating collection: {collection_name!r}")
        await vector_store_service.create_collection(collection_name)

    logger.info(f"Uploading {len(chunks)} chunks for article_id={item.article_id!r}")
    return await vector_store_service.upload_documents(
        collection_name=collection_name,
        docs=chunks,
        embedding_field="content",
    )