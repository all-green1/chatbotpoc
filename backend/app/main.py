from fastapi import FastAPI

from backend.app.routers.ingest import router as ingest_router

app = FastAPI(title="RAG POC")

app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])