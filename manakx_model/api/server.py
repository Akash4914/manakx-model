"""
REST wrapper around the MANAKX pipeline.

Run with:
    uvicorn manakx_model.api.server:app --reload --port 8000

Endpoints:
    POST /analyze         product description / detailed spec -> recommended standards
    POST /analyze-tender   full tender text -> standards audit (referenced/outdated/missing)
    GET  /revision/{ref}   direct "is this standard current?" lookup, e.g. /revision/IS 9137:2010
    GET  /health
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..pipeline import ManakxPipeline

app = FastAPI(
    title="MANAKX Model API",
    description="AI Indian Standards Assistant: semantic standards search, "
    "allied-standards knowledge graph, revision checking, certification "
    "recommendation, and tender document auditing.",
    version="0.2.0",
)

# The frontend (e.g. https://manakx.vercel.app) runs in the browser and
# calls this API directly, so it needs CORS enabled. Set MANAKX_ALLOWED_ORIGINS
# to a comma-separated list in production; defaults to the deployed
# dashboard + localhost for local frontend development.
_default_origins = "https://manakx.vercel.app,http://localhost:3000,http://localhost:5173"
_allowed_origins = os.environ.get("MANAKX_ALLOWED_ORIGINS", _default_origins).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Built once at process startup and reused across requests.
_pipeline = ManakxPipeline()


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Product description or detailed specification")
    retrieval_top_k: int = Field(default=10, ge=1, le=20)
    final_top_k: int = Field(default=5, ge=1, le=10)


class TenderRequest(BaseModel):
    text: str = Field(..., description="Full tender document text")
    retrieval_top_k: int = Field(default=10, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "standards_loaded": len(_pipeline.repository),
        "embedding_backend": _pipeline.semantic_search.backend_name,
        "ranking_backend": _pipeline.ranker.name,
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    try:
        return _pipeline.analyze(
            input_text=request.text,
            retrieval_top_k=request.retrieval_top_k,
            final_top_k=request.final_top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze-tender")
def analyze_tender(request: TenderRequest) -> dict:
    try:
        return _pipeline.analyze_tender(
            input_text=request.text, retrieval_top_k=request.retrieval_top_k
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/revision/{reference}")
def check_revision(reference: str) -> dict:
    return _pipeline.check_revision(reference)
