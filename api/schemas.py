"""Pydantic schemas for the semantic search API."""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class DocumentoInput(BaseModel):
    id: Optional[str] = None
    titulo: str
    texto: str


class IndexarDocumentosRequest(BaseModel):
    documentos: list[DocumentoInput]


class IndexarS3Request(BaseModel):
    bucket: str
    chave: str


class BuscarRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Texto da busca")
    top_k: int = Field(default=3, ge=1, le=50, description="Número de resultados")


class ResultadoBusca(BaseModel):
    id: Optional[str]
    titulo: str
    texto: str
    score: float


class BuscarResponse(BaseModel):
    query: str
    total: int
    resultados: list[ResultadoBusca]


class StatusResponse(BaseModel):
    indexado: bool
    total_documentos: int
    total_termos: int


class HealthResponse(BaseModel):
    status: str
    indexado: bool
