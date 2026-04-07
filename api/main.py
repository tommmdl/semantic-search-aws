"""FastAPI application for semantic search engine."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_search import (
    MotorBuscaSemantica,
    carregar_motor_arquivo,
    carregar_motor_s3,
    salvar_motor_s3,
)
from api.schemas import (
    BuscarRequest,
    BuscarResponse,
    HealthResponse,
    IndexarDocumentosRequest,
    IndexarS3Request,
    ResultadoBusca,
    StatusResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance
motor: MotorBuscaSemantica | None = None
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "documentos.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load default index on startup if available."""
    global motor
    if DEFAULT_DATA_PATH.exists():
        try:
            motor = MotorBuscaSemantica()
            motor.carregar_documentos(str(DEFAULT_DATA_PATH))
            logger.info("Índice carregado automaticamente de %s", DEFAULT_DATA_PATH)
        except Exception as exc:
            logger.warning("Não foi possível carregar índice padrão: %s", exc)
    yield


app = FastAPI(
    title="Semantic Search API",
    description="Motor de busca semântica com TF-IDF e similaridade de cosseno. Persistência em AWS S3.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Infra"])
def health():
    return HealthResponse(status="ok", indexado=motor is not None)


@app.get("/status", response_model=StatusResponse, tags=["Infra"])
def status():
    if motor is None or motor.matriz_tfidf is None:
        return StatusResponse(indexado=False, total_documentos=0, total_termos=0)
    return StatusResponse(
        indexado=True,
        total_documentos=len(motor.documentos),
        total_termos=motor.matriz_tfidf.shape[1],
    )


@app.post("/index/documents", tags=["Indexação"])
def indexar_documentos(body: IndexarDocumentosRequest):
    """Indexa uma lista de documentos enviados no corpo da requisição."""
    global motor
    import json, tempfile, os

    docs = [d.dict() for d in body.documentos]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
        tmp_path = f.name

    try:
        motor = MotorBuscaSemantica()
        motor.carregar_documentos(tmp_path)
    finally:
        os.unlink(tmp_path)

    return {
        "message": f"{len(docs)} documentos indexados com sucesso.",
        "total_termos": motor.matriz_tfidf.shape[1],
    }


@app.post("/index/s3", tags=["Indexação"])
def carregar_de_s3(body: IndexarS3Request):
    """Carrega um índice previamente salvo no Amazon S3."""
    global motor
    motor = carregar_motor_s3(body.bucket, body.chave)
    if motor is None:
        raise HTTPException(status_code=500, detail="Falha ao carregar índice do S3.")
    return {
        "message": f"Índice carregado de s3://{body.bucket}/{body.chave}",
        "total_documentos": len(motor.documentos),
    }


@app.post("/index/s3/save", tags=["Indexação"])
def salvar_em_s3(body: IndexarS3Request):
    """Salva o índice atual no Amazon S3."""
    if motor is None:
        raise HTTPException(status_code=400, detail="Nenhum índice carregado.")
    ok = salvar_motor_s3(motor, body.bucket, body.chave)
    if not ok:
        raise HTTPException(status_code=500, detail="Falha ao salvar no S3.")
    return {"message": f"Índice salvo em s3://{body.bucket}/{body.chave}"}


@app.post("/search", response_model=BuscarResponse, tags=["Busca"])
def buscar(body: BuscarRequest):
    """Busca documentos semanticamente similares à query."""
    if motor is None:
        raise HTTPException(
            status_code=503,
            detail="Índice não carregado. Use POST /index/documents ou /index/s3 primeiro.",
        )
    try:
        resultados = motor.buscar(body.query, top_k=body.top_k)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return BuscarResponse(
        query=body.query,
        total=len(resultados),
        resultados=[ResultadoBusca(**r) for r in resultados],
    )
