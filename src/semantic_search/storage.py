"""Persistence helpers for semantic search engine state."""

from __future__ import annotations

import gzip
import io
import logging
import pickle
from pathlib import Path

from .engine import MotorBuscaSemantica

logger = logging.getLogger(__name__)


def _serializar_motor(motor: MotorBuscaSemantica) -> bytes:
    return gzip.compress(pickle.dumps(motor.exportar_estado()))


def _desserializar_motor(payload: bytes) -> MotorBuscaSemantica:
    estado = pickle.loads(gzip.decompress(payload))
    return MotorBuscaSemantica.restaurar_estado(estado)


def salvar_motor_arquivo(motor: MotorBuscaSemantica, caminho_arquivo: str) -> Path:
    """Salva o estado completo do motor em disco."""
    destino = Path(caminho_arquivo)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(_serializar_motor(motor))
    logger.info("Motor salvo em %s", destino)
    return destino


def carregar_motor_arquivo(caminho_arquivo: str) -> MotorBuscaSemantica:
    """Carrega o estado completo do motor a partir de um arquivo local."""
    origem = Path(caminho_arquivo)
    return _desserializar_motor(origem.read_bytes())


def salvar_motor_s3(motor: MotorBuscaSemantica, bucket: str, chave: str) -> bool:
    """Salva o estado completo do motor no Amazon S3."""
    try:
        import boto3

        payload = io.BytesIO(_serializar_motor(motor))
        boto3.client("s3").upload_fileobj(payload, bucket, chave)
        logger.info("Motor salvo em s3://%s/%s", bucket, chave)
        return True
    except ImportError:
        logger.error("boto3 não instalado. Execute: pip install boto3")
        return False
    except Exception as exc:
        logger.error("Erro ao salvar no S3: %s", exc)
        return False


def carregar_motor_s3(bucket: str, chave: str) -> MotorBuscaSemantica | None:
    """Carrega o estado completo do motor a partir do Amazon S3."""
    try:
        import boto3

        buffer = io.BytesIO()
        boto3.client("s3").download_fileobj(bucket, chave, buffer)
        buffer.seek(0)
        motor = _desserializar_motor(buffer.read())
        logger.info("Motor carregado de s3://%s/%s", bucket, chave)
        return motor
    except ImportError:
        logger.error("boto3 não instalado. Execute: pip install boto3")
        return None
    except Exception as exc:
        logger.error("Erro ao carregar do S3: %s", exc)
        return None
