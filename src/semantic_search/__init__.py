"""Public API for the semantic search package."""

from .engine import MotorBuscaSemantica
from .math_utils import norma_vetor, produto_escalar, similaridade_cosseno
from .storage import (
    carregar_motor_arquivo,
    carregar_motor_s3,
    salvar_motor_arquivo,
    salvar_motor_s3,
)

__all__ = [
    "MotorBuscaSemantica",
    "norma_vetor",
    "produto_escalar",
    "similaridade_cosseno",
    "salvar_motor_arquivo",
    "carregar_motor_arquivo",
    "salvar_motor_s3",
    "carregar_motor_s3",
]
