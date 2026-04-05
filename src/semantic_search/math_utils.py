"""Foundational vector operations used in the educational part of the project."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def norma_vetor(vetor: np.ndarray) -> float:
    """Calcula a norma euclidiana de um vetor denso."""
    return float(np.sqrt(np.sum(vetor**2)))


def produto_escalar(vetor_a: np.ndarray, vetor_b: np.ndarray) -> float:
    """Calcula o produto escalar entre dois vetores densos."""
    return float(np.sum(vetor_a * vetor_b))


def similaridade_cosseno(vetor_a: np.ndarray, vetor_b: np.ndarray) -> float:
    """Calcula similaridade de cosseno entre vetores densos."""
    norma_a = norma_vetor(vetor_a)
    norma_b = norma_vetor(vetor_b)

    if norma_a == 0 or norma_b == 0:
        logger.warning("Vetor nulo detectado; similaridade retornada como 0.")
        return 0.0

    return produto_escalar(vetor_a, vetor_b) / (norma_a * norma_b)
