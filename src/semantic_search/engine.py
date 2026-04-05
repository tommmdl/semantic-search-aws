"""Core semantic search engine."""

from __future__ import annotations

import json
import logging
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class MotorBuscaSemantica:
    """Motor de busca semântica baseado em TF-IDF e similaridade de cosseno."""

    def __init__(self, vectorizer: TfidfVectorizer | None = None) -> None:
        self.vectorizer = vectorizer or TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),
        )
        self.matriz_tfidf = None
        self.documentos: list[dict[str, Any]] = []

    def carregar_documentos(self, caminho_json: str) -> None:
        """Carrega documentos JSON e constrói o índice vetorial."""
        with open(caminho_json, "r", encoding="utf-8") as file:
            documentos = json.load(file)

        self._validar_documentos(documentos)
        self.documentos = documentos

        textos = [doc["texto"] for doc in self.documentos]
        self.matriz_tfidf = self.vectorizer.fit_transform(textos)

        logger.info("%s documentos indexados.", len(self.documentos))
        logger.info(
            "Dimensão do vocabulário: %s termos.", self.matriz_tfidf.shape[1]
        )

    def buscar(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Retorna os documentos mais similares para a query informada."""
        if self.matriz_tfidf is None:
            raise RuntimeError("Índice vazio. Execute carregar_documentos() primeiro.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k deve ser um inteiro maior que zero.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query deve ser uma string não vazia.")

        vetor_query = self.vectorizer.transform([query])
        if vetor_query.nnz == 0:
            logger.warning("Query sem termos conhecidos no vocabulário; retornando [].")
            return []

        scores = cosine_similarity(vetor_query, self.matriz_tfidf).ravel()
        limite = min(top_k, len(self.documentos))
        indices_ordenados = scores.argsort()[::-1][:limite]

        return [
            {
                "id": self.documentos[idx].get("id"),
                "titulo": self.documentos[idx]["titulo"],
                "texto": self.documentos[idx]["texto"],
                "score": round(float(scores[idx]), 4),
            }
            for idx in indices_ordenados
        ]

    def exportar_estado(self) -> dict[str, Any]:
        """Serializa o estado necessário para restaurar o motor."""
        if self.matriz_tfidf is None:
            raise RuntimeError("Não há índice carregado para exportar.")

        return {
            "documentos": self.documentos,
            "vectorizer": self.vectorizer,
            "matriz_tfidf": self.matriz_tfidf,
        }

    @classmethod
    def restaurar_estado(cls, estado: dict[str, Any]) -> "MotorBuscaSemantica":
        """Reconstrói um motor a partir de um estado persistido."""
        documentos = estado.get("documentos")
        vectorizer = estado.get("vectorizer")
        matriz_tfidf = estado.get("matriz_tfidf")

        if not documentos or vectorizer is None or matriz_tfidf is None:
            raise ValueError("Estado inválido para restaurar o motor de busca.")

        motor = cls(vectorizer=vectorizer)
        motor.documentos = documentos
        motor.matriz_tfidf = matriz_tfidf
        return motor

    @staticmethod
    def _validar_documentos(documentos: Any) -> None:
        """Garante formato mínimo esperado do corpus."""
        if not isinstance(documentos, list) or not documentos:
            raise ValueError("O arquivo JSON deve conter uma lista não vazia de documentos.")

        obrigatorios = {"titulo", "texto"}
        for indice, documento in enumerate(documentos):
            if not isinstance(documento, dict):
                raise ValueError(f"Documento na posição {indice} não é um objeto JSON.")
            faltantes = obrigatorios - documento.keys()
            if faltantes:
                raise ValueError(
                    f"Documento na posição {indice} está sem os campos: {sorted(faltantes)}."
                )
            if not str(documento["texto"]).strip():
                raise ValueError(f"Documento na posição {indice} possui texto vazio.")
