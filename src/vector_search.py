"""
vector_search.py
----------------
Módulo de busca semântica usando álgebra linear e integração com AWS S3.

Conceitos aplicados:
- Norma de vetor: para normalizar representações de documentos
- Produto escalar: base do cálculo de similaridade
- Similaridade de cosseno: métrica de comparação entre vetores

Autor: Rafael
"""

import json
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuração de logging — boa prática em projetos reais
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# BLOCO 1 — ÁLGEBRA LINEAR DO ZERO
# Aqui implementamos os conceitos da Pós sem usar bibliotecas prontas.
# Objetivo: entender o que acontece "por baixo do capô".
# =============================================================================

def norma_vetor(vetor: np.ndarray) -> float:
    """
    Calcula a norma euclidiana (comprimento) de um vetor.

    A norma é a raiz quadrada da soma dos quadrados de cada elemento.
    Em busca semântica, usamos a norma para normalizar vetores antes
    de comparar — assim vetores longos e curtos ficam na mesma escala.

    Args:
        vetor: Array numpy com os valores do vetor.

    Returns:
        Valor float representando o comprimento do vetor.

    Exemplo:
        >>> norma_vetor(np.array([3, 4]))
        5.0
    """
    return float(np.sqrt(np.sum(vetor ** 2)))


def produto_escalar(vetor_a: np.ndarray, vetor_b: np.ndarray) -> float:
    """
    Calcula o produto escalar entre dois vetores.

    O produto escalar multiplica os elementos correspondentes e soma tudo.
    Quanto maior o resultado, mais os vetores "apontam na mesma direção",
    ou seja, mais similares são os documentos que eles representam.

    Args:
        vetor_a: Primeiro vetor.
        vetor_b: Segundo vetor.

    Returns:
        Valor float do produto escalar.

    Exemplo:
        >>> produto_escalar(np.array([1, 3, -5]), np.array([4, -2, -1]))
        3.0
    """
    return float(np.sum(vetor_a * vetor_b))


def similaridade_cosseno(vetor_a: np.ndarray, vetor_b: np.ndarray) -> float:
    """
    Calcula a similaridade de cosseno entre dois vetores.

    Essa é a métrica central de busca semântica. Ela divide o produto
    escalar pelo produto das normas — normalizando o resultado para
    ficar entre -1 e 1, independente do tamanho dos vetores.

    Interpretação:
        1.0  → vetores idênticos (documentos muito similares)
        0.0  → vetores perpendiculares (sem relação)
       -1.0  → vetores opostos (sentidos contrários)

    Em textos, raramente vemos valores negativos pois TF-IDF não
    gera pesos negativos.

    Args:
        vetor_a: Vetor do documento ou da query.
        vetor_b: Vetor do documento a comparar.

    Returns:
        Float entre -1 e 1 indicando o grau de similaridade.
    """
    norma_a = norma_vetor(vetor_a)
    norma_b = norma_vetor(vetor_b)

    # Proteção contra divisão por zero (vetor nulo)
    if norma_a == 0 or norma_b == 0:
        logger.warning("Vetor nulo detectado — similaridade retornada como 0.")
        return 0.0

    return produto_escalar(vetor_a, vetor_b) / (norma_a * norma_b)


# =============================================================================
# BLOCO 2 — MOTOR DE BUSCA
# Aqui usamos os conceitos acima para construir um sistema real.
# =============================================================================

class MotorBuscaSemantica:
    """
    Motor de busca semântica baseado em TF-IDF e similaridade de cosseno.

    TF-IDF (Term Frequency-Inverse Document Frequency) transforma texto
    em vetores numéricos, onde cada dimensão representa uma palavra do
    vocabulário. Palavras frequentes num documento mas raras no corpus
    recebem peso maior.

    Atributos:
        vectorizer: Objeto TfidfVectorizer do scikit-learn.
        matriz_tfidf: Matriz onde cada linha é um vetor TF-IDF de um documento.
        documentos: Lista de dicionários com os documentos originais.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2)   # considera palavras simples e pares de palavras
        )
        self.matriz_tfidf = None
        self.documentos = []

    def carregar_documentos(self, caminho_json: str) -> None:
        """
        Carrega documentos de um arquivo JSON e constrói o índice vetorial.

        Args:
            caminho_json: Caminho para o arquivo JSON com os documentos.
        """
        with open(caminho_json, "r", encoding="utf-8") as f:
            self.documentos = json.load(f)

        textos = [doc["texto"] for doc in self.documentos]
        self.matriz_tfidf = self.vectorizer.fit_transform(textos).toarray()

        logger.info(f"{len(self.documentos)} documentos indexados.")
        logger.info(f"Dimensão do vocabulário: {self.matriz_tfidf.shape[1]} termos.")

    def buscar(self, query: str, top_k: int = 3) -> list:
        """
        Busca os documentos mais similares à query fornecida.

        O processo é:
        1. Transforma a query em vetor TF-IDF (mesmo espaço dos documentos)
        2. Calcula similaridade de cosseno entre a query e cada documento
        3. Retorna os top_k documentos com maior similaridade

        Args:
            query: Texto da busca.
            top_k: Número de resultados a retornar.

        Returns:
            Lista de dicionários com os documentos mais similares e seus scores.
        """
        if self.matriz_tfidf is None:
            raise RuntimeError("Índice vazio. Execute carregar_documentos() primeiro.")

        # Vetoriza a query usando o mesmo vocabulário dos documentos
        vetor_query = self.vectorizer.transform([query]).toarray()[0]

        # Calcula similaridade da query com cada documento
        scores = [
            similaridade_cosseno(vetor_query, self.matriz_tfidf[i])
            for i in range(len(self.documentos))
        ]

        # Ordena por score decrescente e retorna os top_k
        indices_ordenados = np.argsort(scores)[::-1][:top_k]

        resultados = []
        for idx in indices_ordenados:
            resultados.append({
                "titulo": self.documentos[idx]["titulo"],
                "texto": self.documentos[idx]["texto"],
                "score": round(scores[idx], 4)
            })

        return resultados


# =============================================================================
# BLOCO 3 — INTEGRAÇÃO AWS S3
# Persiste e carrega o índice vetorial na nuvem.
# =============================================================================

def salvar_indice_s3(matriz: np.ndarray, bucket: str, chave: str) -> bool:
    """
    Salva a matriz de vetores TF-IDF no Amazon S3.

    Em produção, isso permite que múltiplos workers carreguem o mesmo
    índice sem precisar recalcular — economia de tempo e custo.

    Args:
        matriz: Matriz numpy com os vetores dos documentos.
        bucket: Nome do bucket S3.
        chave: Caminho do objeto dentro do bucket (ex: "indices/tfidf.npy").

    Returns:
        True se salvou com sucesso, False caso contrário.
    """
    try:
        import boto3
        import io

        buffer = io.BytesIO()
        np.save(buffer, matriz)
        buffer.seek(0)

        s3 = boto3.client("s3")
        s3.upload_fileobj(buffer, bucket, chave)

        logger.info(f"Índice salvo em s3://{bucket}/{chave}")
        return True

    except ImportError:
        logger.error("boto3 não instalado. Execute: pip install boto3")
        return False
    except Exception as e:
        logger.error(f"Erro ao salvar no S3: {e}")
        return False


def carregar_indice_s3(bucket: str, chave: str) -> np.ndarray | None:
    """
    Carrega a matriz de vetores TF-IDF do Amazon S3.

    Args:
        bucket: Nome do bucket S3.
        chave: Caminho do objeto dentro do bucket.

    Returns:
        Matriz numpy com os vetores, ou None em caso de erro.
    """
    try:
        import boto3
        import io

        s3 = boto3.client("s3")
        buffer = io.BytesIO()
        s3.download_fileobj(bucket, chave, buffer)
        buffer.seek(0)

        matriz = np.load(buffer)
        logger.info(f"Índice carregado de s3://{bucket}/{chave} — shape: {matriz.shape}")
        return matriz

    except ImportError:
        logger.error("boto3 não instalado. Execute: pip install boto3")
        return None
    except Exception as e:
        logger.error(f"Erro ao carregar do S3: {e}")
        return None
