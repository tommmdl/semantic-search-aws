from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic_search import MotorBuscaSemantica, carregar_motor_arquivo, salvar_motor_arquivo


DATASET_PATH = PROJECT_ROOT / "data" / "documentos.json"


def test_busca_sem_indice_lanca_erro() -> None:
    motor = MotorBuscaSemantica()

    with pytest.raises(RuntimeError):
        motor.buscar("aws")


def test_top_k_invalido_lanca_erro() -> None:
    motor = MotorBuscaSemantica()
    motor.carregar_documentos(str(DATASET_PATH))

    with pytest.raises(ValueError):
        motor.buscar("aws", top_k=0)


def test_query_fora_do_vocabulario_retorna_lista_vazia() -> None:
    motor = MotorBuscaSemantica()
    motor.carregar_documentos(str(DATASET_PATH))

    resultados = motor.buscar("criptozoologia", top_k=3)

    assert resultados == []


def test_persistencia_local_restabelece_motor(tmp_path: Path) -> None:
    motor = MotorBuscaSemantica()
    motor.carregar_documentos(str(DATASET_PATH))
    esperado = motor.buscar("armazenamento de dados na aws", top_k=2)

    caminho_modelo = tmp_path / "motor.pkl.gz"
    salvar_motor_arquivo(motor, str(caminho_modelo))
    restaurado = carregar_motor_arquivo(str(caminho_modelo))

    assert restaurado.buscar("armazenamento de dados na aws", top_k=2) == esperado


def test_top_k_maior_que_total_limita_resultados() -> None:
    motor = MotorBuscaSemantica()
    motor.carregar_documentos(str(DATASET_PATH))

    resultados = motor.buscar("machine learning", top_k=99)

    assert len(resultados) == 8
