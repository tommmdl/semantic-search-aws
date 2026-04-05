# Busca Semântica com Álgebra Linear e AWS S3

Projeto de portfólio que demonstra como conceitos fundamentais de álgebra linear,
vetorização TF-IDF e persistência de estado podem ser aplicados em um mecanismo
simples de busca semântica. O foco aqui é didático, mas com organização suficiente
para servir como base de evolução para RAG, APIs e integrações com AWS.

## Sumário

- [Visão geral](#visao-geral)
- [Arquitetura](#arquitetura)
- [Estrutura do repositório](#estrutura-do-repositorio)
- [Pré-requisitos](#pre-requisitos)
- [Instalação](#instalacao)
- [Como executar](#como-executar)
- [Exemplo rápido](#exemplo-rapido)
- [Persistência local e S3](#persistencia-local-e-s3)
- [Testes](#testes)
- [Decisões técnicas](#decisoes-tecnicas)
- [Limitações atuais](#limitacoes-atuais)
- [Roadmap](#roadmap)

## Visao geral

- Implementação de similaridade de cosseno do zero com NumPy
- Motor de busca com TF-IDF esparso e ranqueamento por similaridade de cosseno
- Pipeline completo: texto -> vetor TF-IDF -> busca por similaridade
- Persistência do estado completo do motor em arquivo local ou AWS S3
- Validação de entrada, tratamento de erros, logging e testes automatizados

## Contexto técnico

A busca semântica é a base de sistemas como RAG, motores de recomendação e busca
inteligente. Este projeto implementa uma versão introdutória desse fluxo usando
álgebra linear clássica e bibliotecas Python amplamente adotadas.

## Arquitetura

Fluxo principal:

1. O corpus é carregado de [`data/documentos.json`](/mnt/c/Users/rafael.santiago/Documents/Projeto%20bedrock/semantic-search-aws/data/documentos.json).
2. O motor em [`src/semantic_search/engine.py`](/mnt/c/Users/rafael.santiago/Documents/Projeto%20bedrock/semantic-search-aws/src/semantic_search/engine.py) valida os documentos e ajusta o `TfidfVectorizer`.
3. As consultas são vetorizadas no mesmo espaço do corpus.
4. O ranqueamento é calculado com similaridade de cosseno sobre a matriz TF-IDF esparsa.
5. O estado completo do motor pode ser persistido localmente ou no S3 por [`src/semantic_search/storage.py`](/mnt/c/Users/rafael.santiago/Documents/Projeto%20bedrock/semantic-search-aws/src/semantic_search/storage.py).

Componentes:

- [`src/semantic_search/math_utils.py`](/mnt/c/Users/rafael.santiago/Documents/Projeto%20bedrock/semantic-search-aws/src/semantic_search/math_utils.py): operações vetoriais implementadas do zero para a parte didática.
- [`src/semantic_search/engine.py`](/mnt/c/Users/rafael.santiago/Documents/Projeto%20bedrock/semantic-search-aws/src/semantic_search/engine.py): indexação, validação do corpus e busca.
- [`src/semantic_search/storage.py`](/mnt/c/Users/rafael.santiago/Documents/Projeto%20bedrock/semantic-search-aws/src/semantic_search/storage.py): serialização e restauração do motor.
- [`src/vector_search.py`](/mnt/c/Users/rafael.santiago/Documents/Projeto%20bedrock/semantic-search-aws/src/vector_search.py): camada de compatibilidade para imports antigos.

Documentação complementar:

- [`docs/ARCHITECTURE.md`](/mnt/c/Users/rafael.santiago/Documents/Projeto%20bedrock/semantic-search-aws/docs/ARCHITECTURE.md): visão detalhada da arquitetura e das decisões de projeto.

## Estrutura do repositorio

```text
semantic-search-aws/
├── notebooks/
│   └── 01_busca_semantica.ipynb   # Notebook didático passo a passo
├── docs/
│   └── ARCHITECTURE.md            # Arquitetura, decisões e extensões futuras
├── src/
│   ├── semantic_search/
│   │   ├── engine.py              # Motor de busca e validações
│   │   ├── math_utils.py          # Álgebra linear do zero
│   │   └── storage.py             # Persistência local e S3
│   └── vector_search.py           # Camada de compatibilidade
├── data/
│   └── documentos.json            # Base de dados de exemplo
├── tests/
│   └── test_semantic_search.py    # Casos críticos
├── requirements.txt
└── README.md
```

## Pre-requisitos

- Python 3.10 ou superior
- `pip`
- Credenciais AWS configuradas apenas se você quiser testar persistência no S3

## Instalacao

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Como executar

Notebook:

```bash
jupyter notebook notebooks/01_busca_semantica.ipynb
```

Uso direto pelo módulo:

```bash
python -c "import sys; sys.path.insert(0, 'src'); from semantic_search import MotorBuscaSemantica; m=MotorBuscaSemantica(); m.carregar_documentos('data/documentos.json'); print(m.buscar('aws', top_k=3))"
```

Para usar integração com S3:

```bash
aws configure
```

## Exemplo rapido

```python
from semantic_search import (
    MotorBuscaSemantica,
    salvar_motor_arquivo,
    carregar_motor_arquivo,
)

motor = MotorBuscaSemantica()
motor.carregar_documentos("data/documentos.json")

resultados = motor.buscar("armazenamento de dados aws", top_k=2)
print(resultados)

salvar_motor_arquivo(motor, "artifacts/motor.pkl.gz")
motor_restaurado = carregar_motor_arquivo("artifacts/motor.pkl.gz")

print(motor_restaurado.buscar("armazenamento de dados aws", top_k=2))
```

## Persistencia local e S3

Persistência local:

```python
from semantic_search import salvar_motor_arquivo, carregar_motor_arquivo

salvar_motor_arquivo(motor, "artifacts/motor.pkl.gz")
motor_restaurado = carregar_motor_arquivo("artifacts/motor.pkl.gz")
```

Persistência no S3:

```python
from semantic_search import salvar_motor_s3, carregar_motor_s3

salvar_motor_s3(motor, "meu-bucket", "indices/motor.pkl.gz")
motor_restaurado = carregar_motor_s3("meu-bucket", "indices/motor.pkl.gz")
```

O estado persistido inclui:

- documentos originais
- `vectorizer` treinado
- matriz TF-IDF

Isso garante que o motor restaurado consiga responder buscas sem reindexar o corpus.

## Testes

Os testes automatizados estão em [`tests/test_semantic_search.py`](/mnt/c/Users/rafael.santiago/Documents/Projeto%20bedrock/semantic-search-aws/tests/test_semantic_search.py) e cobrem:

- erro ao buscar sem índice carregado
- validação de `top_k`
- query fora do vocabulário
- restauração do motor persistido
- limite de resultados quando `top_k` excede o total de documentos

Execução:

```bash
pytest
```

## Tecnologias

- Python 3.10+
- NumPy para operações com vetores
- scikit-learn para vetorização TF-IDF e similaridade de cosseno
- boto3 para integração com AWS S3
- pytest para validação dos casos críticos
- Jupyter Notebook para exploração didática

## Conceitos de algebra linear aplicados

| Conceito | Onde é usado |
|---|---|
| Norma de vetor | Normalização dos documentos |
| Produto escalar | Base conceitual da comparação vetorial |
| Similaridade de cosseno | Ranqueamento de resultados |
| Matriz de vetores | Estrutura do índice de busca |

## Decisoes tecnicas

- A matriz TF-IDF é mantida esparsa para evitar custo desnecessário de memória.
- A camada de persistência salva o estado completo do motor, e não apenas a matriz.
- A implementação didática de álgebra linear foi separada do motor operacional.
- O arquivo [`src/vector_search.py`](/mnt/c/Users/rafael.santiago/Documents/Projeto%20bedrock/semantic-search-aws/src/vector_search.py) preserva compatibilidade com versões anteriores do projeto.

## Limitacoes atuais

- O corpus ainda é pequeno e carregado integralmente em memória.
- A busca usa TF-IDF, que não captura semântica profunda como embeddings densos.
- A persistência usa `pickle`, adequada para estudo e prototipagem, mas não ideal como formato aberto de intercâmbio.
- Ainda não há pipeline de CI/CD nem exposição como serviço HTTP.

## Roadmap

- [ ] Substituir TF-IDF por embeddings reais com AWS Bedrock Titan
- [ ] Expor o motor como API com Lambda + API Gateway ou FastAPI
- [ ] Armazenar índice em Amazon OpenSearch
- [ ] Adicionar pipeline de atualização com Step Functions
- [ ] Incluir CI para testes automáticos em push e pull request

## Correcoes estruturais aplicadas

- Persistência corrigida: agora salva e restaura `documentos`, `vectorizer` e `matriz_tfidf`
- Busca corrigida: valida `query` e `top_k` antes de processar
- Query fora do vocabulário: retorna lista vazia em vez de ranqueamento arbitrário
- Escalabilidade melhorada: a matriz TF-IDF permanece esparsa em vez de ser densificada
- Compatibilidade mantida: imports antigos por `src/vector_search.py` continuam funcionando

---

Projeto desenvolvido como portfólio de MLOps/GenAI.
