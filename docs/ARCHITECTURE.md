# Arquitetura do Projeto

## Objetivo

Este projeto demonstra como um mecanismo simples de busca semântica pode ser
construído com conceitos clássicos de álgebra linear e componentes amplamente
usados no ecossistema Python.

Ele foi estruturado em duas camadas:

- camada didática: operações vetoriais implementadas manualmente
- camada operacional: indexação TF-IDF, ranqueamento, persistência e testes

## Visão dos módulos

### `src/semantic_search/math_utils.py`

Contém funções de apoio para:

- norma euclidiana
- produto escalar
- similaridade de cosseno

Esse módulo existe para explicar a matemática por trás da busca, sem depender
diretamente de utilitários prontos do ecossistema.

### `src/semantic_search/engine.py`

Contém a classe `MotorBuscaSemantica`, responsável por:

- carregar e validar o corpus JSON
- ajustar o `TfidfVectorizer`
- armazenar o índice TF-IDF
- executar buscas por similaridade
- exportar e restaurar o estado do motor

Principais decisões:

- a matriz TF-IDF permanece esparsa
- entradas inválidas geram erro explícito
- queries fora do vocabulário retornam lista vazia

### `src/semantic_search/storage.py`

Contém as rotinas de persistência do estado completo do motor.

O payload persistido contém:

- documentos
- vectorizer treinado
- matriz TF-IDF

Sem esse conjunto, o motor não pode ser restaurado corretamente.

### `src/vector_search.py`

Mantém compatibilidade com a API anterior do projeto. Isso evita quebrar
notebooks, scripts ou imports antigos que já dependiam desse arquivo.

## Fluxo de dados

1. O usuário carrega o corpus via `carregar_documentos()`.
2. O motor valida a estrutura dos documentos.
3. O texto é convertido em matriz TF-IDF.
4. A query é vetorizada no mesmo espaço do corpus.
5. A similaridade de cosseno ranqueia os documentos.
6. Opcionalmente, o motor é persistido localmente ou no S3.

## Motivações da reestruturação

Antes da reorganização, havia três problemas centrais:

1. Persistência incompleta: apenas a matriz era salva.
2. Acoplamento alto: álgebra linear, busca e S3 estavam no mesmo arquivo.
3. Falta de guardrails: entradas inválidas produziam comportamento implícito.

Com a nova estrutura, cada responsabilidade ficou isolada e o projeto passou a
ser mais fácil de evoluir, testar e apresentar.

## Extensões futuras recomendadas

- trocar TF-IDF por embeddings do AWS Bedrock
- expor o motor por FastAPI ou Lambda
- adicionar CI para rodar testes em push e pull request
- substituir `pickle` por uma estratégia de persistência mais auditável
- adicionar benchmarks de qualidade e latência
