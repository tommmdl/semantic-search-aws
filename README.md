# Busca Semântica com TF-IDF e AWS S3

Motor de busca semântica com **TF-IDF + similaridade de cosseno**, exposto como **API REST com FastAPI** e persistência de estado no **Amazon S3**.

Projeto de portfólio com foco em MLOps/GenAI — base para evolução em RAG, embeddings e busca inteligente.

---

## Arquitetura

```
documentos.json
      │
      ▼
MotorBuscaSemantica
  TF-IDF Vectorizer
  Cosine Similarity
      │
      ├──► Persistência local (.pkl.gz)
      └──► Persistência AWS S3
            │
            ▼
      FastAPI REST API
      POST /search
      POST /index/documents
      POST /index/s3
      GET  /status
```

---

## Stack

| Camada | Tecnologia |
|--------|------------|
| Motor de busca | TF-IDF · Cosine Similarity (NumPy/Scikit-learn) |
| API | FastAPI · Uvicorn · Pydantic |
| Persistência | Pickle + Gzip · AWS S3 (boto3) |
| Containerização | Docker · Docker Compose |
| Testes | Pytest |

---

## Como executar

### Com Docker (recomendado)

```bash
git clone https://github.com/tommmdl/semantic-search-aws.git
cd semantic-search-aws

docker-compose up -d
```

| Serviço | URL |
|---------|-----|
| API (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

### Sem Docker

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

> O índice é carregado automaticamente de `data/documentos.json` na inicialização.

---

## API

### `POST /search` — Busca semântica

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "aprendizado de máquina", "top_k": 3}'
```

**Resposta:**
```json
{
  "query": "aprendizado de máquina",
  "total": 3,
  "resultados": [
    {
      "id": "1",
      "titulo": "Introdução ao Machine Learning",
      "texto": "...",
      "score": 0.8731
    }
  ]
}
```

---

### `POST /index/documents` — Indexar documentos

```bash
curl -X POST http://localhost:8000/index/documents \
  -H "Content-Type: application/json" \
  -d '{
    "documentos": [
      {"id": "1", "titulo": "Título", "texto": "Conteúdo do documento"}
    ]
  }'
```

---

### `POST /index/s3` — Carregar índice do S3

```bash
curl -X POST http://localhost:8000/index/s3 \
  -H "Content-Type: application/json" \
  -d '{"bucket": "meu-bucket", "chave": "indices/motor.pkl.gz"}'
```

---

### `POST /index/s3/save` — Salvar índice no S3

```bash
curl -X POST http://localhost:8000/index/s3/save \
  -H "Content-Type: application/json" \
  -d '{"bucket": "meu-bucket", "chave": "indices/motor.pkl.gz"}'
```

---

### `GET /status` — Status do índice

```json
{
  "indexado": true,
  "total_documentos": 42,
  "total_termos": 1837
}
```

---

## Estrutura do repositório

```
.
├── api/
│   ├── main.py          # FastAPI — rotas e lifespan
│   └── schemas.py       # Pydantic models
├── src/
│   └── semantic_search/
│       ├── engine.py    # MotorBuscaSemantica (TF-IDF + cosseno)
│       ├── math_utils.py
│       └── storage.py   # Persistência local e S3
├── data/
│   └── documentos.json  # Corpus padrão
├── tests/
│   └── test_semantic_search.py
├── notebooks/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Decisões técnicas

- **TF-IDF esparso** — a matriz permanece esparsa para eficiência de memória
- **Gzip + Pickle** — estado serializado e comprimido para persistência eficiente
- **Lifespan FastAPI** — índice carregado automaticamente na inicialização se corpus padrão existir
- **Query fora do vocabulário** — retorna lista vazia em vez de ranqueamento arbitrário

---

## Roadmap

- [x] Motor TF-IDF com similaridade de cosseno
- [x] Persistência local e AWS S3
- [x] API REST com FastAPI
- [ ] Substituir TF-IDF por embeddings com AWS Bedrock Titan
- [ ] Armazenar índice em Amazon OpenSearch
- [ ] Pipeline de atualização com Step Functions
- [ ] CI/CD com GitHub Actions

---

**Autor:** Rafael Santiago — [github.com/tommmdl](https://github.com/tommmdl)
