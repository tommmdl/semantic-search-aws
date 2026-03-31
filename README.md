# Busca Semântica com Álgebra Linear e AWS S3

Projeto de portfólio que demonstra como conceitos fundamentais de álgebra linear
(produto escalar, norma de vetores e similaridade de cosseno) são aplicados em
sistemas reais de busca semântica — a base de tecnologias como RAG, recomendação
e pesquisa inteligente.

## O que este projeto demonstra

- Implementação de similaridade de cosseno do zero com NumPy
- Comparação com implementações otimizadas (scikit-learn)
- Pipeline completo: texto → vetor TF-IDF → busca por similaridade
- Integração com AWS S3 para persistir e carregar o índice de vetores
- Boas práticas: docstrings, tratamento de erros, logging

## Contexto técnico

A busca semântica é o coração de sistemas RAG (Retrieval-Augmented Generation),
recomendadores de produtos e motores de busca modernos. Este projeto implementa
a versão fundamental desse mecanismo usando apenas álgebra linear clássica.

## Estrutura

```
semantic-search-aws/
├── notebooks/
│   └── 01_busca_semantica.ipynb   # Notebook didático passo a passo
├── src/
│   └── vector_search.py           # Módulo reutilizável
├── data/
│   └── documentos.json            # Base de dados de exemplo
├── requirements.txt
└── README.md
```

## Como executar

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_busca_semantica.ipynb
```

Para usar a integração com S3, configure suas credenciais AWS:
```bash
aws configure
```

## Tecnologias

- Python 3.10+
- NumPy — operações com vetores
- scikit-learn — vetorização TF-IDF
- boto3 — integração AWS S3
- Jupyter Notebook

## Conceitos de álgebra linear aplicados

| Conceito | Onde é usado |
|---|---|
| Norma de vetor | Normalização dos documentos |
| Produto escalar | Cálculo de similaridade |
| Similaridade de cosseno | Ranqueamento de resultados |
| Matriz de vetores | Índice de busca |

## Próximos passos (roadmap)

- [ ] Substituir TF-IDF por embeddings reais (AWS Bedrock Titan)
- [ ] Deploy como API Lambda + API Gateway
- [ ] Armazenar índice em Amazon OpenSearch
- [ ] Adicionar pipeline de atualização com Step Functions

---
*Projeto desenvolvido como portfólio de MLOps/GenAI — Rafael*
