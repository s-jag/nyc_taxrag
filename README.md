# NYC TaxRAG

Terminal-based RAG system for NYC Tax Law (Title 11: Taxation and Finance).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

```bash
# Required
export OPENAI_API_KEY=your-openai-key

# Qdrant Cloud (recommended)
export QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
export QDRANT_API_KEY=your-qdrant-api-key

# Or local Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

## Testing Components

### 1. Chunking

```bash
python scripts/run_chunking.py
# Output: data/processed/chunks/chunks.json (3,498 chunks)
```

### 2. Ingestion

```bash
python scripts/run_ingestion.py --recreate
# Embeds chunks and uploads to Qdrant
```

### 3. Search

```bash
python scripts/test_search.py --query "property tax assessment"
python scripts/test_search.py --query "exemptions" --chapter 2
python scripts/test_search.py --section "11-201"
```

## Architecture

```
Source HTML → Chunking → Embeddings → Qdrant Cloud → Hybrid Search → LLM
                ↓
         3,498 chunks
         746 sections
```

| Stage | Technology |
|-------|------------|
| Embeddings | OpenAI text-embedding-3-small (1536 dims) |
| Sparse | BM25 via fastembed |
| Vector Store | Qdrant Cloud |
| Fusion | Reciprocal Rank Fusion (RRF) |

## Project Structure

```
nyc_taxrag/
├── data/
│   ├── raw/                    # Source HTML/TXT
│   └── processed/chunks/       # Chunked JSON
├── src/
│   ├── chunking/               # zChunk algorithm
│   ├── embedding/              # OpenAI embeddings
│   └── vectorstore/            # Qdrant + hybrid search
├── scripts/
│   ├── run_chunking.py
│   ├── run_ingestion.py
│   └── test_search.py
└── docs/
    ├── chunking.md
    └── vectorstore.md
```

## Stats

| Metric | Value |
|--------|-------|
| Chunks | 3,498 |
| Sections | 746 |
| Tokens | ~1.1M |
| Avg chunk | 328 tokens |

## Docs

- [Chunking](docs/chunking.md)
- [Vector Store](docs/vectorstore.md)
