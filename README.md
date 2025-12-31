# NYC TaxRAG

Terminal-based RAG system for NYC Tax Law (Title 11: Taxation and Finance).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY=your-openai-key
export ANTHROPIC_API_KEY=your-anthropic-key

# Start Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant
```

## Testing Components

### 1. Chunking

```bash
# Run chunking on source document
python scripts/run_chunking.py

# Output: data/processed/chunks/chunks.json (3,498 chunks)
```

### 2. Ingestion (Embeddings + Vector Store)

```bash
# Ingest chunks into Qdrant
python scripts/run_ingestion.py --recreate

# Uses: OpenAI text-embedding-3-small + BM25 sparse vectors
# Output: Qdrant collection "nyc_tax_law"
```

### 3. Search

```bash
# Test search functionality
python scripts/test_search.py --query "property tax assessment"

# Filter by chapter
python scripts/test_search.py --query "exemptions" --chapter 2

# Filter by section
python scripts/test_search.py --section "11-201"
```

## Architecture

```
Source HTML → Chunking → Embeddings → Qdrant → Hybrid Search → LLM → Response
                ↓
         3,498 chunks
         746 sections
         ~1M tokens
```

### Search Pipeline

| Stage | Description |
|-------|-------------|
| Dense | OpenAI text-embedding-3-small (1536 dims) |
| Sparse | BM25 via fastembed |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Expansion | Cross-reference citation expansion |

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
│   ├── run_chunking.py         # Chunk documents
│   ├── run_ingestion.py        # Embed + upsert
│   └── test_search.py          # Test queries
└── docs/
    ├── chunking.md             # Chunking details
    └── vectorstore.md          # Vector store details
```

## Configuration

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | For embeddings |
| `ANTHROPIC_API_KEY` | For LLM generation |

Qdrant runs locally on `localhost:6333` by default.

## Stats

| Metric | Value |
|--------|-------|
| Chunks | 3,498 |
| Sections | 746 |
| Tokens | ~1.1M |
| Avg chunk | 328 tokens |

## Docs

- [Chunking](docs/chunking.md) - zChunk algorithm details
- [Vector Store](docs/vectorstore.md) - Qdrant hybrid search
