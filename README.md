# NYC TaxRAG

A terminal-based RAG (Retrieval-Augmented Generation) system for querying NYC Tax Law (Title 11: Taxation and Finance).

## Overview

This system enables accurate question-answering over NYC tax law by combining:
- **RAG Pipeline**: Semantic search over chunked documents with LLM-powered response generation
- **Fallback System**: Compressed prompt for simple queries or when retrieval fails
- **Custom Metrics**: Evaluation framework for measuring system quality

## Document Statistics

| Metric | Value |
|--------|-------|
| Source | NYC Administrative Code Title 11 |
| Chapters | 43 |
| Sections | 746 (§ 11-XXX format) |
| Words | ~663,000 |
| Tokens | ~1,000,000 |

## Architecture

```
User Query (Terminal)
       │
       ▼
   Query Router ─────────────────┐
       │                         │
       ▼                         ▼
   RAG Pipeline            Fallback System
       │                         │
       ▼                         ▼
   Retriever              Compressed Prompt
   (Vector Search)        (~50K token summary)
       │                         │
       ▼                         │
   Context Builder               │
       │                         │
       └─────────┬───────────────┘
                 ▼
            LLM (Claude)
                 │
                 ▼
         Response + Citations
```

### RAG Pipeline

1. **Query Processing**: Normalize query, detect section references (11-XXX)
2. **Retrieval**: Semantic search via embeddings → retrieve top-K chunks
3. **Context Building**: Aggregate chunks with metadata
4. **Generation**: Claude API call with retrieved context
5. **Citation**: Include source sections in response

### Fallback System

- Pre-generated compressed prompt (~30-50K tokens)
- Contains chapter summaries, key definitions, common procedures
- Triggered when retrieval confidence is low or for general queries
- Direct LLM call with full compressed context

## Project Structure

```
nyc_taxrag/
├── data/
│   ├── raw/                     # Source documents (HTML, TXT)
│   ├── processed/               # Chunks, embeddings, index
│   └── analysis/                # Statistics and reports
├── src/
│   ├── analysis/                # Document analysis
│   ├── chunking/                # Document chunking [Black Box]
│   ├── embedding/               # Embedding generation
│   ├── vectorstore/             # Vector storage
│   ├── retrieval/               # Search and retrieval
│   ├── rag/                     # RAG pipeline
│   ├── fallback/                # Fallback system
│   ├── metrics/                 # Evaluation [Black Box]
│   ├── llm/                     # LLM client
│   └── config/                  # Configuration
├── cli/                         # Terminal interface
├── scripts/                     # Utility scripts
├── deploy/                      # Cloud deployment
└── tests/                       # Test suite
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/nyc_taxrag.git
cd nyc_taxrag

# Install dependencies (using pip)
pip install -r requirements.txt

# Or using Poetry
poetry install

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Set the following environment variables in `.env`:

```bash
NYCTAX_ANTHROPIC_API_KEY=your-anthropic-api-key
NYCTAX_OPENAI_API_KEY=your-openai-api-key  # For embeddings
```

### Usage

```bash
# Generate document statistics
python scripts/generate_basic_stats.py

# Ingest documents (chunk and index)
nyctax ingest --source data/raw/

# Query the system
nyctax query "What are the requirements for real property assessment?"

# Query with specific mode
nyctax query "What is section 11-201?" --mode rag
nyctax query "What is the structure of NYC tax law?" --mode fallback

# Run evaluation
nyctax evaluate --test-set tests/fixtures/test_queries.json
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `nyctax analyze --stats` | Generate document statistics |
| `nyctax ingest --source <path>` | Chunk and index documents |
| `nyctax query "<question>"` | Query the system (auto mode) |
| `nyctax query "<question>" --mode [rag\|fallback]` | Query with specific mode |
| `nyctax evaluate --test-set <path>` | Run evaluation metrics |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| CLI | Typer |
| Configuration | pydantic-settings |
| LLM | Anthropic Claude |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB (local) / Pinecone (cloud) |
| Cloud Deploy | AWS Lambda + API Gateway |

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy src/
```

## Black Box Components

The following components have interfaces defined but implementation is TBD:

### Chunking Algorithm (`src/chunking/`)

- Placeholder: Simple character-based chunking
- Planned: Section-aware chunking that respects § 11-XXX boundaries
- Requirements:
  - Preserve hierarchy metadata (Chapter → Subchapter → Section)
  - Target 500-1000 tokens per chunk
  - Handle cross-references

### Metrics Evaluator (`src/metrics/`)

- Placeholder: Basic retrieval and response metrics
- Planned metrics:
  - Retrieval: Precision@K, Recall@K, MRR, NDCG
  - Response: Relevance, citation accuracy, factual correctness
  - Latency: E2E time, retrieval time, generation time

## License

MIT
