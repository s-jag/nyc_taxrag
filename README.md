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

# With metadata options
python scripts/run_ingestion.py --recreate --jurisdiction NYC --doc-type statute
```

### 3. Search

```bash
python scripts/test_search.py --query "property tax assessment"
python scripts/test_search.py --query "exemptions" --chapter 2
python scripts/test_search.py --section "11-201"
```

### 4. RAG Pipeline

```bash
# Full RAG query with answer generation
python scripts/test_rag.py --query "What are the property tax exemptions?"

# Hybrid retrieval mode
python scripts/test_rag.py --query "assessment process" --mode hybrid

# Retrieve only (no LLM generation)
python scripts/test_rag.py --query "commissioner powers" --retrieve-only

# Custom settings
python scripts/test_rag.py --query "tax rates" --top-k 10 --no-expand

# Fallback options (for low-confidence queries)
python scripts/test_rag.py --query "obscure topic" --fallback-threshold 0.5
python scripts/test_rag.py --query "query" --no-fallback
python scripts/test_rag.py --query "query" --fallback-model o3
```

### 5. Unit Tests

```bash
python -m pytest tests/unit/test_rag_pipeline.py -v
```

## Architecture

```
Source HTML → Chunking → Embeddings → Qdrant Cloud → Hybrid Search → LLM
                ↓                           ↓               ↓
         3,498 chunks              Cross-Ref Expansion   GPT-4o
         746 sections              Context Assembly      Answer
                                          ↓
                                   Low Confidence?
                                          ↓
                                   Fallback → O3-mini
                                   (LAW PACK prompt)
```

| Stage | Technology |
|-------|------------|
| Embeddings | OpenAI text-embedding-3-small (1536 dims) |
| Sparse | BM25 via fastembed |
| Vector Store | Qdrant Cloud |
| Fusion | Reciprocal Rank Fusion (RRF) |
| LLM | OpenAI GPT-4o |
| Fallback | OpenAI O3-mini (reasoning model) |

## RAG Pipeline

The RAG pipeline (`src/rag/pipeline.py`) orchestrates:

1. **Query Embedding** - Converts question to dense vector
2. **Retrieval** - Dense or hybrid search with filtering
3. **Fallback Check** - If top chunk score < 60%, triggers fallback to O3-mini
4. **Cross-Reference Expansion** - Fetches cited sections
5. **Context Assembly** - Deduplication and token budgeting
6. **Generation** - GPT-4o generates answer with citations

### Python API

```python
from src.rag import RAGPipeline, RAGConfig
from src.embedding import OpenAIEmbedder
from src.vectorstore import QdrantStore
from src.llm.providers import OpenAIClient

# Initialize pipeline
pipeline = RAGPipeline(
    embedder=OpenAIEmbedder(),
    store=QdrantStore(),
    llm=OpenAIClient(),
    config=RAGConfig(
        retrieval_mode="hybrid",  # or "dense"
        top_k=10,
        expand_cross_refs=True,
        max_context_tokens=8000,
    ),
)

# Query
response = pipeline.query("What is the property tax rate?")
print(response.answer)
print(response.format_sources())
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `retrieval_mode` | `"dense"` | `"dense"` or `"hybrid"` |
| `top_k` | `10` | Number of chunks to retrieve |
| `expand_cross_refs` | `True` | Expand cited sections |
| `max_context_tokens` | `8000` | Token budget for context |
| `deduplicate` | `True` | Remove duplicate chunks |
| `temperature` | `0.1` | LLM temperature |
| `model` | `"gpt-4o"` | OpenAI model |
| `enable_fallback` | `True` | Enable fallback for low-confidence results |
| `fallback_threshold` | `0.6` | Score threshold to trigger fallback |
| `fallback_model` | `"o3-mini"` | OpenAI reasoning model for fallback |

## Project Structure

```
nyc_taxrag/
├── data/
│   ├── raw/                    # Source HTML/TXT
│   └── processed/chunks/       # Chunked JSON
├── src/
│   ├── chunking/               # zChunk algorithm
│   ├── embedding/              # OpenAI embeddings
│   ├── vectorstore/            # Qdrant + hybrid search
│   ├── rag/                    # RAG pipeline
│   │   ├── pipeline.py         # Main pipeline class
│   │   ├── context.py          # Context assembly
│   │   ├── config.py           # Configuration
│   │   └── types.py            # Type definitions
│   ├── fallback/               # Fallback system
│   │   ├── config.py           # FallbackConfig
│   │   ├── handler.py          # FallbackHandler
│   │   ├── prompt.py           # FallbackPromptBuilder
│   │   └── types.py            # FallbackResponse
│   └── llm/                    # LLM clients
│       ├── client.py           # Abstract interface
│       └── providers/          # Implementations
│           └── openai_client.py
├── scripts/
│   ├── run_chunking.py
│   ├── run_ingestion.py
│   ├── test_search.py
│   └── test_rag.py
├── tests/
│   └── unit/
│       └── test_rag_pipeline.py
├── fallback_prompt.txt         # LAW PACK for fallback
└── docs/
    ├── chunking.md
    ├── vectorstore.md
    └── rag.md
```

## Fallback System

When RAG retrieval returns low-confidence results (top chunk score < 60%), the system automatically falls back to a reasoning model with a comprehensive legal prompt.

### How It Works

1. **Trigger**: Top chunk score below threshold (default: 0.6)
2. **Verbose Output**: Shows retrieved chunks, scores, and explains why fallback triggered
3. **Fallback Prompt**: Loads `fallback_prompt.txt` containing the full NYC Tax Law reference (LAW PACK)
4. **Context Injection**: Low-confidence chunks are prepended as additional context
5. **Reasoning Model**: Sends to OpenAI O3-mini for complex legal reasoning
6. **Response**: Returns structured legal answer with citations

### CLI Output Example

```
LOW CONFIDENCE RETRIEVAL - ACTIVATING FALLBACK

Retrieved Chunks (below 60% threshold):
┌───┬──────────────┬─────────┬──────────────────────────────┐
│ # │ Section      │ Score   │ Preview                      │
├───┼──────────────┼─────────┼──────────────────────────────┤
│ 1 │ § 11-201     │ 0.423   │ Property assessment...       │
│ 2 │ § 11-202     │ 0.387   │ Tax exemptions...            │
└───┴──────────────┴─────────┴──────────────────────────────┘

Fallback Strategy:
  -> Top chunk score: 0.423 (threshold: 60%)
  -> Including 2 low-confidence chunks as context
  -> Using comprehensive NYC Tax Law reference (LAW PACK)
  -> Sending to reasoning model: o3-mini

FALLBACK RESPONSE (o3-mini)
[Generated legal answer with citations]
```

### Disabling Fallback

```bash
# Disable fallback entirely
python scripts/test_rag.py --query "topic" --no-fallback

# Adjust threshold
python scripts/test_rag.py --query "topic" --fallback-threshold 0.4
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
- [RAG Pipeline](docs/rag.md)
