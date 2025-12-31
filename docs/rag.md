# RAG Pipeline

Full Retrieval-Augmented Generation pipeline for NYC Tax Law.

## Quick Start

```bash
# Set environment variables
export OPENAI_API_KEY=your-key
export QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
export QDRANT_API_KEY=your-qdrant-key

# Run RAG query
python scripts/test_rag.py --query "What are the property tax exemptions?"
```

## CLI Usage

```bash
# Full RAG query
python scripts/test_rag.py --query "What is property tax?"

# Hybrid retrieval (dense + sparse)
python scripts/test_rag.py --query "exemptions" --mode hybrid

# Retrieve only (no LLM)
python scripts/test_rag.py --query "assessment" --retrieve-only

# Custom settings
python scripts/test_rag.py --query "rates" --top-k 10 --no-expand
```

| Flag | Description |
|------|-------------|
| `--query`, `-q` | Query text (required) |
| `--mode`, `-m` | `dense` or `hybrid` (default: dense) |
| `--top-k`, `-k` | Chunks to retrieve (default: 5) |
| `--no-expand` | Disable cross-reference expansion |
| `--retrieve-only` | Skip LLM generation |

## Python API

```python
from src.rag import RAGPipeline, RAGConfig
from src.embedding import OpenAIEmbedder
from src.vectorstore import QdrantStore
from src.llm.providers import OpenAIClient

# Initialize
pipeline = RAGPipeline(
    embedder=OpenAIEmbedder(),
    store=QdrantStore(),
    llm=OpenAIClient(),
    config=RAGConfig(retrieval_mode="hybrid"),
)

# Full RAG query
response = pipeline.query("What is the property tax rate?")
print(response.answer)
print(response.format_sources())

# Retrieve only
result = pipeline.retrieve_only("property tax")
for chunk in result.chunks:
    print(f"§ {chunk.section_number}: {chunk.text[:100]}...")
```

## Pipeline Stages

```
Query → Embed → Retrieve → Expand → Assemble → Generate → Response
         ↓         ↓          ↓          ↓           ↓
      OpenAI    Qdrant   Cross-Ref   Dedupe +    GPT-4o
      1536d     Hybrid    Fetch      Budget
```

### 1. Query Embedding

Converts the user's question to a 1536-dimensional dense vector using OpenAI's `text-embedding-3-small`.

### 2. Retrieval

**Dense mode** (default):
- Cosine similarity search on dense vectors
- Best for semantic/conceptual queries

**Hybrid mode**:
- Dense + BM25 sparse search
- RRF fusion combines results
- Best for queries with specific legal terms

### 3. Cross-Reference Expansion

Legal documents cite other sections. When enabled:
1. Extract `§ 11-XXX` citations from retrieved chunks
2. Fetch those sections from Qdrant
3. Add as context (marked with score=0)

### 4. Context Assembly

Prepares context for LLM with:
- **Deduplication**: Remove duplicate chunks by `chunk_id`
- **Sorting**: Primary results first (by score), then expanded refs
- **Token Budget**: Accumulate chunks until limit reached (default: 8000)
- **Formatting**: Structured context with section headers

### 5. Generation

GPT-4o generates answer with:
- System prompt for legal accuracy
- Instruction to cite sections
- Temperature 0.1 for consistency

## Configuration

```python
from src.rag import RAGConfig

config = RAGConfig(
    # Retrieval
    retrieval_mode="dense",      # "dense" or "hybrid"
    top_k=10,                    # Chunks to retrieve

    # Filtering (applied at retrieval)
    section_filter="11-201",     # Filter by section
    chapter_filter=2,            # Filter by chapter

    # Context
    max_context_tokens=8000,     # Token budget
    expand_cross_refs=True,      # Expand citations
    deduplicate=True,            # Remove duplicates

    # Generation
    model="gpt-4o",
    temperature=0.1,
    max_tokens=2048,
)
```

## Response Types

### RAGResponse

```python
@dataclass
class RAGResponse:
    answer: str                    # Generated answer
    sources: list[SearchResult]    # Chunks used
    query: str                     # Original question
    context_tokens: int            # Tokens in context
    retrieval_time_ms: float       # Retrieval latency
    generation_time_ms: float      # LLM latency
    model: str                     # Model used
    retrieval_mode: str            # dense/hybrid

    # Properties
    total_time_ms: float           # Total latency
    source_sections: list[str]     # Unique sections

    # Methods
    format_sources() -> str        # Pretty-print sources
```

### RetrievalResult

```python
@dataclass
class RetrievalResult:
    chunks: list[SearchResult]     # All retrieved chunks
    query: str                     # Original question
    retrieval_mode: str            # dense/hybrid
    filters_applied: dict          # Active filters
    expanded_refs: list[str]       # Sections from expansion
    retrieval_time_ms: float       # Latency

    # Properties
    total_chunks: int              # Total count
    primary_chunks: list           # Score > 0
    expanded_chunks: list          # Score == 0
```

## Files

```
src/rag/
├── __init__.py          # Exports
├── pipeline.py          # RAGPipeline class
├── context.py           # ContextAssembler
├── config.py            # RAGConfig
└── types.py             # RAGResponse, RetrievalResult

src/llm/
├── __init__.py          # Exports
├── client.py            # Abstract LLMClient
└── providers/
    ├── __init__.py
    └── openai_client.py     # GPT-4o implementation
```

## Dependencies

```
openai>=1.50.0           # Embeddings + LLM
qdrant-client>=1.12.0    # Vector store
fastembed>=0.4.0         # BM25 sparse
```
