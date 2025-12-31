# Vector Store

Qdrant-based hybrid search for NYC Tax Law RAG.

## Quick Test

```bash
# Set environment variables
export OPENAI_API_KEY=your-key
export QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
export QDRANT_API_KEY=your-qdrant-key

# Ingest chunks
python scripts/run_ingestion.py --recreate

# Test search
python scripts/test_search.py --query "property tax assessment"
```

### Local Qdrant (alternative)

```bash
docker run -p 6333:6333 qdrant/qdrant
# Leave QDRANT_URL and QDRANT_API_KEY unset
```

## Search Types

| Type | Command | Use Case |
|------|---------|----------|
| Hybrid | Default | Best for most queries |
| Dense only | `--dense` | Semantic/conceptual queries |
| Filtered | `--chapter 2` | Narrow by chapter |
| Section lookup | `--section "11-201"` | Direct section fetch |

## Architecture

```
Query → OpenAI Embedding → Qdrant
              ↓
        BM25 Sparse → Qdrant
              ↓
         RRF Fusion
              ↓
      Cross-Reference Expansion
              ↓
         Results (top-K)
```

## Collection Schema

```
Collection: nyc_tax_law
├── Dense vectors: 1536 dims (OpenAI)
├── Sparse vectors: BM25 (fastembed)
└── Payload indexes: section_number, chapter_number
```

### Payload Fields

| Field | Type | Indexed |
|-------|------|---------|
| `chunk_id` | string | - |
| `text` | string | - |
| `section_number` | keyword | Yes |
| `chapter_number` | integer | Yes |
| `chunk_type` | keyword | Yes |
| `citations` | array | - |

## Python API

```python
from src.embedding import OpenAIEmbedder
from src.vectorstore import QdrantStore, HybridSearcher

# Initialize
embedder = OpenAIEmbedder()
store = QdrantStore()
searcher = HybridSearcher(store.client)

# Search
query_vec = embedder.embed_query("property tax")
sparse_vec = store._create_sparse_vector("property tax")
results = searcher.search(query_vec, sparse_vec, limit=10)

# With filter
results = store.search(query_vec, chapter_filter=2)
```

## Files

```
src/vectorstore/
├── collection.py      # Schema + indexes
├── qdrant_store.py    # Main store class
├── hybrid_search.py   # RRF fusion
└── cross_reference.py # Citation expansion
```

## Dependencies

```
qdrant-client>=1.12.0
fastembed>=0.4.0
openai>=1.50.0
```
