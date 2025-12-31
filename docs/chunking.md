# Chunking Architecture

Document chunking system for NYC Tax Law RAG pipeline.

## Overview

The chunking system splits NYC tax law documents into semantic chunks suitable for RAG retrieval. It supports two strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `zchunk` | Claude API-based semantic chunking | Production (recommended) |
| `fallback` | Regex-based pattern matching | Offline / cost savings |

## Algorithm: zChunk

Based on the zChunk algorithm, which uses LLM intelligence to detect semantic boundaries.

### Split Tokens

```
段 (U+6BB5) - "section" in Chinese → Major boundaries (chapters, sections)
顿 (U+987F) - "pause" in Chinese  → Minor boundaries (sentences, paragraphs)
```

These tokens are chosen because:
1. Encoded as single tokens by Claude/tiktoken
2. Do not appear in NYC tax law corpus
3. Have semantic meaning related to text structure

### Process

1. Send document batches to Claude with system prompt
2. Claude returns text with 段/顿 markers at semantic boundaries
3. Parse markers to create chunks with metadata
4. Merge small chunks, split large chunks

## File Structure

```
src/chunking/
├── chunker.py              # Main Chunker class
├── html_parser.py          # HTML structure extraction
├── prompts/
│   ├── system_prompt.py    # System prompt + tokens
│   └── examples.py         # Few-shot examples
└── strategies/
    ├── base.py             # Abstract base class
    ├── zchunk_strategy.py  # Claude API strategy
    └── fallback_strategy.py # Regex strategy
```

## Usage

### Command Line

```bash
# Fallback strategy (offline)
python scripts/run_chunking.py

# zChunk strategy (requires ANTHROPIC_API_KEY)
python scripts/run_chunking.py --strategy zchunk

# Custom parameters
python scripts/run_chunking.py --chunk-size 1500 --overlap 200
```

### Python API

```python
from src.chunking import Chunker

# Initialize
chunker = Chunker(strategy="fallback")  # or "zchunk"

# Process file
chunks = chunker.process_file("data/raw/document.html")

# Save results
chunker.save_chunks(chunks, "data/processed/chunks")
```

## Chunk Structure

```python
@dataclass
class Chunk:
    id: str           # "chunk_00001"
    text: str         # Chunk content
    metadata: ChunkMetadata
    token_count: int  # Estimated tokens

@dataclass
class ChunkMetadata:
    chapter: str | None
    chapter_number: int | None
    subchapter: str | None
    section: str | None         # "§ 11-201"
    section_number: str | None  # "11-201"
    section_title: str | None
    chunk_type: str             # "big" or "small"
    source_file: str | None
```

## Results

Processed `New York City Finance Chapter 1.html`:

| Metric | Value |
|--------|-------|
| Total Chunks | 3,498 |
| Total Tokens | 1,149,104 |
| Avg Tokens/Chunk | 328 |
| Min/Max Tokens | 50 / 2,305 |
| Unique Sections | 746 |
| Strategy | fallback |

### Output Files

```
data/processed/chunks/
├── chunks.json       # All chunks with metadata
└── chunk_stats.json  # Statistics summary
```

### Sample Chunk

```json
{
  "id": "chunk_00000",
  "text": "Title 11: Taxation and Finance\n\nChapter 1: Department of Finance\n\n§ 11-101 Power of department of finance to adopt a seal...",
  "metadata": {
    "section": "§ 11-101",
    "section_number": "11-101",
    "chapter": "Chapter 1: Department of Finance",
    "chunk_type": "big"
  },
  "token_count": 367
}
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1500 | Target chunk size (chars) |
| `chunk_overlap` | 200 | Overlap between chunks (chars) |
| `min_chunk_size` | 200 | Merge chunks smaller than this |
| `max_chunk_size` | 3000 | Split chunks larger than this |

## Dependencies

```
beautifulsoup4  # HTML parsing
lxml            # HTML parser backend
anthropic       # Claude API (zchunk only)
```
