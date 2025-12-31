"""NYC Tax Law document chunking module."""

from .chunker import Chunker, Chunk, ChunkMetadata
from .html_parser import NYCTaxLawHTMLParser, ParsedDocument, ParsedSection
from .strategies import (
    ChunkingStrategy,
    ZChunkStrategy,
    FallbackChunkingStrategy,
)

__all__ = [
    # Main interface
    "Chunker",
    "Chunk",
    "ChunkMetadata",
    # HTML parsing
    "NYCTaxLawHTMLParser",
    "ParsedDocument",
    "ParsedSection",
    # Strategies
    "ChunkingStrategy",
    "ZChunkStrategy",
    "FallbackChunkingStrategy",
]
