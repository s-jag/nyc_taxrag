"""
Type definitions for the RAG pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.vectorstore import SearchResult


@dataclass
class RetrievalResult:
    """Result from the retrieval stage of the pipeline."""

    chunks: list[SearchResult]
    query: str
    retrieval_mode: str  # "dense" | "hybrid"
    filters_applied: dict[str, Any] = field(default_factory=dict)
    expanded_refs: list[str] = field(default_factory=list)  # Sections added via cross-ref
    retrieval_time_ms: float = 0.0

    @property
    def total_chunks(self) -> int:
        """Total number of chunks retrieved."""
        return len(self.chunks)

    @property
    def primary_chunks(self) -> list[SearchResult]:
        """Chunks from primary retrieval (score > 0)."""
        return [c for c in self.chunks if c.score > 0]

    @property
    def expanded_chunks(self) -> list[SearchResult]:
        """Chunks from cross-reference expansion (score == 0)."""
        return [c for c in self.chunks if c.score == 0]


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""

    answer: str
    sources: list[SearchResult]
    query: str
    context_tokens: int
    retrieval_time_ms: float
    generation_time_ms: float
    model: str
    retrieval_mode: str

    @property
    def total_time_ms(self) -> float:
        """Total pipeline execution time."""
        return self.retrieval_time_ms + self.generation_time_ms

    @property
    def source_sections(self) -> list[str]:
        """List of unique section numbers used as sources."""
        sections = []
        seen = set()
        for source in self.sources:
            if source.section_number and source.section_number not in seen:
                sections.append(source.section_number)
                seen.add(source.section_number)
        return sections

    def format_sources(self) -> str:
        """Format sources for display."""
        if not self.sources:
            return "No sources."

        lines = []
        for i, source in enumerate(self.sources, 1):
            section = source.section_number or "Unknown"
            chunk_type = source.chunk_type or "text"
            score = f"{source.score:.3f}" if source.score > 0 else "ref"
            lines.append(f"[{i}] § {section} ({chunk_type}) - score: {score}")
        return "\n".join(lines)
