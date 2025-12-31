"""
Configuration for the RAG pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RAGConfig:
    """
    Configuration for the RAG pipeline.

    Attributes:
        retrieval_mode: Search mode - "dense" (semantic) or "hybrid" (dense + sparse).
        top_k: Number of chunks to retrieve.
        section_filter: Filter by specific section number.
        chapter_filter: Filter by specific chapter number.
        jurisdiction: Filter by jurisdiction (e.g., "NYC").
        doc_type: Filter by document type (e.g., "statute").
        as_of_date: Filter by effective date (ISO format).
        max_context_tokens: Maximum tokens for context assembly.
        expand_cross_refs: Whether to expand cross-references.
        max_expansion_hops: Maximum hops for cross-reference expansion.
        deduplicate: Whether to deduplicate chunks.
        model: LLM model to use for generation.
        temperature: LLM temperature.
        max_tokens: Maximum tokens for LLM response.
    """

    # Retrieval settings
    retrieval_mode: str = "dense"  # "dense" | "hybrid"
    top_k: int = 10

    # Filtering
    section_filter: str | None = None
    chapter_filter: int | None = None
    jurisdiction: str | None = None
    doc_type: str | None = None
    as_of_date: str | None = None

    # Context assembly
    max_context_tokens: int = 8000
    expand_cross_refs: bool = True
    max_expansion_hops: int = 1

    # Deduplication
    deduplicate: bool = True

    # Generation
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.retrieval_mode not in ("dense", "hybrid"):
            raise ValueError(f"Invalid retrieval_mode: {self.retrieval_mode}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.max_context_tokens < 100:
            raise ValueError(f"max_context_tokens too small: {self.max_context_tokens}")
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"temperature must be 0-2, got {self.temperature}")
