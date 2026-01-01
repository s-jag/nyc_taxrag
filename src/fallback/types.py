"""
Type definitions for the fallback system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.vectorstore import SearchResult


@dataclass
class FallbackResponse:
    """Response from the fallback reasoning model."""

    # The generated answer
    answer: str

    # Model used for generation
    model: str

    # Always True for fallback responses
    was_fallback: bool = True

    # Original chunks from RAG retrieval (low confidence)
    original_chunks: list["SearchResult"] = field(default_factory=list)

    # The top chunk score that triggered fallback
    top_chunk_score: float = 0.0

    # Human-readable reason for fallback
    fallback_reason: str = ""

    # Time taken for generation in milliseconds
    generation_time_ms: float = 0.0

    # Original query
    query: str = ""

    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    def format_sources(self) -> str:
        """Format the original (low-confidence) chunks for display."""
        if not self.original_chunks:
            return "No chunks retrieved"

        lines = ["Low-confidence chunks used as context:"]
        for i, chunk in enumerate(self.original_chunks, 1):
            section = chunk.section_number or "Unknown"
            score = chunk.score
            preview = chunk.text[:60].replace("\n", " ") + "..."
            lines.append(f"  [{i}] {section} (score: {score:.3f}): {preview}")

        return "\n".join(lines)
