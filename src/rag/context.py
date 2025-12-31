"""
Context assembly for the RAG pipeline.

Handles deduplication, token budgeting, and formatting of retrieved chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.vectorstore import SearchResult


@dataclass
class AssembledContext:
    """Result of context assembly."""

    context_text: str
    chunks_used: list[SearchResult]
    total_tokens: int
    chunks_truncated: int  # Number of chunks that didn't fit


class ContextAssembler:
    """
    Assembles context from search results with deduplication and token budgeting.

    Features:
    - Deduplication by chunk_id
    - Deterministic ordering for reproducibility
    - Token budget enforcement
    - Structured context formatting
    """

    def __init__(self, max_tokens: int = 8000):
        """
        Initialize the context assembler.

        Args:
            max_tokens: Maximum tokens for the assembled context.
        """
        self.max_tokens = max_tokens

    def assemble(
        self,
        primary_results: list[SearchResult],
        expanded_results: list[SearchResult] | None = None,
        deduplicate: bool = True,
    ) -> AssembledContext:
        """
        Assemble context from search results.

        Args:
            primary_results: Primary search results (from direct retrieval).
            expanded_results: Optional expanded results (from cross-ref expansion).
            deduplicate: Whether to remove duplicate chunks.

        Returns:
            AssembledContext with formatted text and metadata.
        """
        # Combine results
        all_results = list(primary_results)
        if expanded_results:
            all_results.extend(expanded_results)

        # Deduplicate if requested
        if deduplicate:
            all_results = self._deduplicate(all_results)

        # Sort for deterministic ordering
        all_results = self._sort_results(all_results)

        # Accumulate chunks within token budget
        chunks_used: list[SearchResult] = []
        total_tokens = 0
        chunks_truncated = 0

        for result in all_results:
            chunk_tokens = result.token_count or self._estimate_tokens(result.text)

            if total_tokens + chunk_tokens <= self.max_tokens:
                chunks_used.append(result)
                total_tokens += chunk_tokens
            else:
                chunks_truncated += 1

        # Format context
        context_text = self._format_context(chunks_used)

        return AssembledContext(
            context_text=context_text,
            chunks_used=chunks_used,
            total_tokens=total_tokens,
            chunks_truncated=chunks_truncated,
        )

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Remove duplicate chunks, keeping the highest-scored version.

        Args:
            results: List of search results.

        Returns:
            Deduplicated list.
        """
        seen: dict[str, SearchResult] = {}

        for result in results:
            chunk_id = result.chunk_id
            if chunk_id not in seen or result.score > seen[chunk_id].score:
                seen[chunk_id] = result

        return list(seen.values())

    def _sort_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Sort results for deterministic ordering.

        Order: (is_primary desc, score desc, section_number asc)
        Primary results (score > 0) come first, then by score, then by section.

        Args:
            results: List of search results.

        Returns:
            Sorted list.
        """
        def sort_key(r: SearchResult) -> tuple:
            is_primary = 1 if r.score > 0 else 0
            # Negative score for descending order
            score = -r.score if r.score else 0
            section = r.section_number or ""
            return (-is_primary, score, section)

        return sorted(results, key=sort_key)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses a simple heuristic of ~4 chars per token.

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count.
        """
        return len(text) // 4

    def _format_context(self, chunks: list[SearchResult]) -> str:
        """
        Format chunks into a structured context string.

        Args:
            chunks: List of chunks to format.

        Returns:
            Formatted context string.
        """
        if not chunks:
            return ""

        sections = []
        for i, chunk in enumerate(chunks, 1):
            header = self._format_chunk_header(chunk, i)
            sections.append(f"{header}\n{chunk.text}")

        return "\n\n---\n\n".join(sections)

    def _format_chunk_header(self, chunk: SearchResult, index: int) -> str:
        """
        Format the header for a single chunk.

        Args:
            chunk: The chunk to format.
            index: Index in the context.

        Returns:
            Formatted header string.
        """
        parts = [f"[Source {index}]"]

        if chunk.section_number:
            parts.append(f"§ {chunk.section_number}")

        if chunk.chapter:
            parts.append(f"Chapter {chunk.chapter_number}: {chunk.chapter}")
        elif chunk.chapter_number:
            parts.append(f"Chapter {chunk.chapter_number}")

        if chunk.chunk_type and chunk.chunk_type != "unknown":
            parts.append(f"({chunk.chunk_type})")

        return " ".join(parts)
