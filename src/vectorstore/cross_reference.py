"""
Cross-reference expansion for NYC Tax Law RAG.

Legal documents frequently reference other sections. This module
extracts citations and expands search results to include
referenced sections (2-hop retrieval).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .qdrant_store import QdrantStore, SearchResult


# Citation patterns for NYC Tax Law
SECTION_PATTERN = re.compile(r"§\s*(11-[\d.]+)")
CHAPTER_PATTERN = re.compile(r"chapter\s+(\d+)", re.IGNORECASE)
SUBDIVISION_PATTERN = re.compile(r"subdivision\s+\(?([a-z])\)?", re.IGNORECASE)


@dataclass
class CitationExpansionConfig:
    """Configuration for citation expansion."""

    max_hops: int = 1  # Number of expansion hops
    max_citations_per_result: int = 5  # Max citations to follow per result
    max_expanded_results: int = 10  # Max total expanded results
    include_original: bool = True  # Include original results in output


def extract_section_citations(text: str) -> list[str]:
    """
    Extract section references from text.

    Args:
        text: Text to extract citations from.

    Returns:
        List of section numbers (e.g., ["11-201", "11-202"]).
    """
    return SECTION_PATTERN.findall(text)


def extract_all_citations(text: str) -> dict[str, list[str]]:
    """
    Extract all types of citations from text.

    Args:
        text: Text to extract citations from.

    Returns:
        Dictionary with citation types and their values.
    """
    return {
        "sections": SECTION_PATTERN.findall(text),
        "chapters": CHAPTER_PATTERN.findall(text),
        "subdivisions": SUBDIVISION_PATTERN.findall(text),
    }


class CrossReferenceExpander:
    """
    Expands search results by following cross-references.

    When a chunk references other sections (e.g., "pursuant to § 11-201"),
    this expander fetches those referenced sections to provide more context.
    """

    def __init__(
        self,
        store: QdrantStore,
        config: CitationExpansionConfig | None = None,
    ):
        """
        Initialize the cross-reference expander.

        Args:
            store: QdrantStore instance for fetching chunks.
            config: Expansion configuration.
        """
        self.store = store
        self.config = config or CitationExpansionConfig()

    def expand_results(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Expand search results by following cross-references.

        Args:
            results: Original search results.

        Returns:
            Expanded list including referenced sections.
        """
        if not results:
            return results

        # Collect all cited sections from results
        cited_sections: set[str] = set()
        original_sections: set[str] = set()

        for result in results:
            # Track original sections to avoid duplicates
            if result.section_number:
                original_sections.add(result.section_number)

            # Extract citations from text
            citations = extract_section_citations(result.text)
            for citation in citations[: self.config.max_citations_per_result]:
                cited_sections.add(citation)

        # Remove sections we already have
        sections_to_fetch = cited_sections - original_sections

        if not sections_to_fetch:
            return results

        # Fetch cited sections
        expanded = self.store.search_by_sections(
            list(sections_to_fetch)[: self.config.max_expanded_results],
            limit_per_section=1,
        )

        # Combine results
        if self.config.include_original:
            # Mark expanded results with lower score
            for r in expanded:
                r.score = 0.0  # Indicate these are expansion results
            return results + expanded
        else:
            return expanded

    def expand_with_hops(
        self,
        results: list[SearchResult],
        hops: int | None = None,
    ) -> list[SearchResult]:
        """
        Expand results with multiple hops.

        Args:
            results: Original search results.
            hops: Number of expansion hops (default: config.max_hops).

        Returns:
            Expanded results after all hops.
        """
        hops = hops or self.config.max_hops
        current_results = results
        seen_sections: set[str] = set()

        # Track seen sections from original results
        for result in results:
            if result.section_number:
                seen_sections.add(result.section_number)

        for hop in range(hops):
            # Collect new citations
            new_citations: set[str] = set()

            for result in current_results:
                citations = extract_section_citations(result.text)
                for citation in citations:
                    if citation not in seen_sections:
                        new_citations.add(citation)

            if not new_citations:
                break

            # Fetch new sections
            new_results = self.store.search_by_sections(
                list(new_citations)[: self.config.max_expanded_results],
                limit_per_section=1,
            )

            # Update seen sections
            for result in new_results:
                if result.section_number:
                    seen_sections.add(result.section_number)

            # Add to results
            current_results = current_results + new_results

        return current_results

    def get_citation_graph(
        self,
        results: list[SearchResult],
    ) -> dict[str, list[str]]:
        """
        Build a citation graph from search results.

        Args:
            results: Search results to analyze.

        Returns:
            Dictionary mapping section numbers to their citations.
        """
        graph: dict[str, list[str]] = {}

        for result in results:
            if result.section_number:
                citations = extract_section_citations(result.text)
                # Filter out self-references
                citations = [c for c in citations if c != result.section_number]
                if citations:
                    graph[result.section_number] = citations

        return graph
