#!/usr/bin/env python3
"""
Test search script for NYC Tax Law RAG.

Tests various search capabilities:
- Dense (semantic) search
- Sparse (BM25) search
- Hybrid search with RRF fusion
- Filtered search by chapter/section
- Cross-reference expansion

Usage:
    python scripts/test_search.py
    python scripts/test_search.py --query "property tax assessment"
    python scripts/test_search.py --chapter 2
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from src.embedding import OpenAIEmbedder
from src.vectorstore import (
    QdrantStore,
    HybridSearcher,
    CrossReferenceExpander,
    SearchResult,
)


console = Console()


def print_results(results: list[SearchResult], title: str = "Search Results"):
    """Pretty print search results."""
    console.print(f"\n[bold blue]{title}[/] ({len(results)} results)")

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    for i, result in enumerate(results, 1):
        # Truncate text for display
        text_preview = result.text[:200] + "..." if len(result.text) > 200 else result.text

        panel_content = f"""
**Section:** {result.section or 'N/A'}
**Chapter:** {result.chapter_number or 'N/A'}
**Score:** {result.score:.4f}
**Tokens:** {result.token_count}
**Citations:** {', '.join(result.citations[:3]) if result.citations else 'None'}

---

{text_preview}
"""
        console.print(Panel(
            Markdown(panel_content),
            title=f"[bold]#{i} {result.chunk_id}[/bold]",
            border_style="green" if result.score > 0 else "dim",
        ))


def run_tests(
    query: str | None = None,
    chapter_filter: int | None = None,
    section_filter: str | None = None,
):
    """Run search tests."""

    # Initialize components
    console.print("\n[bold blue]Initializing components...[/]")

    try:
        embedder = OpenAIEmbedder()
        console.print(f"[green]Embedder: {embedder.model_name}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to initialize embedder: {e}[/red]")
        return

    try:
        store = QdrantStore()
        info = store.get_info()
        console.print(f"[green]Qdrant: {info['points_count']} points[/green]")
    except Exception as e:
        console.print(f"[red]Failed to connect to Qdrant: {e}[/red]")
        console.print("[yellow]Make sure Qdrant is running and ingestion has been completed[/yellow]")
        return

    hybrid_searcher = HybridSearcher(store.client)
    expander = CrossReferenceExpander(store)

    # Default test queries
    test_queries = [
        "What is the property tax assessment process?",
        "real property tax exemption",
        "commissioner of finance powers",
    ]

    if query:
        test_queries = [query]

    for test_query in test_queries:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Query:[/] {test_query}")
        if chapter_filter:
            console.print(f"[dim]Filter: chapter={chapter_filter}[/dim]")
        if section_filter:
            console.print(f"[dim]Filter: section={section_filter}[/dim]")
        console.print("=" * 60)

        # Generate query embedding
        query_dense = embedder.embed_query(test_query)

        # Generate sparse embedding
        query_sparse = store._create_sparse_vector(test_query)

        # Test 1: Dense search only
        console.print("\n[bold cyan]Test 1: Dense (Semantic) Search[/]")
        results = store.search(
            query_dense,
            limit=3,
            section_filter=section_filter,
            chapter_filter=chapter_filter,
        )
        print_results(results, "Dense Search")

        # Test 2: Hybrid search
        console.print("\n[bold cyan]Test 2: Hybrid Search (Dense + Sparse + RRF)[/]")
        results = hybrid_searcher.search(
            query_dense,
            query_sparse,
            limit=3,
            section_filter=section_filter,
            chapter_filter=chapter_filter,
        )
        print_results(results, "Hybrid Search")

        # Test 3: Hybrid with cross-reference expansion
        console.print("\n[bold cyan]Test 3: Hybrid + Cross-Reference Expansion[/]")
        base_results = hybrid_searcher.search(
            query_dense,
            query_sparse,
            limit=3,
            section_filter=section_filter,
            chapter_filter=chapter_filter,
        )
        expanded = expander.expand_results(base_results)
        print_results(expanded, "Expanded Results")

        # Show citation graph
        if base_results:
            graph = expander.get_citation_graph(expanded)
            if graph:
                console.print("\n[bold cyan]Citation Graph:[/]")
                table = Table()
                table.add_column("Section", style="cyan")
                table.add_column("Cites", style="green")
                for section, citations in graph.items():
                    table.add_row(section, ", ".join(citations[:5]))
                console.print(table)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test NYC Tax Law RAG search")
    parser.add_argument(
        "--query",
        type=str,
        help="Custom query to test",
    )
    parser.add_argument(
        "--chapter",
        type=int,
        help="Filter by chapter number",
    )
    parser.add_argument(
        "--section",
        type=str,
        help="Filter by section number (e.g., '11-201')",
    )

    args = parser.parse_args()

    console.print("\n[bold]NYC Tax Law RAG - Search Test[/bold]")
    console.print("=" * 50)

    run_tests(
        query=args.query,
        chapter_filter=args.chapter,
        section_filter=args.section,
    )


if __name__ == "__main__":
    main()
