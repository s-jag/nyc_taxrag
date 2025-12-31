#!/usr/bin/env python3
"""
Test script for the RAG pipeline.

Usage:
    python scripts/test_rag.py --query "What is property tax?"
    python scripts/test_rag.py --query "exemptions" --mode hybrid
    python scripts/test_rag.py --retrieve-only --query "assessment"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from src.rag import RAGPipeline, RAGConfig
from src.embedding import OpenAIEmbedder
from src.vectorstore import QdrantStore
from src.llm.providers import OpenAIClient


console = Console()


def run_rag_query(
    query: str,
    mode: str = "dense",
    top_k: int = 5,
    expand_refs: bool = True,
    retrieve_only: bool = False,
) -> None:
    """Run a RAG query and display results."""

    console.print(f"\n[bold blue]Query:[/] {query}")
    console.print(f"[dim]Mode: {mode}, Top-K: {top_k}, Expand refs: {expand_refs}[/dim]\n")

    # Initialize components
    console.print("[dim]Initializing pipeline...[/dim]")

    try:
        embedder = OpenAIEmbedder()
        store = QdrantStore()
        llm = OpenAIClient()

        config = RAGConfig(
            retrieval_mode=mode,
            top_k=top_k,
            expand_cross_refs=expand_refs,
            max_context_tokens=6000,
        )

        pipeline = RAGPipeline(
            embedder=embedder,
            store=store,
            llm=llm,
            config=config,
        )
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        return

    if retrieve_only:
        # Retrieval only mode
        console.print("[bold]Running retrieval only...[/bold]\n")
        result = pipeline.retrieve_only(query)

        console.print(f"[green]Retrieved {len(result.chunks)} chunks in {result.retrieval_time_ms:.0f}ms[/green]")

        if result.expanded_refs:
            console.print(f"[dim]Expanded refs: {', '.join(result.expanded_refs)}[/dim]")

        # Show chunks
        table = Table(title="Retrieved Chunks")
        table.add_column("#", style="dim", width=3)
        table.add_column("Section", style="cyan", width=12)
        table.add_column("Score", style="green", width=8)
        table.add_column("Type", style="yellow", width=8)
        table.add_column("Text Preview", style="white")

        for i, chunk in enumerate(result.chunks[:10], 1):
            score = f"{chunk.score:.3f}" if chunk.score > 0 else "ref"
            text_preview = chunk.text[:80].replace("\n", " ") + "..."
            table.add_row(
                str(i),
                chunk.section_number or "N/A",
                score,
                chunk.chunk_type or "N/A",
                text_preview,
            )

        console.print(table)

    else:
        # Full RAG query
        console.print("[bold]Running full RAG pipeline...[/bold]\n")
        response = pipeline.query(query)

        # Display answer
        console.print(Panel(
            Markdown(response.answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
        ))

        # Display sources
        console.print("\n[bold]Sources:[/bold]")
        table = Table(show_header=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Section", style="cyan", width=12)
        table.add_column("Chapter", style="blue", width=20)
        table.add_column("Score", style="green", width=8)

        for i, source in enumerate(response.sources[:8], 1):
            score = f"{source.score:.3f}" if source.score > 0 else "ref"
            chapter = source.chapter[:20] if source.chapter else "N/A"
            table.add_row(
                str(i),
                source.section_number or "N/A",
                chapter,
                score,
            )

        console.print(table)

        # Display stats
        console.print(f"\n[dim]Stats: {response.context_tokens} context tokens, "
                     f"retrieval: {response.retrieval_time_ms:.0f}ms, "
                     f"generation: {response.generation_time_ms:.0f}ms, "
                     f"model: {response.model}[/dim]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test the RAG pipeline")
    parser.add_argument(
        "--query", "-q",
        type=str,
        required=True,
        help="Query to run",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="dense",
        choices=["dense", "hybrid"],
        help="Retrieval mode (default: dense)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Disable cross-reference expansion",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Only retrieve, don't generate answer",
    )

    args = parser.parse_args()

    console.print("\n[bold]NYC Tax Law RAG - Test[/bold]")
    console.print("=" * 50)

    run_rag_query(
        query=args.query,
        mode=args.mode,
        top_k=args.top_k,
        expand_refs=not args.no_expand,
        retrieve_only=args.retrieve_only,
    )


if __name__ == "__main__":
    main()
