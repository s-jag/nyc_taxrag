#!/usr/bin/env python3
"""
Ingestion script for NYC Tax Law RAG.

Loads chunks from the processed data directory and upserts them
into Qdrant with dense (OpenAI) and sparse (BM25) embeddings.

Usage:
    python scripts/run_ingestion.py
    python scripts/run_ingestion.py --in-memory  # For testing
    python scripts/run_ingestion.py --recreate   # Delete and recreate collection
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from src.embedding import OpenAIEmbedder
from src.vectorstore import QdrantStore, get_collection_info


console = Console()


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load chunks from JSON file."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("chunks", [])


def run_ingestion(
    chunks_path: Path,
    in_memory: bool = False,
    recreate: bool = False,
    batch_size: int = 50,
) -> dict:
    """
    Run the ingestion pipeline.

    Args:
        chunks_path: Path to chunks.json file.
        in_memory: Use in-memory Qdrant (for testing).
        recreate: Delete and recreate collection.
        batch_size: Batch size for embeddings.

    Returns:
        Statistics dictionary.
    """
    stats = {
        "chunks_loaded": 0,
        "chunks_ingested": 0,
        "total_tokens_embedded": 0,
        "embedding_time_s": 0,
        "ingestion_time_s": 0,
    }

    # Load chunks
    console.print(f"\n[bold blue]Loading chunks from:[/] {chunks_path}")
    chunks = load_chunks(chunks_path)
    stats["chunks_loaded"] = len(chunks)
    console.print(f"[green]Loaded {len(chunks)} chunks[/green]")

    if not chunks:
        console.print("[red]No chunks to ingest![/red]")
        return stats

    # Initialize embedder
    console.print("\n[bold blue]Initializing OpenAI embedder...[/]")
    try:
        embedder = OpenAIEmbedder()
        console.print(f"[green]Using model: {embedder.model_name} ({embedder.dimensions} dims)[/green]")
    except Exception as e:
        console.print(f"[red]Failed to initialize embedder: {e}[/red]")
        console.print("[yellow]Make sure OPENAI_API_KEY is set[/yellow]")
        return stats

    # Initialize Qdrant store
    console.print("\n[bold blue]Initializing Qdrant store...[/]")
    mode = "in-memory" if in_memory else "localhost:6333"
    console.print(f"[dim]Mode: {mode}[/dim]")

    try:
        store = QdrantStore(in_memory=in_memory)
        store.create_collection(recreate=recreate)
        console.print("[green]Collection ready[/green]")
    except Exception as e:
        console.print(f"[red]Failed to initialize Qdrant: {e}[/red]")
        if not in_memory:
            console.print("[yellow]Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant[/yellow]")
        return stats

    # Generate embeddings
    console.print("\n[bold blue]Generating embeddings...[/]")
    texts = [chunk.get("text", "") for chunk in chunks]

    all_embeddings = []
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding chunks", total=len(texts))

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = embedder.embed_texts(batch)
            all_embeddings.extend(result.embeddings)
            stats["total_tokens_embedded"] += result.total_tokens
            progress.update(task, advance=len(batch))

    stats["embedding_time_s"] = round(time.time() - start_time, 2)
    console.print(f"[green]Generated {len(all_embeddings)} embeddings in {stats['embedding_time_s']}s[/green]")
    console.print(f"[dim]Total tokens: {stats['total_tokens_embedded']:,}[/dim]")

    # Upsert to Qdrant
    console.print("\n[bold blue]Upserting to Qdrant...[/]")
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Upserting chunks", total=len(chunks))

        # Process in batches with proper start_id to avoid ID collisions
        upsert_batch_size = 100
        for i in range(0, len(chunks), upsert_batch_size):
            batch_chunks = chunks[i:i + upsert_batch_size]
            batch_embeddings = all_embeddings[i:i + upsert_batch_size]

            store.upsert_chunks(
                batch_chunks,
                batch_embeddings,
                batch_size=upsert_batch_size,
                start_id=i,  # Pass start_id to avoid ID collisions
            )
            stats["chunks_ingested"] += len(batch_chunks)
            progress.update(task, advance=len(batch_chunks))

    stats["ingestion_time_s"] = round(time.time() - start_time, 2)
    console.print(f"[green]Upserted {stats['chunks_ingested']} chunks in {stats['ingestion_time_s']}s[/green]")

    # Print collection info
    console.print("\n[bold blue]Collection Info:[/]")
    try:
        info = store.get_info()
        table = Table(show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        for key, value in info.items():
            table.add_row(key, str(value))
        console.print(table)
    except Exception as e:
        console.print(f"[yellow]Could not get collection info: {e}[/yellow]")

    return stats


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest NYC Tax Law chunks into Qdrant")
    parser.add_argument(
        "--chunks",
        type=str,
        default="data/processed/chunks/chunks.json",
        help="Path to chunks.json file",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use in-memory Qdrant (for testing)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate collection",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for embeddings",
    )

    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        console.print(f"[red]Chunks file not found: {chunks_path}[/red]")
        console.print("[yellow]Run chunking first: python scripts/run_chunking.py[/yellow]")
        sys.exit(1)

    console.print("\n[bold]NYC Tax Law RAG - Ingestion Pipeline[/bold]")
    console.print("=" * 50)

    stats = run_ingestion(
        chunks_path=chunks_path,
        in_memory=args.in_memory,
        recreate=args.recreate,
        batch_size=args.batch_size,
    )

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    summary_table.add_row("Chunks loaded", str(stats["chunks_loaded"]))
    summary_table.add_row("Chunks ingested", str(stats["chunks_ingested"]))
    summary_table.add_row("Tokens embedded", f"{stats['total_tokens_embedded']:,}")
    summary_table.add_row("Embedding time", f"{stats['embedding_time_s']}s")
    summary_table.add_row("Ingestion time", f"{stats['ingestion_time_s']}s")
    console.print(summary_table)


if __name__ == "__main__":
    main()
