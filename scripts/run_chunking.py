#!/usr/bin/env python3
"""
Run the chunking pipeline on NYC Tax Law documents.

This script processes the NYC tax law HTML/TXT files and generates chunks
suitable for RAG retrieval.

Usage:
    # Using fallback strategy (offline, no API key needed)
    python scripts/run_chunking.py

    # Using zchunk strategy (requires ANTHROPIC_API_KEY)
    python scripts/run_chunking.py --strategy zchunk

    # Process specific file
    python scripts/run_chunking.py --input data/raw/document.html
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.chunking import Chunker


def main():
    parser = argparse.ArgumentParser(
        description="Chunk NYC Tax Law documents for RAG"
    )
    parser.add_argument(
        "--strategy",
        choices=["zchunk", "fallback"],
        default="fallback",
        help="Chunking strategy to use (default: fallback)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file path (default: auto-detect in data/raw/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/chunks",
        help="Output directory for chunks (default: data/processed/chunks)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1500,
        help="Target chunk size in characters (default: 1500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)",
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save each chunk as a separate file",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (for zchunk strategy)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and show stats without saving",
    )

    args = parser.parse_args()

    # Find input file if not specified
    input_file = args.input
    if input_file is None:
        raw_dir = project_root / "data" / "raw"
        # Prefer HTML over TXT
        html_file = raw_dir / "New York City Finance Chapter 1.html"
        txt_file = raw_dir / "New York City Finance Chapter 1.txt"

        if html_file.exists():
            input_file = str(html_file)
        elif txt_file.exists():
            input_file = str(txt_file)
        else:
            print(f"Error: No input files found in {raw_dir}")
            print("Expected: 'New York City Finance Chapter 1.html' or '.txt'")
            sys.exit(1)

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    output_dir = project_root / args.output

    print("=" * 60)
    print("NYC Tax Law Chunking Pipeline")
    print("=" * 60)
    print(f"Input:     {input_path.name}")
    print(f"Strategy:  {args.strategy}")
    print(f"Output:    {output_dir}")
    print(f"Chunk size: {args.chunk_size} chars")
    print(f"Overlap:   {args.overlap} chars")
    print("=" * 60)

    # Initialize chunker
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    if args.strategy == "zchunk" and not api_key:
        print("Warning: zchunk strategy requires ANTHROPIC_API_KEY")
        print("Set via --api-key or ANTHROPIC_API_KEY environment variable")
        print("Falling back to 'fallback' strategy...")
        args.strategy = "fallback"

    print(f"\nInitializing {args.strategy} chunker...")
    chunker = Chunker(
        strategy=args.strategy,
        api_key=api_key,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )

    # Process file
    print(f"Processing {input_path.name}...")
    print("This may take a while for large documents...")
    print()

    try:
        chunks = chunker.process_file(input_path)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

    # Print statistics
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        token_counts = [c.token_count for c in chunks]
        char_counts = [len(c.text) for c in chunks]

        print(f"Total tokens: {sum(token_counts):,}")
        print(f"Avg tokens/chunk: {sum(token_counts) // len(chunks)}")
        print(f"Min/Max tokens: {min(token_counts)} / {max(token_counts)}")
        print()
        print(f"Total characters: {sum(char_counts):,}")
        print(f"Avg chars/chunk: {sum(char_counts) // len(chunks)}")
        print()

        # Count by metadata
        sections = set(c.metadata.section_number for c in chunks if c.metadata.section_number)
        chapters = set(c.metadata.chapter for c in chunks if c.metadata.chapter)
        print(f"Unique sections: {len(sections)}")
        print(f"Unique chapters: {len(chapters)}")

    # Sample chunks
    if chunks:
        print()
        print("-" * 60)
        print("SAMPLE CHUNKS")
        print("-" * 60)

        for i, chunk in enumerate(chunks[:3]):
            print(f"\n[Chunk {chunk.id}]")
            print(f"  Tokens: {chunk.token_count}")
            if chunk.metadata.section:
                print(f"  Section: {chunk.metadata.section}")
            if chunk.metadata.chapter:
                print(f"  Chapter: {chunk.metadata.chapter}")
            print(f"  Text: {chunk.text[:200]}...")

    # Save if not dry run
    if not args.dry_run:
        print()
        print("-" * 60)
        print("SAVING CHUNKS")
        print("-" * 60)

        output_file = chunker.save_chunks(
            chunks,
            output_dir,
            save_individual=args.save_individual,
        )
        print(f"Saved to: {output_file}")
        print(f"Stats saved to: {output_dir / 'chunk_stats.json'}")
        if args.save_individual:
            print(f"Individual chunks: {output_dir / 'individual/'}")
    else:
        print()
        print("(Dry run - chunks not saved)")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
