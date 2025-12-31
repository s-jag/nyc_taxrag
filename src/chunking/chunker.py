"""
Document chunking module for NYC Tax Law.

This module provides the main Chunker interface for splitting NYC tax law documents
into chunks suitable for RAG retrieval.

Strategies available:
- "zchunk": Uses Claude API to detect semantic boundaries (recommended)
- "fallback": Uses regex patterns, works offline

Usage:
    chunker = Chunker(strategy="zchunk")
    chunks = chunker.process_file("data/raw/document.html")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from .strategies import (
    Chunk,
    ChunkMetadata,
    ChunkingStrategy,
    ZChunkStrategy,
    FallbackChunkingStrategy,
)
from .html_parser import NYCTaxLawHTMLParser, ParsedDocument


# Re-export for backwards compatibility
__all__ = [
    "Chunk",
    "ChunkMetadata",
    "ChunkingStrategy",
    "Chunker",
]


StrategyType = Literal["zchunk", "fallback"]


class Chunker:
    """
    Main chunker interface for NYC Tax Law documents.

    Supports multiple chunking strategies:
    - "zchunk": Claude API-based semantic chunking (recommended)
    - "fallback": Regex-based chunking for offline use

    Usage:
        # With Claude API (recommended)
        chunker = Chunker(strategy="zchunk", api_key="...")
        chunks = chunker.process_file("data/raw/document.html")

        # Offline fallback
        chunker = Chunker(strategy="fallback")
        chunks = chunker.process_file("data/raw/document.txt")
    """

    def __init__(
        self,
        strategy: StrategyType = "fallback",
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        min_chunk_size: int = 200,
        max_chunk_size: int = 3000,
    ):
        """
        Initialize the chunker.

        Args:
            strategy: Chunking strategy to use:
                - "zchunk": Claude API-based (requires api_key or ANTHROPIC_API_KEY env var)
                - "fallback": Regex-based, works offline
            api_key: Anthropic API key (for zchunk strategy).
            model: Claude model for zchunk strategy.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks in characters.
            min_chunk_size: Minimum chunk size (merge smaller chunks).
            max_chunk_size: Maximum chunk size (split larger chunks).
        """
        self.strategy_name = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize HTML parser
        self.html_parser = NYCTaxLawHTMLParser()

        # Initialize strategy
        if strategy == "zchunk":
            self._strategy: ChunkingStrategy = ZChunkStrategy(
                api_key=api_key,
                model=model,
                batch_size=chunk_size * 5,  # Larger batches for API efficiency
                overlap=chunk_overlap,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
            )
        elif strategy == "fallback":
            self._strategy = FallbackChunkingStrategy(
                target_chunk_size=chunk_size,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                overlap=chunk_overlap,
            )
        else:
            raise ValueError(
                f"Unknown chunking strategy: {strategy}. "
                f"Available: ['zchunk', 'fallback']."
            )

    def chunk_text(
        self,
        text: str,
        source_file: str | None = None,
        base_metadata: ChunkMetadata | None = None,
    ) -> list[Chunk]:
        """
        Chunk a text string.

        Args:
            text: Text to chunk.
            source_file: Optional source file name for metadata.
            base_metadata: Optional base metadata to inherit.

        Returns:
            List of Chunk objects.
        """
        return self._strategy.chunk(text, source_file, base_metadata)

    def chunk_html(
        self,
        html_content: str,
        source_file: str | None = None,
    ) -> list[Chunk]:
        """
        Chunk HTML content.

        Parses the HTML structure first, then chunks the extracted text.

        Args:
            html_content: Raw HTML string.
            source_file: Optional source file name.

        Returns:
            List of Chunk objects.
        """
        # Parse HTML to extract text and structure
        doc = self.html_parser.parse_html(html_content, source_file)

        # Chunk the raw text
        base_metadata = ChunkMetadata(title=doc.title, source_file=source_file)
        return self.chunk_text(doc.raw_text, source_file, base_metadata)

    def process_file(self, file_path: str | Path) -> list[Chunk]:
        """
        Process a file and return chunks.

        Automatically detects HTML vs TXT and processes accordingly.

        Args:
            file_path: Path to the file to process.

        Returns:
            List of Chunk objects.
        """
        path = Path(file_path)

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Detect file type and process
        if path.suffix.lower() in ('.html', '.htm'):
            return self.chunk_html(content, source_file=path.name)
        else:
            return self.chunk_text(content, source_file=path.name)

    def process_html_file(self, file_path: str | Path) -> tuple[ParsedDocument, list[Chunk]]:
        """
        Process an HTML file and return both parsed document and chunks.

        Args:
            file_path: Path to the HTML file.

        Returns:
            Tuple of (ParsedDocument, list of Chunks).
        """
        path = Path(file_path)
        doc = self.html_parser.parse_file(path)

        base_metadata = ChunkMetadata(title=doc.title, source_file=path.name)
        chunks = self.chunk_text(doc.raw_text, path.name, base_metadata)

        return doc, chunks

    def save_chunks(
        self,
        chunks: list[Chunk],
        output_dir: str | Path,
        save_individual: bool = False,
    ) -> Path:
        """
        Save chunks to JSON file(s).

        Args:
            chunks: List of chunks to save.
            output_dir: Directory to save chunks.
            save_individual: If True, also save each chunk as separate file.

        Returns:
            Path to the main chunks.json file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main chunks file
        output_file = output_path / "chunks.json"
        data = {
            "metadata": {
                "strategy": self.strategy_name,
                "chunk_count": len(chunks),
                "total_tokens": sum(c.token_count for c in chunks),
                "avg_chunk_tokens": sum(c.token_count for c in chunks) // len(chunks) if chunks else 0,
                "generated_at": datetime.now().isoformat(),
            },
            "chunks": [c.to_dict() for c in chunks],
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Optionally save individual chunks
        if save_individual:
            chunks_dir = output_path / "individual"
            chunks_dir.mkdir(exist_ok=True)

            for chunk in chunks:
                chunk_file = chunks_dir / f"{chunk.id}.json"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk.to_dict(), f, indent=2, ensure_ascii=False)

        # Save statistics
        self._save_statistics(chunks, output_path)

        return output_file

    def _save_statistics(self, chunks: list[Chunk], output_path: Path) -> None:
        """Save chunk statistics to a separate file."""
        if not chunks:
            return

        token_counts = [c.token_count for c in chunks]
        char_counts = [len(c.text) for c in chunks]

        stats = {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "total_characters": sum(char_counts),
            "tokens": {
                "min": min(token_counts),
                "max": max(token_counts),
                "avg": sum(token_counts) // len(chunks),
            },
            "characters": {
                "min": min(char_counts),
                "max": max(char_counts),
                "avg": sum(char_counts) // len(chunks),
            },
            "by_chapter": self._count_by_metadata(chunks, "chapter"),
            "by_section": len(set(c.metadata.section_number for c in chunks if c.metadata.section_number)),
            "strategy": self.strategy_name,
            "generated_at": datetime.now().isoformat(),
        }

        stats_file = output_path / "chunk_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

    def _count_by_metadata(self, chunks: list[Chunk], field: str) -> dict[str, int]:
        """Count chunks by a metadata field."""
        counts: dict[str, int] = {}
        for chunk in chunks:
            value = getattr(chunk.metadata, field, None)
            if value:
                key = str(value)
                counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def load_chunks(file_path: str | Path) -> list[Chunk]:
        """
        Load chunks from a JSON file.

        Args:
            file_path: Path to the chunks.json file.

        Returns:
            List of Chunk objects.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return [Chunk.from_dict(c) for c in data.get("chunks", [])]
