"""
Document chunking module for NYC Tax Law.

[BLACK BOX - Interface defined, implementation TBD]

This module provides the chunking interface for splitting NYC tax law documents
into chunks suitable for RAG retrieval. The actual chunking algorithm will be
implemented later based on requirements.

Design considerations for implementation:
- Respect section boundaries (§ 11-XXX)
- Preserve hierarchy metadata (Chapter → Subchapter → Section)
- Target chunk size: 500-1000 tokens with configurable overlap
- Handle cross-references and legal citations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class ChunkMetadata:
    """Metadata associated with a document chunk."""

    # Document hierarchy
    title: str = "Title 11: Taxation and Finance"
    chapter: str | None = None
    chapter_number: int | None = None
    subchapter: str | None = None
    subchapter_number: int | None = None
    section: str | None = None
    section_number: str | None = None  # e.g., "11-201"

    # Position information
    start_line: int | None = None
    end_line: int | None = None
    chunk_index: int = 0

    # Source tracking
    source_file: str | None = None

    def to_dict(self) -> dict:
        """Convert metadata to dictionary for storage."""
        return {
            "title": self.title,
            "chapter": self.chapter,
            "chapter_number": self.chapter_number,
            "subchapter": self.subchapter,
            "subchapter_number": self.subchapter_number,
            "section": self.section,
            "section_number": self.section_number,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_index": self.chunk_index,
            "source_file": self.source_file,
        }


@dataclass
class Chunk:
    """A chunk of document text with associated metadata."""

    id: str
    text: str
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    # Token count (computed lazily)
    _token_count: int | None = None

    @property
    def token_count(self) -> int:
        """Estimate token count for this chunk."""
        if self._token_count is None:
            # Simple heuristic: ~4 characters per token
            self._token_count = len(self.text) // 4
        return self._token_count

    def to_dict(self) -> dict:
        """Convert chunk to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
            "token_count": self.token_count,
        }


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, source_file: str | None = None) -> Iterator[Chunk]:
        """
        Split text into chunks.

        Args:
            text: The full document text to chunk.
            source_file: Optional source file name for metadata.

        Yields:
            Chunk objects with text and metadata.
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        pass


class PlaceholderChunker(ChunkingStrategy):
    """
    Placeholder chunking implementation.

    [BLACK BOX - This is a stub implementation]

    This simple implementation splits by a fixed number of characters.
    It will be replaced with a section-aware chunking algorithm.
    """

    def __init__(self, chunk_size: int = 3000, overlap: int = 400):
        """
        Initialize the placeholder chunker.

        Args:
            chunk_size: Target chunk size in characters.
            overlap: Overlap between chunks in characters.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def get_strategy_name(self) -> str:
        return "placeholder"

    def chunk(self, text: str, source_file: str | None = None) -> Iterator[Chunk]:
        """
        Simple character-based chunking (placeholder implementation).

        TODO: Replace with section-aware chunking that:
        - Respects § 11-XXX section boundaries
        - Extracts chapter/subchapter metadata
        - Handles subsection hierarchy
        - Preserves cross-references
        """
        chunk_index = 0
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at a paragraph boundary
            if end < len(text):
                # Look for paragraph break within last 20% of chunk
                search_start = start + int(self.chunk_size * 0.8)
                newline_pos = text.rfind('\n\n', search_start, end)
                if newline_pos > search_start:
                    end = newline_pos

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"chunk_{chunk_index:05d}"

                # Create basic metadata (to be enhanced in real implementation)
                metadata = ChunkMetadata(
                    chunk_index=chunk_index,
                    source_file=source_file,
                )

                yield Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata=metadata,
                )

                chunk_index += 1

            # Move to next chunk with overlap
            start = end - self.overlap
            if start <= (end - self.chunk_size):
                start = end  # Prevent infinite loop


class Chunker:
    """
    Main chunker interface for NYC Tax Law documents.

    [BLACK BOX - Strategy selection and configuration TBD]

    Usage:
        chunker = Chunker(strategy="section_aware")
        chunks = chunker.process_file("data/raw/document.txt")
    """

    def __init__(
        self,
        strategy: str = "placeholder",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        """
        Initialize the chunker.

        Args:
            strategy: Chunking strategy name. Options:
                - "placeholder": Simple character-based (default, for testing)
                - "section_aware": Respects section boundaries (TODO)
                - "hierarchical": Maintains document hierarchy (TODO)
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap between chunks in tokens.
        """
        self.strategy_name = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Convert token targets to approximate character counts
        char_size = chunk_size * 4  # ~4 chars per token
        char_overlap = chunk_overlap * 4

        # Initialize strategy
        # TODO: Add more strategies as they are implemented
        if strategy == "placeholder":
            self._strategy = PlaceholderChunker(
                chunk_size=char_size,
                overlap=char_overlap,
            )
        else:
            raise ValueError(
                f"Unknown chunking strategy: {strategy}. "
                f"Available: ['placeholder']. "
                f"More strategies coming soon."
            )

    def chunk_text(self, text: str, source_file: str | None = None) -> list[Chunk]:
        """
        Chunk a text string.

        Args:
            text: Text to chunk.
            source_file: Optional source file name for metadata.

        Returns:
            List of Chunk objects.
        """
        return list(self._strategy.chunk(text, source_file))

    def process_file(self, file_path: str | Path) -> list[Chunk]:
        """
        Process a file and return chunks.

        Args:
            file_path: Path to the file to process.

        Returns:
            List of Chunk objects.
        """
        path = Path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        return self.chunk_text(text, source_file=path.name)

    def save_chunks(self, chunks: list[Chunk], output_dir: str | Path) -> Path:
        """
        Save chunks to JSON file.

        Args:
            chunks: List of chunks to save.
            output_dir: Directory to save chunks.

        Returns:
            Path to the saved file.
        """
        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / "chunks.json"
        data = {
            "strategy": self.strategy_name,
            "chunk_count": len(chunks),
            "chunks": [c.to_dict() for c in chunks],
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        return output_file
