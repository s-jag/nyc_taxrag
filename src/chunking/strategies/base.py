"""
Abstract base class for chunking strategies.

All chunking strategies should inherit from ChunkingStrategy
and implement the chunk() method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    section_title: str | None = None

    # Chunk classification
    chunk_type: str = "small"  # "big" or "small"

    # Position information
    start_char: int | None = None
    end_char: int | None = None
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
            "section_title": self.section_title,
            "chunk_type": self.chunk_type,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_index": self.chunk_index,
            "source_file": self.source_file,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkMetadata":
        """Create ChunkMetadata from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Chunk:
    """A chunk of document text with associated metadata."""

    id: str
    text: str
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    # Token count (computed lazily)
    _token_count: int | None = field(default=None, repr=False)

    @property
    def token_count(self) -> int:
        """Estimate token count for this chunk."""
        if self._token_count is None:
            # Simple heuristic: ~4 characters per token
            self._token_count = len(self.text) // 4
        return self._token_count

    @token_count.setter
    def token_count(self, value: int) -> None:
        """Set the token count explicitly."""
        self._token_count = value

    def to_dict(self) -> dict:
        """Convert chunk to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Create Chunk from dictionary."""
        metadata = ChunkMetadata.from_dict(data.get("metadata", {}))
        chunk = cls(
            id=data["id"],
            text=data["text"],
            metadata=metadata,
        )
        if "token_count" in data:
            chunk._token_count = data["token_count"]
        return chunk


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        source_file: str | None = None,
        base_metadata: ChunkMetadata | None = None,
    ) -> list[Chunk]:
        """
        Split text into chunks.

        Args:
            text: The full document text to chunk.
            source_file: Optional source file name for metadata.
            base_metadata: Optional base metadata to inherit.

        Returns:
            List of Chunk objects with text and metadata.
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        pass

    def chunk_iterator(
        self,
        text: str,
        source_file: str | None = None,
        base_metadata: ChunkMetadata | None = None,
    ) -> Iterator[Chunk]:
        """
        Iterate over chunks (generator version of chunk()).

        Default implementation just iterates over chunk() result.
        Override for memory-efficient streaming.
        """
        yield from self.chunk(text, source_file, base_metadata)
