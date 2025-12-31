"""
Base embedding interface for NYC Tax Law RAG.

Provides abstract base class for embedding providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""

    embeddings: list[list[float]]
    model: str
    total_tokens: int


class Embedder(ABC):
    """Abstract base class for text embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        """
        Embed multiple texts.

        Args:
            texts: Sequence of texts to embed.

        Returns:
            EmbeddingResult with embeddings and metadata.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass
