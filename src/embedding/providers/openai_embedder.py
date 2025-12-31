"""
OpenAI embedding provider for NYC Tax Law RAG.

Uses text-embedding-3-small for efficient, high-quality embeddings.
"""

from __future__ import annotations

import os
import time
from typing import Sequence

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

from ..embedder import Embedder, EmbeddingResult


class OpenAIEmbedder(Embedder):
    """
    OpenAI embedding provider using text-embedding-3-small.

    Features:
    - 1536 dimensions
    - Batch processing with rate limiting
    - Automatic retry on failures
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSIONS = 1536
    MAX_BATCH_SIZE = 2048  # OpenAI limit
    DEFAULT_BATCH_SIZE = 100  # Practical batch size

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the OpenAI embedder.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model name (default: text-embedding-3-small).
            dimensions: Embedding dimensions (default: 1536).
            batch_size: Batch size for embedding requests.
            max_retries: Maximum retry attempts on failure.
            retry_delay: Delay between retries in seconds.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required for OpenAIEmbedder. "
                "Install with: pip install openai"
            )

        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        self.client = OpenAI(api_key=self._api_key)
        self._model = model
        self._dimensions = dimensions
        self.batch_size = min(batch_size, self.MAX_BATCH_SIZE)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        """
        Embed multiple texts with batching.

        Args:
            texts: Sequence of texts to embed.

        Returns:
            EmbeddingResult with all embeddings.
        """
        if not texts:
            return EmbeddingResult(embeddings=[], model=self._model, total_tokens=0)

        all_embeddings: list[list[float]] = []
        total_tokens = 0

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            result = self._embed_batch(batch)
            all_embeddings.extend(result.embeddings)
            total_tokens += result.total_tokens

        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self._model,
            total_tokens=total_tokens,
        )

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector.
        """
        result = self._embed_batch([query])
        return result.embeddings[0]

    def _embed_batch(self, texts: list[str]) -> EmbeddingResult:
        """
        Embed a single batch with retry logic.

        Args:
            texts: Batch of texts to embed.

        Returns:
            EmbeddingResult for the batch.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self._model,
                    input=texts,
                    dimensions=self._dimensions,
                )

                embeddings = [item.embedding for item in response.data]
                total_tokens = response.usage.total_tokens

                return EmbeddingResult(
                    embeddings=embeddings,
                    model=self._model,
                    total_tokens=total_tokens,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                continue

        raise RuntimeError(
            f"OpenAI embedding failed after {self.max_retries} attempts: {last_error}"
        )
