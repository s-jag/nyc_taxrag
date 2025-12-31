"""Embedding module for NYC Tax Law RAG."""

from .embedder import Embedder, EmbeddingResult
from .providers.openai_embedder import OpenAIEmbedder

__all__ = [
    "Embedder",
    "EmbeddingResult",
    "OpenAIEmbedder",
]
