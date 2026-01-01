"""
Fallback module for low-confidence RAG results.

This module provides a fallback system that activates when RAG retrieval
returns low-confidence results. It uses a comprehensive legal prompt with
the full NYC Tax Law reference material and sends to OpenAI's reasoning model.
"""

from .config import FallbackConfig
from .handler import FallbackHandler
from .prompt import FallbackPromptBuilder
from .types import FallbackResponse

__all__ = [
    "FallbackConfig",
    "FallbackHandler",
    "FallbackPromptBuilder",
    "FallbackResponse",
]
