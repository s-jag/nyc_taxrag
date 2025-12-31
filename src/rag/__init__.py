"""
RAG pipeline module for NYC Tax Law.
"""

from .config import RAGConfig
from .context import ContextAssembler, AssembledContext
from .pipeline import RAGPipeline, create_pipeline, SYSTEM_PROMPT
from .types import RAGResponse, RetrievalResult

__all__ = [
    "RAGConfig",
    "RAGPipeline",
    "RAGResponse",
    "RetrievalResult",
    "ContextAssembler",
    "AssembledContext",
    "create_pipeline",
    "SYSTEM_PROMPT",
]
