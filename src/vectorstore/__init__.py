"""Vector store module for NYC Tax Law RAG."""

from .collection import (
    COLLECTION_NAME,
    CollectionConfig,
    create_collection,
    get_collection_info,
)
from .qdrant_store import QdrantStore, SearchResult, extract_citations
from .hybrid_search import HybridSearcher, HybridSearchConfig
from .cross_reference import (
    CrossReferenceExpander,
    CitationExpansionConfig,
    extract_section_citations,
)

__all__ = [
    # Collection
    "COLLECTION_NAME",
    "CollectionConfig",
    "create_collection",
    "get_collection_info",
    # Store
    "QdrantStore",
    "SearchResult",
    "extract_citations",
    # Hybrid search
    "HybridSearcher",
    "HybridSearchConfig",
    # Cross-reference
    "CrossReferenceExpander",
    "CitationExpansionConfig",
    "extract_section_citations",
]
