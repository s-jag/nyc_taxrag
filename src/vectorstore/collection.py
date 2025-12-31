"""
Qdrant collection configuration for NYC Tax Law RAG.

Defines the collection schema with hybrid search support
(dense + sparse vectors) and payload indexes for filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        SparseVectorParams,
        Modifier,
        PayloadSchemaType,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None  # type: ignore


# Collection configuration
COLLECTION_NAME = "nyc_tax_law"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_DIMENSIONS = 1536  # OpenAI text-embedding-3-small


@dataclass
class CollectionConfig:
    """Configuration for the NYC Tax Law collection."""

    name: str = COLLECTION_NAME
    dense_dimensions: int = DENSE_DIMENSIONS
    dense_distance: str = "Cosine"
    sparse_modifier: str = "IDF"

    # Payload indexes for fast filtering
    indexed_fields: list[tuple[str, str]] = field(default_factory=lambda: [
        ("section_number", "keyword"),
        ("chapter_number", "integer"),
        ("chunk_type", "keyword"),
    ])


def create_collection(
    client: Any,  # QdrantClient
    config: CollectionConfig | None = None,
    recreate: bool = False,
) -> None:
    """
    Create the NYC Tax Law collection with hybrid search support.

    Args:
        client: Qdrant client instance.
        config: Collection configuration (uses defaults if None).
        recreate: If True, delete existing collection first.
    """
    if not QDRANT_AVAILABLE:
        raise ImportError(
            "qdrant-client is required. Install with: pip install qdrant-client"
        )

    config = config or CollectionConfig()

    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == config.name for c in collections)

    if exists:
        if recreate:
            client.delete_collection(config.name)
        else:
            return  # Collection already exists

    # Create collection with dense and sparse vectors
    client.create_collection(
        collection_name=config.name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=config.dense_dimensions,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                modifier=Modifier.IDF,
            ),
        },
    )

    # Create payload indexes for fast filtering
    for field_name, field_type in config.indexed_fields:
        schema = {
            "keyword": PayloadSchemaType.KEYWORD,
            "integer": PayloadSchemaType.INTEGER,
            "float": PayloadSchemaType.FLOAT,
            "text": PayloadSchemaType.TEXT,
        }.get(field_type, PayloadSchemaType.KEYWORD)

        client.create_payload_index(
            collection_name=config.name,
            field_name=field_name,
            field_schema=schema,
        )


def get_collection_info(client: Any, name: str = COLLECTION_NAME) -> dict[str, Any]:
    """
    Get information about the collection.

    Args:
        client: Qdrant client instance.
        name: Collection name.

    Returns:
        Dictionary with collection info.
    """
    if not QDRANT_AVAILABLE:
        raise ImportError(
            "qdrant-client is required. Install with: pip install qdrant-client"
        )

    info = client.get_collection(name)

    # Handle different qdrant-client versions
    result = {
        "name": name,
        "points_count": info.points_count,
        "status": info.status.name if hasattr(info.status, 'name') else str(info.status),
    }

    # These attributes may not exist in newer versions
    if hasattr(info, 'vectors_count'):
        result["vectors_count"] = info.vectors_count
    if hasattr(info, 'indexed_vectors_count'):
        result["indexed_vectors_count"] = info.indexed_vectors_count

    return result


# Payload schema documentation
PAYLOAD_SCHEMA = """
Payload Schema for NYC Tax Law chunks:

{
    "chunk_id": str,           # "chunk_00001"
    "text": str,               # Full chunk text
    "section": str | None,     # "§ 11-201"
    "section_number": str,     # "11-201"
    "section_title": str,      # Section title
    "chapter": str | None,     # "Chapter 2: ..."
    "chapter_number": int,     # 2
    "subchapter": str | None,  # "Subchapter 1: ..."
    "chunk_type": str,         # "big" or "small"
    "token_count": int,        # Token count
    "source_file": str,        # Source filename
    "citations": list[str],    # Cross-references found ["11-201", "11-202"]
}
"""
