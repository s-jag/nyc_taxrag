"""
Qdrant vector store for NYC Tax Law RAG.

Provides the main interface for storing and retrieving chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        SparseVector,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None  # type: ignore

try:
    from fastembed import SparseTextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    SparseTextEmbedding = None  # type: ignore

from .collection import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    CollectionConfig,
    create_collection,
    get_collection_info,
)


# Citation pattern for cross-references
CITATION_PATTERN = re.compile(r"§\s*(11-[\d.]+)")


@dataclass
class SearchResult:
    """Result from a vector search."""

    chunk_id: str
    text: str
    score: float
    section: str | None
    section_number: str | None
    chapter: str | None
    chapter_number: int | None
    chunk_type: str
    token_count: int
    citations: list[str]

    @classmethod
    def from_scored_point(cls, point: Any) -> "SearchResult":
        """Create from Qdrant ScoredPoint."""
        payload = point.payload or {}
        return cls(
            chunk_id=payload.get("chunk_id", ""),
            text=payload.get("text", ""),
            score=point.score,
            section=payload.get("section"),
            section_number=payload.get("section_number"),
            chapter=payload.get("chapter"),
            chapter_number=payload.get("chapter_number"),
            chunk_type=payload.get("chunk_type", "unknown"),
            token_count=payload.get("token_count", 0),
            citations=payload.get("citations", []),
        )


def extract_citations(text: str) -> list[str]:
    """
    Extract section citations from text.

    Args:
        text: Text to extract citations from.

    Returns:
        List of section numbers (e.g., ["11-201", "11-202"]).
    """
    return CITATION_PATTERN.findall(text)


class QdrantStore:
    """
    Qdrant vector store for NYC Tax Law chunks.

    Supports:
    - Dense vector search (OpenAI embeddings)
    - Sparse vector search (BM25)
    - Hybrid search with RRF fusion
    - Payload filtering
    """

    def __init__(
        self,
        url: str | None = None,
        port: int = 6333,
        api_key: str | None = None,
        collection_name: str = COLLECTION_NAME,
        in_memory: bool = False,
    ):
        """
        Initialize the Qdrant store.

        Args:
            url: Qdrant server URL. If None, checks QDRANT_URL env var, then localhost.
            port: Qdrant server port (ignored for cloud URLs).
            api_key: API key for Qdrant Cloud. If None, checks QDRANT_API_KEY env var.
            collection_name: Name of the collection.
            in_memory: If True, use in-memory storage (for testing).
        """
        import os

        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )

        self.collection_name = collection_name

        # Check environment variables
        url = url or os.getenv("QDRANT_URL")
        api_key = api_key or os.getenv("QDRANT_API_KEY")

        if in_memory:
            self.client = QdrantClient(":memory:")
        elif url and api_key:
            # Cloud connection
            self.client = QdrantClient(url=url, api_key=api_key)
        elif url:
            # URL without API key (local with custom URL)
            self.client = QdrantClient(url=url)
        else:
            # Default: localhost
            self.client = QdrantClient(host="localhost", port=port)

        # Initialize sparse embedding model
        self._sparse_model = None

    @property
    def sparse_model(self) -> Any:
        """Lazy-load the sparse embedding model."""
        if self._sparse_model is None:
            if not FASTEMBED_AVAILABLE:
                raise ImportError(
                    "fastembed is required for sparse embeddings. "
                    "Install with: pip install fastembed"
                )
            self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        return self._sparse_model

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the collection.

        Args:
            recreate: If True, delete and recreate if exists.
        """
        create_collection(self.client, recreate=recreate)

    def get_info(self) -> dict[str, Any]:
        """Get collection information."""
        return get_collection_info(self.client, self.collection_name)

    def _create_sparse_vector(self, text: str) -> SparseVector:
        """
        Create a sparse vector from text using BM25.

        Args:
            text: Text to vectorize.

        Returns:
            SparseVector for Qdrant.
        """
        embeddings = list(self.sparse_model.embed([text]))
        if embeddings:
            emb = embeddings[0]
            return SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist(),
            )
        return SparseVector(indices=[], values=[])

    def _create_sparse_vectors_batch(self, texts: list[str]) -> list[SparseVector]:
        """
        Create sparse vectors for multiple texts.

        Args:
            texts: Texts to vectorize.

        Returns:
            List of SparseVectors.
        """
        embeddings = list(self.sparse_model.embed(texts))
        return [
            SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist(),
            )
            for emb in embeddings
        ]

    def upsert_chunks(
        self,
        chunks: Sequence[dict[str, Any]],
        dense_embeddings: list[list[float]],
        batch_size: int = 100,
        start_id: int = 0,
    ) -> int:
        """
        Upsert chunks with their embeddings.

        Args:
            chunks: Chunk dictionaries with id, text, metadata.
            dense_embeddings: Dense embeddings for each chunk.
            batch_size: Batch size for upsert operations.
            start_id: Starting ID for points (to avoid collisions across calls).

        Returns:
            Number of points upserted.
        """
        if len(chunks) != len(dense_embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(dense_embeddings)} embeddings"
            )

        total_upserted = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = list(chunks[i:i + batch_size])
            batch_embeddings = dense_embeddings[i:i + batch_size]

            # Extract texts for sparse embedding
            texts = [c.get("text", "") for c in batch_chunks]
            sparse_vectors = self._create_sparse_vectors_batch(texts)

            # Create points
            points = []
            for j, (chunk, dense_emb, sparse_emb) in enumerate(
                zip(batch_chunks, batch_embeddings, sparse_vectors)
            ):
                # Calculate global point ID
                point_id = start_id + i + j

                # Extract citations from text
                citations = extract_citations(chunk.get("text", ""))

                # Build payload
                metadata = chunk.get("metadata", {})
                payload = {
                    "chunk_id": chunk.get("id", f"chunk_{point_id:05d}"),
                    "text": chunk.get("text", ""),
                    "section": metadata.get("section"),
                    "section_number": metadata.get("section_number"),
                    "section_title": metadata.get("section_title"),
                    "chapter": metadata.get("chapter"),
                    "chapter_number": metadata.get("chapter_number"),
                    "subchapter": metadata.get("subchapter"),
                    "chunk_type": metadata.get("chunk_type", "unknown"),
                    "token_count": chunk.get("token_count", 0),
                    "source_file": metadata.get("source_file"),
                    "citations": citations,
                    # New metadata fields for filtering
                    "jurisdiction": metadata.get("jurisdiction", "NYC"),
                    "doc_type": metadata.get("doc_type", "statute"),
                    "doc_version": metadata.get("doc_version"),
                    "effective_date": metadata.get("effective_date"),
                }

                point = PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: dense_emb,
                        SPARSE_VECTOR_NAME: sparse_emb,
                    },
                    payload=payload,
                )
                points.append(point)

            # Upsert batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            total_upserted += len(points)

        return total_upserted

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        section_filter: str | None = None,
        chapter_filter: int | None = None,
    ) -> list[SearchResult]:
        """
        Search using dense vectors only.

        Args:
            query_vector: Query embedding.
            limit: Maximum results to return.
            section_filter: Filter by section number.
            chapter_filter: Filter by chapter number.

        Returns:
            List of SearchResults.
        """
        # Build filter
        filter_conditions = []
        if section_filter:
            filter_conditions.append(
                FieldCondition(
                    key="section_number",
                    match=MatchValue(value=section_filter),
                )
            )
        if chapter_filter is not None:
            filter_conditions.append(
                FieldCondition(
                    key="chapter_number",
                    match=MatchValue(value=chapter_filter),
                )
            )

        search_filter = Filter(must=filter_conditions) if filter_conditions else None

        # Use query_points with named vector (new API)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=limit,
            query_filter=search_filter,
            with_payload=True,
        )

        return [SearchResult.from_scored_point(r) for r in results.points]

    def search_by_sections(
        self,
        section_numbers: list[str],
        limit_per_section: int = 1,
    ) -> list[SearchResult]:
        """
        Fetch chunks by section numbers (for cross-reference expansion).

        Args:
            section_numbers: List of section numbers to fetch.
            limit_per_section: Max chunks per section.

        Returns:
            List of SearchResults.
        """
        results = []

        for section in section_numbers:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="section_number",
                            match=MatchValue(value=section),
                        )
                    ]
                ),
                limit=limit_per_section,
                with_payload=True,
            )

            for point in scroll_result[0]:
                # Create a pseudo SearchResult with score 0
                payload = point.payload or {}
                result = SearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    text=payload.get("text", ""),
                    score=0.0,  # No score for scroll results
                    section=payload.get("section"),
                    section_number=payload.get("section_number"),
                    chapter=payload.get("chapter"),
                    chapter_number=payload.get("chapter_number"),
                    chunk_type=payload.get("chunk_type", "unknown"),
                    token_count=payload.get("token_count", 0),
                    citations=payload.get("citations", []),
                )
                results.append(result)

        return results

    def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)
