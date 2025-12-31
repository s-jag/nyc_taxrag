"""
Hybrid search for NYC Tax Law RAG.

Combines dense (semantic) and sparse (BM25) search with
Reciprocal Rank Fusion (RRF) for optimal retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Filter,
        FieldCondition,
        MatchValue,
        Prefetch,
        Query,
        FusionQuery,
        Fusion,
        SparseVector,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from .collection import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from .qdrant_store import SearchResult


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""

    dense_weight: float = 0.7  # Weight for dense (semantic) results
    sparse_weight: float = 0.3  # Weight for sparse (BM25) results
    prefetch_limit: int = 20  # Number of candidates to prefetch from each index
    rrf_k: int = 60  # RRF constant (default from Qdrant)


class HybridSearcher:
    """
    Hybrid search combining dense and sparse vectors.

    Uses Qdrant's native prefetch + RRF fusion for efficient
    two-stage retrieval.
    """

    def __init__(
        self,
        client: Any,  # QdrantClient
        collection_name: str = COLLECTION_NAME,
        config: HybridSearchConfig | None = None,
    ):
        """
        Initialize the hybrid searcher.

        Args:
            client: Qdrant client instance.
            collection_name: Name of the collection.
            config: Search configuration.
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )

        self.client = client
        self.collection_name = collection_name
        self.config = config or HybridSearchConfig()

    def search(
        self,
        query_dense: list[float],
        query_sparse: SparseVector,
        limit: int = 10,
        section_filter: str | None = None,
        chapter_filter: int | None = None,
    ) -> list[SearchResult]:
        """
        Perform hybrid search with RRF fusion.

        Args:
            query_dense: Dense embedding from OpenAI.
            query_sparse: Sparse vector from BM25.
            limit: Maximum results to return.
            section_filter: Filter by section number.
            chapter_filter: Filter by chapter number.

        Returns:
            List of SearchResults, ranked by RRF fusion.
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

        # Build prefetch queries for both vector types
        prefetch = [
            Prefetch(
                query=query_dense,
                using=DENSE_VECTOR_NAME,
                limit=self.config.prefetch_limit,
                filter=search_filter,
            ),
            Prefetch(
                query=query_sparse,
                using=SPARSE_VECTOR_NAME,
                limit=self.config.prefetch_limit,
                filter=search_filter,
            ),
        ]

        # Execute hybrid search with RRF fusion
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        return [SearchResult.from_scored_point(r) for r in results.points]

    def search_dense_only(
        self,
        query_dense: list[float],
        limit: int = 10,
        section_filter: str | None = None,
        chapter_filter: int | None = None,
    ) -> list[SearchResult]:
        """
        Search using only dense vectors (semantic search).

        Args:
            query_dense: Dense embedding from OpenAI.
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
            query=query_dense,
            using=DENSE_VECTOR_NAME,
            limit=limit,
            query_filter=search_filter,
            with_payload=True,
        )

        return [SearchResult.from_scored_point(r) for r in results.points]

    def search_sparse_only(
        self,
        query_sparse: SparseVector,
        limit: int = 10,
        section_filter: str | None = None,
        chapter_filter: int | None = None,
    ) -> list[SearchResult]:
        """
        Search using only sparse vectors (BM25 keyword search).

        Args:
            query_sparse: Sparse vector from BM25.
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
            query=query_sparse,
            using=SPARSE_VECTOR_NAME,
            limit=limit,
            query_filter=search_filter,
            with_payload=True,
        )

        return [SearchResult.from_scored_point(r) for r in results.points]
