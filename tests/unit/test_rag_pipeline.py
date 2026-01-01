"""
Unit tests for the RAG pipeline.
"""

import pytest
from unittest.mock import Mock, MagicMock

from src.rag import RAGConfig, RAGPipeline, ContextAssembler, AssembledContext
from src.rag.types import RAGResponse, RetrievalResult
from src.vectorstore import SearchResult
from src.llm.client import LLMResponse


# Fixtures

@pytest.fixture
def mock_search_results():
    """Create mock search results for testing."""
    return [
        SearchResult(
            chunk_id="chunk_001",
            text="Property tax is levied under § 11-201.",
            score=0.95,
            section="§ 11-201",
            section_number="11-201",
            chapter="Real Property Tax",
            chapter_number=2,
            chunk_type="text",
            token_count=50,
            citations=["11-202"],
        ),
        SearchResult(
            chunk_id="chunk_002",
            text="Exemptions are provided in § 11-202.",
            score=0.85,
            section="§ 11-202",
            section_number="11-202",
            chapter="Real Property Tax",
            chapter_number=2,
            chunk_type="text",
            token_count=45,
            citations=[],
        ),
        SearchResult(
            chunk_id="chunk_003",
            text="Assessment procedures under § 11-203.",
            score=0.75,
            section="§ 11-203",
            section_number="11-203",
            chapter="Real Property Tax",
            chapter_number=2,
            chunk_type="text",
            token_count=40,
            citations=["11-201"],
        ),
    ]


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = Mock()
    embedder.embed_query.return_value = [0.1] * 1536
    return embedder


@pytest.fixture
def mock_store(mock_search_results):
    """Create a mock vector store."""
    store = Mock()
    store.search.return_value = mock_search_results
    store._create_sparse_vector.return_value = Mock()
    store.client = Mock()
    return store


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = Mock()
    llm.model_name = "gpt-4o"
    llm.generate.return_value = LLMResponse(
        content="Based on § 11-201, property tax is levied on real property.",
        model="gpt-4o",
        input_tokens=500,
        output_tokens=50,
        stop_reason="stop",
    )
    return llm


# Context Assembler Tests

class TestContextAssembler:
    """Tests for ContextAssembler."""

    def test_assemble_basic(self, mock_search_results):
        """Test basic context assembly."""
        assembler = ContextAssembler(max_tokens=8000)
        result = assembler.assemble(mock_search_results)

        assert isinstance(result, AssembledContext)
        assert len(result.chunks_used) == 3
        assert result.total_tokens == 50 + 45 + 40
        assert result.chunks_truncated == 0
        assert "[Source 1]" in result.context_text
        assert "§ 11-201" in result.context_text

    def test_deduplication(self, mock_search_results):
        """Test that duplicate chunks are removed."""
        # Add a duplicate with lower score
        duplicate = SearchResult(
            chunk_id="chunk_001",  # Same ID
            text="Property tax is levied under § 11-201.",
            score=0.5,  # Lower score
            section="§ 11-201",
            section_number="11-201",
            chapter="Real Property Tax",
            chapter_number=2,
            chunk_type="text",
            token_count=50,
            citations=[],
        )
        results_with_dup = mock_search_results + [duplicate]

        assembler = ContextAssembler(max_tokens=8000)
        result = assembler.assemble(results_with_dup, deduplicate=True)

        # Should only have 3 unique chunks
        assert len(result.chunks_used) == 3

        # The kept chunk should have the higher score
        kept_chunk = next(c for c in result.chunks_used if c.chunk_id == "chunk_001")
        assert kept_chunk.score == 0.95

    def test_token_budget(self, mock_search_results):
        """Test that context stays within token budget."""
        assembler = ContextAssembler(max_tokens=100)  # Small budget
        result = assembler.assemble(mock_search_results)

        assert result.total_tokens <= 100
        assert result.chunks_truncated > 0  # Some chunks didn't fit

    def test_deterministic_ordering(self, mock_search_results):
        """Test that ordering is deterministic."""
        assembler = ContextAssembler(max_tokens=8000)

        # Run assembly multiple times
        results = [assembler.assemble(mock_search_results) for _ in range(3)]

        # All should produce same order
        for r in results[1:]:
            assert [c.chunk_id for c in r.chunks_used] == [
                c.chunk_id for c in results[0].chunks_used
            ]

    def test_expanded_results_sorted_after_primary(self, mock_search_results):
        """Test that expanded results (score=0) come after primary results."""
        expanded = SearchResult(
            chunk_id="chunk_ref",
            text="Referenced section content.",
            score=0.0,  # Expanded result
            section="§ 11-204",
            section_number="11-204",
            chapter="Real Property Tax",
            chapter_number=2,
            chunk_type="text",
            token_count=30,
            citations=[],
        )

        assembler = ContextAssembler(max_tokens=8000)
        result = assembler.assemble(mock_search_results, expanded_results=[expanded])

        # Expanded result should be last
        assert result.chunks_used[-1].chunk_id == "chunk_ref"
        assert result.chunks_used[-1].score == 0.0


# RAG Config Tests

class TestRAGConfig:
    """Tests for RAGConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RAGConfig()

        assert config.retrieval_mode == "dense"
        assert config.top_k == 10
        assert config.max_context_tokens == 8000
        assert config.expand_cross_refs is True
        assert config.deduplicate is True
        assert config.temperature == 0.1

    def test_invalid_retrieval_mode(self):
        """Test that invalid retrieval mode raises error."""
        with pytest.raises(ValueError, match="Invalid retrieval_mode"):
            RAGConfig(retrieval_mode="invalid")

    def test_invalid_top_k(self):
        """Test that invalid top_k raises error."""
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            RAGConfig(top_k=0)

    def test_hybrid_mode(self):
        """Test hybrid mode configuration."""
        config = RAGConfig(retrieval_mode="hybrid")
        assert config.retrieval_mode == "hybrid"


# RAG Pipeline Tests

class TestRAGPipeline:
    """Tests for RAGPipeline."""

    def test_pipeline_initialization(self, mock_embedder, mock_store, mock_llm):
        """Test pipeline initialization."""
        config = RAGConfig()
        pipeline = RAGPipeline(
            embedder=mock_embedder,
            store=mock_store,
            llm=mock_llm,
            config=config,
        )

        assert pipeline.embedder is mock_embedder
        assert pipeline.store is mock_store
        assert pipeline.llm is mock_llm
        assert pipeline.config is config

    def test_dense_retrieval(self, mock_embedder, mock_store, mock_llm, mock_search_results):
        """Test dense retrieval mode."""
        config = RAGConfig(retrieval_mode="dense", expand_cross_refs=False)
        pipeline = RAGPipeline(
            embedder=mock_embedder,
            store=mock_store,
            llm=mock_llm,
            config=config,
        )

        result = pipeline.retrieve("What is property tax?")

        assert isinstance(result, RetrievalResult)
        assert result.retrieval_mode == "dense"
        assert len(result.chunks) == 3
        mock_embedder.embed_query.assert_called_once_with("What is property tax?")
        mock_store.search.assert_called_once()

    def test_full_query(self, mock_embedder, mock_store, mock_llm):
        """Test full RAG query."""
        config = RAGConfig(expand_cross_refs=False, enable_fallback=False)
        pipeline = RAGPipeline(
            embedder=mock_embedder,
            store=mock_store,
            llm=mock_llm,
            config=config,
        )

        response = pipeline.query("What is property tax?")

        assert isinstance(response, RAGResponse)
        assert "§ 11-201" in response.answer
        assert len(response.sources) > 0
        assert response.retrieval_mode == "dense"
        assert response.model == "gpt-4o"
        mock_llm.generate.assert_called_once()

    def test_retrieve_only(self, mock_embedder, mock_store, mock_llm):
        """Test retrieve-only mode (no generation)."""
        config = RAGConfig(expand_cross_refs=False)
        pipeline = RAGPipeline(
            embedder=mock_embedder,
            store=mock_store,
            llm=mock_llm,
            config=config,
        )

        result = pipeline.retrieve_only("What is property tax?")

        assert isinstance(result, RetrievalResult)
        # LLM should not be called
        mock_llm.generate.assert_not_called()


# RAG Response Tests

class TestRAGResponse:
    """Tests for RAGResponse."""

    def test_format_sources(self, mock_search_results):
        """Test source formatting."""
        response = RAGResponse(
            answer="Test answer",
            sources=mock_search_results,
            query="Test query",
            context_tokens=100,
            retrieval_time_ms=50.0,
            generation_time_ms=100.0,
            model="claude-sonnet-4-20250514",
            retrieval_mode="dense",
        )

        formatted = response.format_sources()
        assert "§ 11-201" in formatted
        assert "§ 11-202" in formatted
        assert "[1]" in formatted

    def test_total_time(self, mock_search_results):
        """Test total time calculation."""
        response = RAGResponse(
            answer="Test",
            sources=mock_search_results,
            query="Test",
            context_tokens=100,
            retrieval_time_ms=50.0,
            generation_time_ms=100.0,
            model="test",
            retrieval_mode="dense",
        )

        assert response.total_time_ms == 150.0

    def test_source_sections(self, mock_search_results):
        """Test unique source sections extraction."""
        response = RAGResponse(
            answer="Test",
            sources=mock_search_results,
            query="Test",
            context_tokens=100,
            retrieval_time_ms=0,
            generation_time_ms=0,
            model="test",
            retrieval_mode="dense",
        )

        sections = response.source_sections
        assert len(sections) == 3
        assert "11-201" in sections
        assert "11-202" in sections
        assert "11-203" in sections


# Retrieval Result Tests

class TestRetrievalResult:
    """Tests for RetrievalResult."""

    def test_primary_vs_expanded_chunks(self, mock_search_results):
        """Test separation of primary and expanded chunks."""
        # Add an expanded result
        expanded = SearchResult(
            chunk_id="expanded",
            text="Expanded content",
            score=0.0,  # Expansion indicator
            section="§ 11-999",
            section_number="11-999",
            chapter=None,
            chapter_number=None,
            chunk_type="text",
            token_count=10,
            citations=[],
        )
        all_results = mock_search_results + [expanded]

        result = RetrievalResult(
            chunks=all_results,
            query="test",
            retrieval_mode="dense",
        )

        assert len(result.primary_chunks) == 3
        assert len(result.expanded_chunks) == 1
        assert result.total_chunks == 4
