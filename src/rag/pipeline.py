"""
RAG Pipeline for NYC Tax Law.

Orchestrates retrieval, context assembly, and generation.
"""

from __future__ import annotations

import time
import logging
from typing import TYPE_CHECKING

from .config import RAGConfig
from .context import ContextAssembler
from .types import RAGResponse, RetrievalResult

if TYPE_CHECKING:
    from rich.console import Console
    from src.embedding import OpenAIEmbedder
    from src.llm.client import LLMClient
    from src.vectorstore import QdrantStore, SearchResult
    from src.fallback import FallbackHandler, FallbackResponse


logger = logging.getLogger(__name__)


# System prompt for NYC Tax Law RAG
SYSTEM_PROMPT = """You are an expert assistant for NYC Tax Law (Title 11: Taxation and Finance).

Your role is to answer questions accurately based on the provided legal context. Follow these guidelines:

1. ACCURACY: Only answer based on the provided context. If the context doesn't contain enough information, say so clearly.

2. CITATIONS: Always cite the specific section numbers (e.g., § 11-201) when referencing the law.

3. STRUCTURE: For complex answers, organize your response with clear headings or bullet points.

4. LEGAL LANGUAGE: Maintain precision with legal terminology while explaining concepts clearly.

5. LIMITATIONS: If a question requires information not in the provided context, acknowledge this limitation.

6. NO SPECULATION: Do not make up or assume information not present in the context."""


class RAGPipeline:
    """
    Main RAG pipeline for NYC Tax Law.

    Orchestrates:
    - Query embedding
    - Dense or hybrid retrieval
    - Cross-reference expansion
    - Context assembly with deduplication
    - LLM generation

    Example:
        >>> from src.embedding import OpenAIEmbedder
        >>> from src.vectorstore import QdrantStore
        >>> from src.llm.providers import OpenAIClient
        >>> from src.rag import RAGPipeline, RAGConfig
        >>>
        >>> pipeline = RAGPipeline(
        ...     embedder=OpenAIEmbedder(),
        ...     store=QdrantStore(),
        ...     llm=OpenAIClient(),
        ... )
        >>> response = pipeline.query("What is the property tax rate?")
        >>> print(response.answer)
    """

    def __init__(
        self,
        embedder: "OpenAIEmbedder",
        store: "QdrantStore",
        llm: "LLMClient",
        config: RAGConfig | None = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            embedder: Embedder for query vectorization.
            store: Vector store for retrieval.
            llm: LLM client for generation.
            config: Pipeline configuration.
        """
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.config = config or RAGConfig()

        # Initialize components
        self.assembler = ContextAssembler(max_tokens=self.config.max_context_tokens)

        # Lazy-load hybrid searcher, expander, and fallback handler
        self._hybrid_searcher = None
        self._expander = None
        self._fallback_handler = None

    @property
    def fallback_handler(self) -> "FallbackHandler":
        """Lazy-load fallback handler."""
        if self._fallback_handler is None:
            from src.fallback import FallbackHandler, FallbackConfig

            fallback_config = FallbackConfig(
                confidence_threshold=self.config.fallback_threshold,
                fallback_model=self.config.fallback_model,
                prompt_path=self.config.fallback_prompt_path,
            )
            self._fallback_handler = FallbackHandler(config=fallback_config)
        return self._fallback_handler

    @property
    def hybrid_searcher(self):
        """Lazy-load hybrid searcher."""
        if self._hybrid_searcher is None:
            from src.vectorstore import HybridSearcher
            self._hybrid_searcher = HybridSearcher(self.store.client)
        return self._hybrid_searcher

    @property
    def expander(self):
        """Lazy-load cross-reference expander."""
        if self._expander is None:
            from src.vectorstore import CrossReferenceExpander
            self._expander = CrossReferenceExpander(self.store)
        return self._expander

    def query(
        self,
        question: str,
        console: "Console | None" = None,
    ) -> "RAGResponse | FallbackResponse":
        """
        Execute the full RAG pipeline.

        Args:
            question: User's question.
            console: Optional Rich console for verbose fallback output.

        Returns:
            RAGResponse with answer and sources, or FallbackResponse if fallback triggered.
        """
        logger.info(f"RAG query: {question[:100]}...")

        # 1. Retrieve relevant chunks
        retrieval_result = self.retrieve(question)

        # 2. Check if fallback should be triggered
        if self.config.enable_fallback:
            if self.fallback_handler.should_fallback(retrieval_result):
                logger.info(
                    f"Triggering fallback: top score below threshold "
                    f"({self.config.fallback_threshold})"
                )
                return self.fallback_handler.execute_fallback(
                    question=question,
                    retrieval_result=retrieval_result,
                    console=console,
                )

        # Normal RAG flow continues
        # 3. Assemble context
        context = self.assembler.assemble(
            primary_results=retrieval_result.primary_chunks,
            expanded_results=retrieval_result.expanded_chunks,
            deduplicate=self.config.deduplicate,
        )

        logger.info(
            f"Context assembled: {context.total_tokens} tokens, "
            f"{len(context.chunks_used)} chunks, "
            f"{context.chunks_truncated} truncated"
        )

        # 4. Generate response
        gen_start = time.perf_counter()

        prompt = self._build_prompt(question, context.context_text)
        llm_response = self.llm.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        generation_time_ms = (time.perf_counter() - gen_start) * 1000

        logger.info(
            f"Generated response: {llm_response.output_tokens} tokens in {generation_time_ms:.0f}ms"
        )

        return RAGResponse(
            answer=llm_response.content,
            sources=context.chunks_used,
            query=question,
            context_tokens=context.total_tokens,
            retrieval_time_ms=retrieval_result.retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            model=self.llm.model_name,
            retrieval_mode=self.config.retrieval_mode,
        )

    def retrieve(self, question: str) -> RetrievalResult:
        """
        Retrieve relevant chunks for a question.

        Args:
            question: User's question.

        Returns:
            RetrievalResult with chunks and metadata.
        """
        start_time = time.perf_counter()

        # Embed query
        query_dense = self.embedder.embed_query(question)

        # Retrieve based on mode
        if self.config.retrieval_mode == "hybrid":
            results = self._hybrid_retrieve(query_dense, question)
        else:
            results = self._dense_retrieve(query_dense)

        # Expand cross-references if enabled
        expanded_refs: list[str] = []
        if self.config.expand_cross_refs and results:
            original_sections = {r.section_number for r in results if r.section_number}
            expanded_results = self.expander.expand_results(results)

            # Track which sections were added via expansion
            for r in expanded_results:
                if r.section_number and r.section_number not in original_sections:
                    expanded_refs.append(r.section_number)

            results = expanded_results

        retrieval_time_ms = (time.perf_counter() - start_time) * 1000

        filters_applied = {}
        if self.config.section_filter:
            filters_applied["section"] = self.config.section_filter
        if self.config.chapter_filter:
            filters_applied["chapter"] = self.config.chapter_filter

        logger.info(
            f"Retrieved {len(results)} chunks ({self.config.retrieval_mode}) "
            f"in {retrieval_time_ms:.0f}ms, expanded: {len(expanded_refs)}"
        )

        return RetrievalResult(
            chunks=results,
            query=question,
            retrieval_mode=self.config.retrieval_mode,
            filters_applied=filters_applied,
            expanded_refs=expanded_refs,
            retrieval_time_ms=retrieval_time_ms,
        )

    def _dense_retrieve(self, query_vector: list[float]) -> list["SearchResult"]:
        """Perform dense-only retrieval."""
        return self.store.search(
            query_vector=query_vector,
            limit=self.config.top_k,
            section_filter=self.config.section_filter,
            chapter_filter=self.config.chapter_filter,
        )

    def _hybrid_retrieve(
        self,
        query_vector: list[float],
        query_text: str,
    ) -> list["SearchResult"]:
        """Perform hybrid retrieval with RRF fusion."""
        query_sparse = self.store._create_sparse_vector(query_text)
        return self.hybrid_searcher.search(
            query_dense=query_vector,
            query_sparse=query_sparse,
            limit=self.config.top_k,
            section_filter=self.config.section_filter,
            chapter_filter=self.config.chapter_filter,
        )

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build the prompt for the LLM.

        Args:
            question: User's question.
            context: Assembled context from retrieval.

        Returns:
            Formatted prompt string.
        """
        return f"""Based on the following NYC Tax Law excerpts, please answer the question.

## Context

{context}

## Question

{question}

## Answer

Please provide a clear, accurate answer based on the context above. Cite specific sections where applicable."""

    def retrieve_only(self, question: str) -> RetrievalResult:
        """
        Perform retrieval without generation.

        Useful for testing retrieval quality.

        Args:
            question: User's question.

        Returns:
            RetrievalResult with chunks.
        """
        return self.retrieve(question)


def create_pipeline(
    config: RAGConfig | None = None,
    in_memory: bool = False,
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline with default components.

    Args:
        config: Optional pipeline configuration.
        in_memory: Use in-memory Qdrant for testing.

    Returns:
        Configured RAGPipeline instance.
    """
    from src.embedding import OpenAIEmbedder
    from src.vectorstore import QdrantStore
    from src.llm.providers import OpenAIClient

    embedder = OpenAIEmbedder()
    store = QdrantStore(in_memory=in_memory)
    llm = OpenAIClient(model=config.model if config else "gpt-4o")

    return RAGPipeline(
        embedder=embedder,
        store=store,
        llm=llm,
        config=config,
    )
