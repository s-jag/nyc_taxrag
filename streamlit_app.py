"""
NYC TaxRAG - Streamlit QA Testing UI

Lightweight server-side UI for testing the RAG pipeline.
Deploy to Streamlit Cloud for free hosting with secrets management.
"""

import os
import streamlit as st

# Set page config first (must be first Streamlit command)
st.set_page_config(
    page_title="NYC TaxRAG QA",
    page_icon="",
    layout="wide",
)

# Load secrets into environment variables for the pipeline
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "QDRANT_URL" in st.secrets:
    os.environ["QDRANT_URL"] = st.secrets["QDRANT_URL"]
if "QDRANT_API_KEY" in st.secrets:
    os.environ["QDRANT_API_KEY"] = st.secrets["QDRANT_API_KEY"]

from src.rag import RAGPipeline, RAGConfig
from src.embedding import OpenAIEmbedder
from src.vectorstore import QdrantStore
from src.llm.providers import OpenAIClient
from src.fallback import FallbackResponse


@st.cache_resource
def get_pipeline(
    retrieval_mode: str,
    top_k: int,
    enable_fallback: bool,
    fallback_threshold: float,
) -> RAGPipeline:
    """Initialize and cache the RAG pipeline."""
    embedder = OpenAIEmbedder()
    store = QdrantStore()
    llm = OpenAIClient()

    config = RAGConfig(
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        expand_cross_refs=True,
        max_context_tokens=6000,
        enable_fallback=enable_fallback,
        fallback_threshold=fallback_threshold,
    )

    return RAGPipeline(
        embedder=embedder,
        store=store,
        llm=llm,
        config=config,
    )


def main():
    """Main Streamlit application."""
    st.title("NYC Tax Law RAG - QA Testing")
    st.caption("Query the NYC Administrative Code Title 11 (Taxation and Finance)")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        retrieval_mode = st.selectbox(
            "Retrieval Mode",
            options=["dense", "hybrid"],
            index=0,
            help="Dense uses semantic search only. Hybrid adds BM25 keyword matching.",
        )

        top_k = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of chunks to retrieve",
        )

        st.divider()
        st.subheader("Fallback Settings")

        enable_fallback = st.checkbox(
            "Enable Fallback",
            value=True,
            help="Use reasoning model when retrieval confidence is low",
        )

        fallback_threshold = st.slider(
            "Fallback Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            disabled=not enable_fallback,
            help="Trigger fallback when top chunk score is below this",
        )

        st.divider()
        st.caption("Keys are stored securely on the server.")

    # Main query input
    query = st.text_input(
        "Enter your question about NYC Tax Law:",
        placeholder="e.g., What are the property tax exemptions?",
    )

    # Process query
    if query:
        try:
            with st.spinner("Processing query..."):
                pipeline = get_pipeline(
                    retrieval_mode=retrieval_mode,
                    top_k=top_k,
                    enable_fallback=enable_fallback,
                    fallback_threshold=fallback_threshold,
                )
                response = pipeline.query(query)

            # Check if fallback was triggered
            if isinstance(response, FallbackResponse):
                st.warning(
                    f"Fallback triggered: {response.fallback_reason}",
                    icon="",
                )

                # Display answer
                st.subheader("Answer")
                st.markdown(response.answer)

                # Display low-confidence chunks
                if response.original_chunks:
                    st.subheader("Low-Confidence Chunks (used as context)")
                    chunks_data = []
                    for i, chunk in enumerate(response.original_chunks[:5], 1):
                        chunks_data.append({
                            "#": i,
                            "Section": chunk.section_number or "N/A",
                            "Score": f"{chunk.score:.3f}",
                            "Preview": chunk.text[:100] + "...",
                        })
                    st.dataframe(chunks_data, use_container_width=True)

                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Model", response.model)
                col2.metric("Generation", f"{response.generation_time_ms:.0f}ms")
                col3.metric("Tokens", response.total_tokens)

            else:
                # Normal RAG response
                st.subheader("Answer")
                st.markdown(response.answer)

                # Display sources
                if response.sources:
                    st.subheader("Sources")
                    sources_data = []
                    for i, source in enumerate(response.sources[:8], 1):
                        score = f"{source.score:.3f}" if source.score > 0 else "ref"
                        sources_data.append({
                            "#": i,
                            "Section": source.section_number or "N/A",
                            "Chapter": (source.chapter[:25] + "...") if source.chapter and len(source.chapter) > 25 else (source.chapter or "N/A"),
                            "Score": score,
                            "Type": source.chunk_type or "N/A",
                        })
                    st.dataframe(sources_data, use_container_width=True)

                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Context Tokens", response.context_tokens)
                col2.metric("Retrieval", f"{response.retrieval_time_ms:.0f}ms")
                col3.metric("Generation", f"{response.generation_time_ms:.0f}ms")
                col4.metric("Model", response.model)

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.exception(e)

    # Footer
    st.divider()
    st.caption(
        "NYC TaxRAG - RAG system for NYC Tax Law (Title 11). "
        "For QA and verification testing only."
    )


if __name__ == "__main__":
    main()
