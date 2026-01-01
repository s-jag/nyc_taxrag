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


# Available models for response generation
AVAILABLE_MODELS = {
    "gpt-4o": "GPT-4o (Default)",
    "gpt-4o-mini": "GPT-4o Mini (Faster)",
    "gpt-4-turbo": "GPT-4 Turbo",
    "o3-mini": "O3 Mini (Reasoning)",
    "o1-mini": "O1 Mini (Reasoning)",
    "o1": "O1 (Reasoning - Advanced)",
}


@st.cache_resource
def get_embedder_and_store():
    """Cache embedder and store separately (don't change with model)."""
    embedder = OpenAIEmbedder()
    store = QdrantStore()
    return embedder, store


def get_pipeline(
    retrieval_mode: str,
    top_k: int,
    enable_fallback: bool,
    fallback_threshold: float,
    model: str,
    fallback_model: str,
) -> RAGPipeline:
    """Initialize the RAG pipeline with specified model."""
    embedder, store = get_embedder_and_store()
    llm = OpenAIClient(model=model)

    config = RAGConfig(
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        expand_cross_refs=True,
        max_context_tokens=6000,
        enable_fallback=enable_fallback,
        fallback_threshold=fallback_threshold,
        model=model,
        fallback_model=fallback_model,
    )

    return RAGPipeline(
        embedder=embedder,
        store=store,
        llm=llm,
        config=config,
    )


def display_chunk_viewer(chunks, title="Retrieved Chunks"):
    """Display expandable chunk viewer."""
    st.subheader(title)

    if not chunks:
        st.info("No chunks to display.")
        return

    # Create a selectbox to choose which chunk to view
    chunk_options = []
    for i, chunk in enumerate(chunks, 1):
        section = chunk.section_number or "N/A"
        score = f"{chunk.score:.3f}" if chunk.score > 0 else "ref"
        preview = chunk.text[:50].replace("\n", " ") + "..."
        chunk_options.append(f"[{i}] Section {section} (Score: {score}) - {preview}")

    selected_idx = st.selectbox(
        "Select a chunk to view full text:",
        range(len(chunk_options)),
        format_func=lambda x: chunk_options[x],
        key=f"chunk_select_{title}",
    )

    # Display the selected chunk
    if selected_idx is not None and chunks:
        selected_chunk = chunks[selected_idx]

        with st.container():
            # Chunk metadata
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(f"**Section:** {selected_chunk.section_number or 'N/A'}")
            col2.markdown(f"**Score:** {selected_chunk.score:.3f}" if selected_chunk.score > 0 else "**Score:** ref")
            col3.markdown(f"**Type:** {selected_chunk.chunk_type or 'N/A'}")
            col4.markdown(f"**Tokens:** {selected_chunk.token_count}")

            # Chapter info
            if selected_chunk.chapter:
                st.markdown(f"**Chapter:** {selected_chunk.chapter}")

            # Full text in a code block for readability
            st.markdown("**Full Text:**")
            st.text_area(
                label="Chunk text",
                value=selected_chunk.text,
                height=200,
                disabled=True,
                label_visibility="collapsed",
            )

            # Citations if any
            if selected_chunk.citations:
                st.markdown(f"**Citations:** {', '.join(selected_chunk.citations)}")


def main():
    """Main Streamlit application."""
    st.title("NYC Tax Law RAG - QA Testing")
    st.caption("Query the NYC Administrative Code Title 11 (Taxation and Finance)")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Model selection
        st.subheader("Model Settings")

        model = st.selectbox(
            "Response Model",
            options=list(AVAILABLE_MODELS.keys()),
            index=0,
            format_func=lambda x: AVAILABLE_MODELS[x],
            help="Model used for generating the response",
        )

        fallback_model = st.selectbox(
            "Fallback Model",
            options=["o3-mini", "o1-mini", "o1"],
            index=0,
            format_func=lambda x: AVAILABLE_MODELS.get(x, x),
            help="Reasoning model used when fallback is triggered",
        )

        st.divider()
        st.subheader("Retrieval Settings")

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
            with st.spinner(f"Processing query with {AVAILABLE_MODELS[model]}..."):
                pipeline = get_pipeline(
                    retrieval_mode=retrieval_mode,
                    top_k=top_k,
                    enable_fallback=enable_fallback,
                    fallback_threshold=fallback_threshold,
                    model=model,
                    fallback_model=fallback_model,
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

                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Model", response.model)
                col2.metric("Generation", f"{response.generation_time_ms:.0f}ms")
                col3.metric("Tokens", response.total_tokens)

                # Display low-confidence chunks with viewer
                if response.original_chunks:
                    st.divider()
                    display_chunk_viewer(
                        response.original_chunks,
                        title="Low-Confidence Chunks (used as context)"
                    )

            else:
                # Normal RAG response
                st.subheader("Answer")
                st.markdown(response.answer)

                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Context Tokens", response.context_tokens)
                col2.metric("Retrieval", f"{response.retrieval_time_ms:.0f}ms")
                col3.metric("Generation", f"{response.generation_time_ms:.0f}ms")
                col4.metric("Model", response.model)

                # Display sources summary table
                if response.sources:
                    st.divider()
                    st.subheader("Sources Summary")
                    sources_data = []
                    for i, source in enumerate(response.sources[:10], 1):
                        score = f"{source.score:.3f}" if source.score > 0 else "ref"
                        sources_data.append({
                            "#": i,
                            "Section": source.section_number or "N/A",
                            "Chapter": (source.chapter[:30] + "...") if source.chapter and len(source.chapter) > 30 else (source.chapter or "N/A"),
                            "Score": score,
                            "Type": source.chunk_type or "N/A",
                            "Tokens": source.token_count,
                        })
                    st.dataframe(sources_data, use_container_width=True)

                    # Expandable chunk viewer
                    st.divider()
                    display_chunk_viewer(response.sources, title="View Full Chunk Text")

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
