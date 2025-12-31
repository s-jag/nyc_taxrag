"""
Configuration settings for NYC TaxRAG.

Uses pydantic-settings for type-safe configuration with environment variable support.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="NYCTAX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # =========================================================================
    # API Keys
    # =========================================================================
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for embeddings",
    )

    # =========================================================================
    # LLM Configuration
    # =========================================================================
    llm_provider: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description="LLM provider to use",
    )
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model name for LLM generation",
    )
    llm_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for LLM response",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Temperature for LLM generation",
    )

    # =========================================================================
    # Embedding Configuration
    # =========================================================================
    embedding_provider: Literal["openai", "local"] = Field(
        default="openai",
        description="Embedding provider",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model name for embeddings",
    )

    # =========================================================================
    # Vector Store Configuration
    # =========================================================================
    vectorstore_provider: Literal["chroma", "pinecone", "faiss"] = Field(
        default="chroma",
        description="Vector store provider",
    )
    vectorstore_path: str = Field(
        default="data/processed/index",
        description="Path for local vector store",
    )

    # Pinecone-specific settings
    pinecone_api_key: str = Field(
        default="",
        description="Pinecone API key",
    )
    pinecone_environment: str = Field(
        default="",
        description="Pinecone environment",
    )
    pinecone_index_name: str = Field(
        default="nyc-taxrag",
        description="Pinecone index name",
    )

    # =========================================================================
    # Retrieval Configuration
    # =========================================================================
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve",
    )
    retrieval_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for retrieval",
    )
    use_reranking: bool = Field(
        default=False,
        description="Enable reranking of retrieved chunks",
    )

    # =========================================================================
    # Chunking Configuration
    # =========================================================================
    chunk_size: int = Field(
        default=800,
        ge=100,
        le=4000,
        description="Target chunk size in tokens",
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Overlap between chunks in tokens",
    )

    # =========================================================================
    # Fallback Configuration
    # =========================================================================
    fallback_enabled: bool = Field(
        default=True,
        description="Enable fallback system",
    )
    fallback_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold below which to use fallback",
    )
    compressed_prompt_path: str = Field(
        default="data/processed/compressed_prompt.txt",
        description="Path to compressed prompt file",
    )

    # =========================================================================
    # Paths
    # =========================================================================
    raw_data_path: str = Field(
        default="data/raw",
        description="Raw data directory",
    )
    processed_data_path: str = Field(
        default="data/processed",
        description="Processed data directory",
    )

    # =========================================================================
    # Helper Properties
    # =========================================================================
    @property
    def raw_data_dir(self) -> Path:
        """Get raw data directory as Path."""
        return Path(self.raw_data_path)

    @property
    def processed_data_dir(self) -> Path:
        """Get processed data directory as Path."""
        return Path(self.processed_data_path)

    @property
    def vectorstore_dir(self) -> Path:
        """Get vector store directory as Path."""
        return Path(self.vectorstore_path)


# Global settings instance (lazy loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
