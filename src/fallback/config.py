"""
Configuration for the fallback system.
"""

from dataclasses import dataclass


@dataclass
class FallbackConfig:
    """Configuration for fallback to reasoning model when RAG confidence is low."""

    # Trigger threshold - fallback activates when top chunk score is below this
    confidence_threshold: float = 0.6

    # OpenAI reasoning model to use for fallback
    fallback_model: str = "o3-mini"

    # Whether to include low-confidence chunks in the fallback prompt
    include_low_confidence_chunks: bool = True

    # Maximum number of low-confidence chunks to include
    max_chunks_in_fallback: int = 3

    # Path to the fallback prompt file
    prompt_path: str = "fallback_prompt.txt"

    # Whether to show verbose output in CLI
    verbose: bool = True

    # Temperature for reasoning model (lower = more deterministic)
    temperature: float = 1.0  # O3 models work best with temperature 1.0

    # Max tokens for fallback response
    max_tokens: int = 4096

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.max_chunks_in_fallback < 0:
            raise ValueError("max_chunks_in_fallback must be >= 0")
