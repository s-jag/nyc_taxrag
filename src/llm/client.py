"""
Abstract LLM client interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.

    Implementations should handle:
    - API authentication
    - Request formatting
    - Response parsing
    - Error handling
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt/question.
            system: Optional system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0-2).

        Returns:
            LLMResponse with generated content.
        """
        pass

    def generate_text(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        """
        Generate a response and return just the text content.

        Convenience method that wraps generate().

        Args:
            prompt: The user prompt/question.
            system: Optional system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0-2).

        Returns:
            Generated text content.
        """
        response = self.generate(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.content
