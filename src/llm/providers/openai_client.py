"""
OpenAI LLM client.
"""

from __future__ import annotations

import os

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

from ..client import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    """
    OpenAI client implementation.

    Uses the OpenAI Python SDK to interact with GPT models.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
    ):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model identifier to use.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        self.client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """
        Generate a response using OpenAI.

        Args:
            prompt: The user prompt/question.
            system: Optional system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0-2).

        Returns:
            LLMResponse with generated content.
        """
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract content from response
        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""

        return LLMResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            stop_reason=response.choices[0].finish_reason if response.choices else None,
        )
