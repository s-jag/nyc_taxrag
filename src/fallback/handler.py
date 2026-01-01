"""
Fallback handler for low-confidence RAG results.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from openai import OpenAI

from .config import FallbackConfig
from .prompt import FallbackPromptBuilder
from .types import FallbackResponse

if TYPE_CHECKING:
    from rich.console import Console
    from src.rag.types import RetrievalResult


class FallbackHandler:
    """Handles fallback to reasoning model when RAG confidence is low."""

    def __init__(
        self,
        config: FallbackConfig | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the fallback handler.

        Args:
            config: Fallback configuration
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.config = config or FallbackConfig()
        self.client = OpenAI(api_key=api_key)
        self.prompt_builder = FallbackPromptBuilder(self.config.prompt_path)

    def should_fallback(self, retrieval_result: "RetrievalResult") -> bool:
        """
        Determine if fallback should be triggered.

        Args:
            retrieval_result: Result from RAG retrieval

        Returns:
            True if fallback should be triggered
        """
        # No chunks at all - definitely fallback
        if not retrieval_result.primary_chunks:
            return True

        # Check top chunk score against threshold
        top_score = max(c.score for c in retrieval_result.primary_chunks)
        return top_score < self.config.confidence_threshold

    def get_top_score(self, retrieval_result: "RetrievalResult") -> float:
        """Get the top chunk score from retrieval results."""
        if not retrieval_result.primary_chunks:
            return 0.0
        return max(c.score for c in retrieval_result.primary_chunks)

    def execute_fallback(
        self,
        question: str,
        retrieval_result: "RetrievalResult",
        console: "Console | None" = None,
    ) -> FallbackResponse:
        """
        Execute the fallback with verbose output.

        Args:
            question: Original user question
            retrieval_result: Result from RAG retrieval
            console: Rich console for verbose output

        Returns:
            FallbackResponse with the generated answer
        """
        top_score = self.get_top_score(retrieval_result)

        # Get low-confidence chunks to include
        chunks_to_include = []
        if self.config.include_low_confidence_chunks:
            chunks_to_include = retrieval_result.primary_chunks[
                : self.config.max_chunks_in_fallback
            ]

        # Print verbose output
        if self.config.verbose and console:
            self._print_fallback_header(console, top_score)
            self._print_retrieved_chunks(console, retrieval_result.primary_chunks)
            self._print_fallback_strategy(console, chunks_to_include)

        # Build the fallback prompt
        prompt = self.prompt_builder.build_prompt(
            question=question,
            low_confidence_chunks=chunks_to_include if chunks_to_include else None,
        )

        # Call the reasoning model
        if self.config.verbose and console:
            console.print("\n[bold cyan]Generating response with fallback model...[/]")

        start_time = time.perf_counter()
        response = self._call_reasoning_model(prompt)
        generation_time_ms = (time.perf_counter() - start_time) * 1000

        # Build response
        fallback_response = FallbackResponse(
            answer=response["content"],
            model=self.config.fallback_model,
            was_fallback=True,
            original_chunks=list(retrieval_result.primary_chunks),
            top_chunk_score=top_score,
            fallback_reason=self._get_fallback_reason(top_score),
            generation_time_ms=generation_time_ms,
            query=question,
            input_tokens=response.get("input_tokens", 0),
            output_tokens=response.get("output_tokens", 0),
        )

        # Print response
        if self.config.verbose and console:
            self._print_fallback_response(console, fallback_response)

        return fallback_response

    def _call_reasoning_model(self, prompt: str) -> dict:
        """Call the OpenAI reasoning model."""
        response = self.client.chat.completions.create(
            model=self.config.fallback_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=self.config.max_tokens,
        )

        choice = response.choices[0]
        return {
            "content": choice.message.content or "",
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }

    def _get_fallback_reason(self, top_score: float) -> str:
        """Generate human-readable fallback reason."""
        if top_score == 0:
            return "No matching documents found in vector store"
        return (
            f"Top chunk score ({top_score:.1%}) below "
            f"confidence threshold ({self.config.confidence_threshold:.0%})"
        )

    def _print_fallback_header(self, console: "Console", top_score: float) -> None:
        """Print the fallback activation header."""
        from rich.panel import Panel

        console.print()
        console.print(
            Panel(
                "[bold yellow]LOW CONFIDENCE RETRIEVAL - ACTIVATING FALLBACK[/]",
                style="yellow",
                expand=False,
            )
        )

    def _print_retrieved_chunks(
        self, console: "Console", chunks: list
    ) -> None:
        """Print table of retrieved chunks."""
        from rich.table import Table

        if not chunks:
            console.print("\n[dim]No chunks retrieved from vector store[/]")
            return

        console.print(
            f"\n[bold]Retrieved Chunks[/] (below {self.config.confidence_threshold:.0%} threshold):"
        )

        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Section", width=12)
        table.add_column("Score", width=8)
        table.add_column("Preview", width=50, overflow="ellipsis")

        for i, chunk in enumerate(chunks[:5], 1):  # Show top 5
            section = chunk.section_number or "N/A"
            score = f"{chunk.score:.3f}"
            preview = chunk.text[:60].replace("\n", " ")
            if len(chunk.text) > 60:
                preview += "..."

            # Color code by score
            if chunk.score >= 0.5:
                score_style = "yellow"
            elif chunk.score >= 0.3:
                score_style = "orange3"
            else:
                score_style = "red"

            table.add_row(str(i), section, f"[{score_style}]{score}[/]", preview)

        console.print(table)

    def _print_fallback_strategy(
        self, console: "Console", chunks_to_include: list
    ) -> None:
        """Print the fallback strategy being used."""
        console.print(f"\n[bold cyan]Fallback Strategy:[/]")
        console.print(
            f"  [dim]->[/] Top chunk score: [yellow]{self.get_top_score_display(chunks_to_include)}[/] "
            f"(threshold: {self.config.confidence_threshold:.0%})"
        )
        console.print(
            f"  [dim]->[/] Including {len(chunks_to_include)} low-confidence chunks as context"
        )
        console.print(
            f"  [dim]->[/] Using comprehensive NYC Tax Law reference (LAW PACK)"
        )
        console.print(
            f"  [dim]->[/] Sending to reasoning model: [bold]{self.config.fallback_model}[/]"
        )

    def get_top_score_display(self, chunks: list) -> str:
        """Get displayable top score."""
        if not chunks:
            return "0.000"
        return f"{max(c.score for c in chunks):.3f}"

    def _print_fallback_response(
        self, console: "Console", response: FallbackResponse
    ) -> None:
        """Print the fallback response."""
        from rich.panel import Panel
        from rich.markdown import Markdown

        console.print()
        console.print(
            Panel(
                Markdown(response.answer),
                title=f"[bold green]FALLBACK RESPONSE ({response.model})[/]",
                border_style="green",
            )
        )

        # Stats line
        console.print(
            f"\n[dim]Stats: fallback=True, model={response.model}, "
            f"generation={response.generation_time_ms:.0f}ms, "
            f"tokens={response.total_tokens}[/]"
        )
