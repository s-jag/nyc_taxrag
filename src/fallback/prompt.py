"""
Fallback prompt builder - loads and formats the fallback prompt template.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.vectorstore import SearchResult


class FallbackPromptBuilder:
    """Builds fallback prompts by loading the template and injecting context."""

    def __init__(self, prompt_path: str = "fallback_prompt.txt"):
        """
        Initialize the prompt builder.

        Args:
            prompt_path: Path to the fallback prompt template file
        """
        self.prompt_path = Path(prompt_path)
        self._template: str | None = None

    @property
    def template(self) -> str:
        """Lazy load the template."""
        if self._template is None:
            self._template = self._load_template()
        return self._template

    def _load_template(self) -> str:
        """Load the fallback prompt template from file."""
        if not self.prompt_path.exists():
            raise FileNotFoundError(
                f"Fallback prompt template not found: {self.prompt_path}"
            )

        return self.prompt_path.read_text(encoding="utf-8")

    def build_prompt(
        self,
        question: str,
        low_confidence_chunks: list["SearchResult"] | None = None,
    ) -> str:
        """
        Build the complete fallback prompt.

        Args:
            question: The user's question
            low_confidence_chunks: Optional list of low-confidence chunks to include

        Returns:
            The complete prompt ready for the reasoning model
        """
        # Start with the template
        prompt = self.template

        # Replace the question placeholder
        prompt = prompt.replace("{USER_QUESTION}", question)

        # If we have low-confidence chunks, prepend them as additional context
        if low_confidence_chunks:
            context_section = self._format_chunks_as_context(low_confidence_chunks)
            # Insert context before the [USER TASK] section
            if "[USER TASK]" in prompt:
                prompt = prompt.replace(
                    "[USER TASK]",
                    f"{context_section}\n\n[USER TASK]",
                )
            else:
                # If no USER TASK marker, prepend to the prompt
                prompt = f"{context_section}\n\n{prompt}"

        return prompt

    def _format_chunks_as_context(
        self, chunks: list["SearchResult"]
    ) -> str:
        """Format chunks as additional context section."""
        lines = [
            "[ADDITIONAL CONTEXT FROM RAG RETRIEVAL]",
            "The following excerpts were retrieved but had low confidence scores.",
            "They may still provide useful context:\n",
        ]

        for i, chunk in enumerate(chunks, 1):
            section = chunk.section_number or "Unknown"
            score = chunk.score
            lines.append(f"--- Excerpt {i} (Section {section}, score: {score:.3f}) ---")
            lines.append(chunk.text)
            lines.append("")

        lines.append("[END ADDITIONAL CONTEXT]")
        return "\n".join(lines)

    def get_template_stats(self) -> dict:
        """Get statistics about the template."""
        template = self.template
        return {
            "path": str(self.prompt_path),
            "char_count": len(template),
            "line_count": template.count("\n") + 1,
            "has_question_placeholder": "{USER_QUESTION}" in template,
            "has_law_pack": "[LAW PACK]" in template,
        }
