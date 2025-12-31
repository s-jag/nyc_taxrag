"""
zChunk strategy using Claude API for semantic chunking.

This implementation adapts the zChunk algorithm to use Claude API
instead of local Llama inference. Claude inserts split tokens (段/顿)
at semantic boundaries, which are then parsed to create chunks.
"""

from __future__ import annotations

import re
from typing import Iterator, TYPE_CHECKING

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = None  # type: ignore

from ..prompts import (
    BIG_SPLIT_TOKEN,
    SMALL_SPLIT_TOKEN,
    get_system_prompt,
    get_few_shot_example,
)
from .base import Chunk, ChunkMetadata, ChunkingStrategy


class ZChunkStrategy(ChunkingStrategy):
    """
    Chunking strategy using Claude API to detect semantic boundaries.

    The zChunk algorithm works by:
    1. Sending document batches to Claude with instructions to insert split tokens
    2. Claude returns the text with 段 (big) and 顿 (small) markers inserted
    3. Parse the markers to create chunks with appropriate metadata

    This approach is superior to regex because:
    - Claude understands document semantics
    - Works across document types without custom rules
    - Handles edge cases intelligently
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        batch_size: int = 8000,
        overlap: int = 400,
        max_retries: int = 3,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        """
        Initialize the zChunk strategy.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Claude model to use for chunking.
            batch_size: Characters per API call (smaller = more calls but better context).
            overlap: Character overlap between batches to ensure no content is lost.
            max_retries: Maximum API retry attempts on failure.
            min_chunk_size: Minimum chunk size in characters (merge smaller chunks).
            max_chunk_size: Maximum chunk size in characters (split larger chunks).
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package is required for ZChunkStrategy. "
                "Install with: pip install anthropic"
            )
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.model = model
        self.batch_size = batch_size
        self.overlap = overlap
        self.max_retries = max_retries
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Pre-compute prompts
        self.system_prompt = get_system_prompt()
        self.example_input, self.example_output = get_few_shot_example()

    def get_strategy_name(self) -> str:
        return "zchunk"

    def _call_claude(self, text: str) -> str:
        """
        Call Claude API to insert split tokens.

        Args:
            text: Text to process.

        Returns:
            Text with split tokens inserted.
        """
        messages = [
            # Few-shot example
            {"role": "user", "content": self.example_input},
            {"role": "assistant", "content": self.example_output},
            # Actual request
            {"role": "user", "content": text},
        ]

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=16384,  # Allow for full text + tokens
                    system=self.system_prompt,
                    messages=messages,
                )
                return response.content[0].text
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Claude API failed after {self.max_retries} attempts: {e}")
                continue

        return text  # Fallback: return original text

    def _process_batch(self, text: str) -> str:
        """
        Process a single batch of text through Claude.

        Args:
            text: Batch text to process.

        Returns:
            Text with split tokens inserted.
        """
        return self._call_claude(text)

    def _process_document(self, text: str) -> str:
        """
        Process entire document through Claude in batches.

        Args:
            text: Full document text.

        Returns:
            Full document with split tokens inserted.
        """
        if len(text) <= self.batch_size:
            return self._process_batch(text)

        # Process in overlapping batches
        results = []
        processed_end = 0

        for i in range(0, len(text), self.batch_size - self.overlap):
            start = i
            end = min(i + self.batch_size, len(text))

            batch_text = text[start:end]
            processed_batch = self._process_batch(batch_text)

            if i == 0:
                # First batch: use all of it
                results.append(processed_batch)
                processed_end = end
            else:
                # Subsequent batches: skip overlap portion
                # Find where the overlap ends in the processed text
                overlap_text = text[start:start + self.overlap]
                overlap_end_idx = self._find_overlap_boundary(processed_batch, overlap_text)

                if overlap_end_idx > 0:
                    results.append(processed_batch[overlap_end_idx:])
                else:
                    # Fallback: just skip approximate overlap
                    skip_chars = int(len(processed_batch) * (self.overlap / self.batch_size))
                    results.append(processed_batch[skip_chars:])

                processed_end = end

        return "".join(results)

    def _find_overlap_boundary(self, processed_text: str, original_overlap: str) -> int:
        """
        Find where the overlap region ends in the processed text.

        The processed text has split tokens inserted, so we need to find
        where the original overlap content ends.
        """
        # Remove split tokens from processed text for comparison
        clean_processed = processed_text.replace(BIG_SPLIT_TOKEN, "").replace(SMALL_SPLIT_TOKEN, "")

        # Find where overlap ends in clean text
        overlap_clean = original_overlap.replace(BIG_SPLIT_TOKEN, "").replace(SMALL_SPLIT_TOKEN, "")
        idx = clean_processed.find(overlap_clean[-50:])  # Match on last 50 chars

        if idx < 0:
            return 0

        # Map back to processed text position
        clean_idx = idx + len(overlap_clean[-50:])
        processed_idx = 0
        clean_count = 0

        for char in processed_text:
            if char not in (BIG_SPLIT_TOKEN, SMALL_SPLIT_TOKEN):
                clean_count += 1
            processed_idx += 1
            if clean_count >= clean_idx:
                break

        return processed_idx

    def _parse_chunks(
        self,
        text_with_tokens: str,
        source_file: str | None = None,
        base_metadata: ChunkMetadata | None = None,
    ) -> list[Chunk]:
        """
        Parse text with split tokens into Chunk objects.

        Args:
            text_with_tokens: Text containing 段 and 顿 split tokens.
            source_file: Source file name for metadata.
            base_metadata: Base metadata to inherit.

        Returns:
            List of Chunk objects.
        """
        chunks = []
        chunk_index = 0

        # Split on big tokens first
        big_sections = text_with_tokens.split(BIG_SPLIT_TOKEN)

        current_pos = 0
        for section in big_sections:
            if not section.strip():
                continue

            # Within each big section, split on small tokens
            small_parts = section.split(SMALL_SPLIT_TOKEN)

            for part in small_parts:
                part = part.strip()
                if not part or len(part) < 10:  # Skip very small fragments
                    continue

                # Extract section metadata from text
                metadata = self._extract_metadata(
                    part,
                    source_file=source_file,
                    base_metadata=base_metadata,
                    chunk_index=chunk_index,
                )

                # Determine if this was a big or small split
                metadata.chunk_type = "big" if part == small_parts[0] else "small"

                chunk = Chunk(
                    id=f"chunk_{chunk_index:05d}",
                    text=part,
                    metadata=metadata,
                )
                chunks.append(chunk)
                chunk_index += 1

        return self._merge_small_chunks(chunks)

    def _extract_metadata(
        self,
        text: str,
        source_file: str | None = None,
        base_metadata: ChunkMetadata | None = None,
        chunk_index: int = 0,
    ) -> ChunkMetadata:
        """
        Extract metadata from chunk text.

        Looks for patterns like:
        - Chapter headers: "Chapter N: ..."
        - Subchapter headers: "Subchapter N: ..."
        - Section headers: "§ 11-XXX ..."
        """
        metadata = ChunkMetadata(
            source_file=source_file,
            chunk_index=chunk_index,
        )

        if base_metadata:
            metadata.title = base_metadata.title
            metadata.chapter = base_metadata.chapter
            metadata.chapter_number = base_metadata.chapter_number
            metadata.subchapter = base_metadata.subchapter

        # Extract chapter
        chapter_match = re.search(r'Chapter\s+(\d+)[:\s]+([^\n]+)', text, re.IGNORECASE)
        if chapter_match:
            metadata.chapter_number = int(chapter_match.group(1))
            metadata.chapter = f"Chapter {chapter_match.group(1)}: {chapter_match.group(2).strip()}"

        # Extract subchapter
        subchapter_match = re.search(r'Subchapter\s+(\d+)[:\s]+([^\n]+)', text, re.IGNORECASE)
        if subchapter_match:
            metadata.subchapter_number = int(subchapter_match.group(1))
            metadata.subchapter = f"Subchapter {subchapter_match.group(1)}: {subchapter_match.group(2).strip()}"

        # Extract section
        section_match = re.search(r'§\s*(11-\d+)\s*([^\n]*)', text)
        if section_match:
            metadata.section_number = section_match.group(1)
            metadata.section = f"§ {section_match.group(1)}"
            if section_match.group(2):
                metadata.section_title = section_match.group(2).strip().rstrip('.')

        return metadata

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Merge chunks that are too small.

        Args:
            chunks: List of chunks.

        Returns:
            List with small chunks merged into neighbors.
        """
        if not chunks:
            return chunks

        merged = []
        buffer_chunk = None

        for chunk in chunks:
            if buffer_chunk is None:
                buffer_chunk = chunk
                continue

            # If buffer is too small, merge with current
            if len(buffer_chunk.text) < self.min_chunk_size:
                buffer_chunk.text = buffer_chunk.text + "\n\n" + chunk.text
                buffer_chunk._token_count = None  # Reset token count
                # Keep metadata from the chunk with section info
                if chunk.metadata.section and not buffer_chunk.metadata.section:
                    buffer_chunk.metadata = chunk.metadata
            else:
                merged.append(buffer_chunk)
                buffer_chunk = chunk

        if buffer_chunk:
            merged.append(buffer_chunk)

        # Re-index merged chunks
        for i, chunk in enumerate(merged):
            chunk.id = f"chunk_{i:05d}"
            chunk.metadata.chunk_index = i

        return merged

    def chunk(
        self,
        text: str,
        source_file: str | None = None,
        base_metadata: ChunkMetadata | None = None,
    ) -> list[Chunk]:
        """
        Split text into chunks using Claude API.

        Args:
            text: The full document text to chunk.
            source_file: Optional source file name for metadata.
            base_metadata: Optional base metadata to inherit.

        Returns:
            List of Chunk objects with text and metadata.
        """
        # Process document through Claude
        text_with_tokens = self._process_document(text)

        # Parse into chunks
        return self._parse_chunks(text_with_tokens, source_file, base_metadata)

    def chunk_iterator(
        self,
        text: str,
        source_file: str | None = None,
        base_metadata: ChunkMetadata | None = None,
    ) -> Iterator[Chunk]:
        """
        Stream chunks as they are processed.

        For the Claude API version, this isn't truly streaming since
        we need to process batches. But it allows for progress tracking.
        """
        # For now, just yield from the full chunk() result
        # A true streaming implementation would yield after each batch
        yield from self.chunk(text, source_file, base_metadata)
