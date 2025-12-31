"""
Fallback chunking strategy using regex patterns.

This strategy is used when:
- Claude API is unavailable
- Cost savings are needed
- Quick local processing is preferred

It uses regex patterns specific to NYC tax law structure to split documents.
"""

from __future__ import annotations

import re
from typing import Iterator

from .base import Chunk, ChunkMetadata, ChunkingStrategy


class FallbackChunkingStrategy(ChunkingStrategy):
    """
    Simple regex-based chunking strategy for NYC tax law documents.

    Splits documents based on:
    - Section markers (§ 11-XXX)
    - Chapter/Subchapter headers
    - Paragraph breaks

    This is less intelligent than zChunk but works offline and is fast.
    """

    def __init__(
        self,
        target_chunk_size: int = 1500,
        min_chunk_size: int = 200,
        max_chunk_size: int = 3000,
        overlap: int = 100,
    ):
        """
        Initialize the fallback strategy.

        Args:
            target_chunk_size: Target chunk size in characters.
            min_chunk_size: Minimum chunk size (merge smaller chunks).
            max_chunk_size: Maximum chunk size (split larger chunks).
            overlap: Character overlap between chunks.
        """
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

        # Patterns for splitting
        self.section_pattern = re.compile(r'(§\s*11-\d+[^\n]*)')
        self.chapter_pattern = re.compile(r'(Chapter\s+\d+[:\s][^\n]*)', re.IGNORECASE)
        self.subchapter_pattern = re.compile(r'(Subchapter\s+\d+[:\s][^\n]*)', re.IGNORECASE)
        self.subsection_pattern = re.compile(r'\n\s+([a-z])\.\s+', re.IGNORECASE)

    def get_strategy_name(self) -> str:
        return "fallback"

    def _split_on_sections(self, text: str) -> list[tuple[str, str | None]]:
        """
        Split text on section markers (§ 11-XXX).

        Returns list of (text, section_number) tuples.
        """
        # Find all section starts
        sections = []
        matches = list(self.section_pattern.finditer(text))

        if not matches:
            return [(text, None)]

        # Add text before first section
        if matches[0].start() > 0:
            sections.append((text[:matches[0].start()].strip(), None))

        # Add each section
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()

            # Extract section number
            section_match = re.search(r'§\s*(11-\d+)', section_text)
            section_num = section_match.group(1) if section_match else None

            sections.append((section_text, section_num))

        return sections

    def _split_on_paragraphs(self, text: str) -> list[str]:
        """
        Split text on paragraph breaks.
        """
        # Split on double newlines or subsection markers
        parts = re.split(r'\n\n+', text)
        return [p.strip() for p in parts if p.strip()]

    def _split_large_chunk(self, text: str) -> list[str]:
        """
        Split a chunk that's too large into smaller pieces.
        """
        if len(text) <= self.max_chunk_size:
            return [text]

        chunks = []
        paragraphs = self._split_on_paragraphs(text)

        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size > self.target_chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Add overlap from end of previous chunk
                if self.overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1][-self.overlap:]
                    current_chunk = [overlap_text + para] if overlap_text else [para]
                    current_size = len(current_chunk[0])
                else:
                    current_chunk = [para]
                    current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size + 2  # +2 for \n\n

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Merge chunks that are too small.
        """
        if not chunks:
            return chunks

        merged = []
        buffer_chunk = None

        for chunk in chunks:
            if buffer_chunk is None:
                buffer_chunk = chunk
                continue

            if len(buffer_chunk.text) < self.min_chunk_size:
                buffer_chunk.text = buffer_chunk.text + "\n\n" + chunk.text
                buffer_chunk._token_count = None
                if chunk.metadata.section and not buffer_chunk.metadata.section:
                    buffer_chunk.metadata = chunk.metadata
            else:
                merged.append(buffer_chunk)
                buffer_chunk = chunk

        if buffer_chunk:
            merged.append(buffer_chunk)

        # Re-index
        for i, chunk in enumerate(merged):
            chunk.id = f"chunk_{i:05d}"
            chunk.metadata.chunk_index = i

        return merged

    def _extract_metadata(
        self,
        text: str,
        section_number: str | None = None,
        source_file: str | None = None,
        chunk_index: int = 0,
    ) -> ChunkMetadata:
        """
        Extract metadata from chunk text.
        """
        metadata = ChunkMetadata(
            source_file=source_file,
            chunk_index=chunk_index,
        )

        # Extract chapter
        chapter_match = self.chapter_pattern.search(text)
        if chapter_match:
            num_match = re.search(r'Chapter\s+(\d+)', chapter_match.group(1), re.IGNORECASE)
            if num_match:
                metadata.chapter_number = int(num_match.group(1))
                metadata.chapter = chapter_match.group(1).strip()

        # Extract subchapter
        subchapter_match = self.subchapter_pattern.search(text)
        if subchapter_match:
            num_match = re.search(r'Subchapter\s+(\d+)', subchapter_match.group(1), re.IGNORECASE)
            if num_match:
                metadata.subchapter_number = int(num_match.group(1))
                metadata.subchapter = subchapter_match.group(1).strip()

        # Extract section
        if section_number:
            metadata.section_number = section_number
            metadata.section = f"§ {section_number}"

            # Try to get section title
            section_match = re.search(rf'§\s*{re.escape(section_number)}\s+([^\n]+)', text)
            if section_match:
                metadata.section_title = section_match.group(1).strip().rstrip('.')

        return metadata

    def chunk(
        self,
        text: str,
        source_file: str | None = None,
        base_metadata: ChunkMetadata | None = None,
    ) -> list[Chunk]:
        """
        Split text into chunks using regex patterns.

        Args:
            text: The full document text to chunk.
            source_file: Optional source file name for metadata.
            base_metadata: Optional base metadata to inherit.

        Returns:
            List of Chunk objects with text and metadata.
        """
        chunks = []
        chunk_index = 0

        # First, split on sections
        sections = self._split_on_sections(text)

        for section_text, section_number in sections:
            if not section_text.strip():
                continue

            # If section is too large, split further
            if len(section_text) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(section_text)
                for i, sub_text in enumerate(sub_chunks):
                    metadata = self._extract_metadata(
                        sub_text,
                        section_number=section_number,
                        source_file=source_file,
                        chunk_index=chunk_index,
                    )
                    metadata.chunk_type = "big" if i == 0 else "small"

                    if base_metadata:
                        metadata.title = base_metadata.title

                    chunks.append(Chunk(
                        id=f"chunk_{chunk_index:05d}",
                        text=sub_text,
                        metadata=metadata,
                    ))
                    chunk_index += 1
            else:
                metadata = self._extract_metadata(
                    section_text,
                    section_number=section_number,
                    source_file=source_file,
                    chunk_index=chunk_index,
                )
                metadata.chunk_type = "big"

                if base_metadata:
                    metadata.title = base_metadata.title

                chunks.append(Chunk(
                    id=f"chunk_{chunk_index:05d}",
                    text=section_text,
                    metadata=metadata,
                ))
                chunk_index += 1

        return self._merge_small_chunks(chunks)

    def chunk_iterator(
        self,
        text: str,
        source_file: str | None = None,
        base_metadata: ChunkMetadata | None = None,
    ) -> Iterator[Chunk]:
        """
        Iterate over chunks.
        """
        yield from self.chunk(text, source_file, base_metadata)
