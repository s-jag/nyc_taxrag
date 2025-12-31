"""
HTML parser for NYC Tax Law documents.

Extracts text and structural metadata from the HTML format used by
NYC Administrative Code exports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from bs4 import BeautifulSoup, Tag

from .strategies.base import ChunkMetadata


@dataclass
class ParsedSection:
    """A parsed section from the HTML document."""

    text: str
    section_id: str | None = None
    section_number: str | None = None  # e.g., "11-201"
    section_title: str | None = None
    chapter: str | None = None
    chapter_number: int | None = None
    subchapter: str | None = None
    subchapter_number: int | None = None
    element_type: str = "content"  # "title", "chapter", "subchapter", "section", "content"

    def to_metadata(self, source_file: str | None = None) -> ChunkMetadata:
        """Convert to ChunkMetadata."""
        return ChunkMetadata(
            chapter=self.chapter,
            chapter_number=self.chapter_number,
            subchapter=self.subchapter,
            subchapter_number=self.subchapter_number,
            section=f"§ {self.section_number}" if self.section_number else None,
            section_number=self.section_number,
            section_title=self.section_title,
            source_file=source_file,
        )


@dataclass
class ParsedDocument:
    """A fully parsed HTML document."""

    title: str = "Title 11: Taxation and Finance"
    sections: list[ParsedSection] = field(default_factory=list)
    raw_text: str = ""
    source_file: str | None = None

    def get_full_text(self) -> str:
        """Get concatenated text from all sections."""
        if self.raw_text:
            return self.raw_text
        return "\n\n".join(s.text for s in self.sections if s.text.strip())


class NYCTaxLawHTMLParser:
    """
    Parser for NYC Tax Law HTML documents.

    The HTML uses specific CSS classes to mark document structure:
    - "rbox Title": Document title
    - "rbox Chapter": Chapter headers
    - "Subchapter toc-destination rbox": Subchapter headers
    - "Section toc-destination rbox": Section headers (§ 11-XXX)
    - "rbox Normal-Level": Content paragraphs
    - "EdNoteSm": Editor's notes
    """

    # CSS class patterns for document elements
    TITLE_CLASS = "rbox Title"
    CHAPTER_CLASS = "rbox Chapter"
    SUBCHAPTER_PATTERN = re.compile(r'Subchapter.*rbox', re.IGNORECASE)
    SECTION_CLASS = "Section toc-destination rbox"
    CONTENT_CLASS = "rbox Normal-Level"

    def __init__(self):
        """Initialize the parser."""
        self.current_chapter: str | None = None
        self.current_chapter_number: int | None = None
        self.current_subchapter: str | None = None
        self.current_subchapter_number: int | None = None

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Replace HTML entities
        text = text.replace('\xa0', ' ')  # non-breaking space
        text = text.replace('\u00a7', '§')  # section symbol
        text = text.replace('&#167;', '§')
        text = text.replace('&#160;', ' ')

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def _extract_text_from_element(self, element: Tag) -> str:
        """Extract text from an HTML element, handling nested tags."""
        # Get all text, handling nested elements
        text_parts = []

        for child in element.descendants:
            if isinstance(child, str):
                text_parts.append(child)

        text = ''.join(text_parts)
        return self._clean_text(text)

    def _parse_chapter(self, element: Tag) -> ParsedSection:
        """Parse a chapter header element."""
        text = self._extract_text_from_element(element)

        # Extract chapter number
        match = re.search(r'Chapter\s+(\d+)[:\s]*(.+)?', text, re.IGNORECASE)
        chapter_number = int(match.group(1)) if match else None
        chapter_title = match.group(2).strip() if match and match.group(2) else None

        self.current_chapter = text
        self.current_chapter_number = chapter_number
        # Reset subchapter when entering new chapter
        self.current_subchapter = None
        self.current_subchapter_number = None

        return ParsedSection(
            text=text,
            chapter=self.current_chapter,
            chapter_number=chapter_number,
            element_type="chapter",
        )

    def _parse_subchapter(self, element: Tag) -> ParsedSection:
        """Parse a subchapter header element."""
        text = self._extract_text_from_element(element)

        # Extract subchapter number
        match = re.search(r'Subchapter\s+(\d+)[:\s]*(.+)?', text, re.IGNORECASE)
        subchapter_number = int(match.group(1)) if match else None

        self.current_subchapter = text
        self.current_subchapter_number = subchapter_number

        return ParsedSection(
            text=text,
            chapter=self.current_chapter,
            chapter_number=self.current_chapter_number,
            subchapter=self.current_subchapter,
            subchapter_number=subchapter_number,
            element_type="subchapter",
        )

    def _parse_section(self, element: Tag) -> ParsedSection:
        """Parse a section header element (§ 11-XXX)."""
        text = self._extract_text_from_element(element)

        # Extract section number and title
        match = re.search(r'§\s*(11-[\d.]+)\s*(.+)?', text)
        section_number = match.group(1) if match else None
        section_title = match.group(2).strip().rstrip('.') if match and match.group(2) else None

        # Get element ID if available
        anchor = element.find('a', {'id': True})
        section_id = anchor['id'] if anchor else None

        return ParsedSection(
            text=text,
            section_id=section_id,
            section_number=section_number,
            section_title=section_title,
            chapter=self.current_chapter,
            chapter_number=self.current_chapter_number,
            subchapter=self.current_subchapter,
            subchapter_number=self.current_subchapter_number,
            element_type="section",
        )

    def _parse_content(self, element: Tag) -> ParsedSection:
        """Parse a content paragraph element."""
        text = self._extract_text_from_element(element)

        # Check if this is an editor's note
        editor_note = element.find(class_='EdNoteSm')
        element_type = "editor_note" if editor_note else "content"

        return ParsedSection(
            text=text,
            chapter=self.current_chapter,
            chapter_number=self.current_chapter_number,
            subchapter=self.current_subchapter,
            subchapter_number=self.current_subchapter_number,
            element_type=element_type,
        )

    def _get_element_class(self, element: Tag) -> str | None:
        """Get the class attribute as a string."""
        classes = element.get('class', [])
        if isinstance(classes, list):
            return ' '.join(classes)
        return classes

    def parse_html(self, html_content: str, source_file: str | None = None) -> ParsedDocument:
        """
        Parse HTML content and extract structured sections.

        Args:
            html_content: Raw HTML string.
            source_file: Source file name for metadata.

        Returns:
            ParsedDocument with extracted sections.
        """
        soup = BeautifulSoup(html_content, 'lxml')
        doc = ParsedDocument(source_file=source_file)

        # Reset state
        self.current_chapter = None
        self.current_chapter_number = None
        self.current_subchapter = None
        self.current_subchapter_number = None

        # Find all relevant elements
        for element in soup.find_all('div', class_=True):
            element_class = self._get_element_class(element)
            if not element_class:
                continue

            # Parse based on class
            if self.TITLE_CLASS in element_class:
                text = self._extract_text_from_element(element)
                doc.title = text
                doc.sections.append(ParsedSection(
                    text=text,
                    element_type="title",
                ))
            elif self.CHAPTER_CLASS in element_class:
                doc.sections.append(self._parse_chapter(element))
            elif self.SUBCHAPTER_PATTERN.search(element_class):
                doc.sections.append(self._parse_subchapter(element))
            elif self.SECTION_CLASS in element_class:
                doc.sections.append(self._parse_section(element))
            elif self.CONTENT_CLASS in element_class:
                section = self._parse_content(element)
                if section.text.strip():  # Only add non-empty content
                    doc.sections.append(section)

        # Build raw text
        doc.raw_text = self._build_raw_text(doc.sections)

        return doc

    def _build_raw_text(self, sections: list[ParsedSection]) -> str:
        """Build raw text from sections, preserving structure."""
        parts = []
        current_section_number = None

        for section in sections:
            if section.element_type == "title":
                parts.append(section.text + "\n")
            elif section.element_type == "chapter":
                parts.append("\n" + section.text + "\n")
            elif section.element_type == "subchapter":
                parts.append(section.text + "\n")
            elif section.element_type == "section":
                current_section_number = section.section_number
                parts.append("\n" + section.text + "\n")
            elif section.element_type in ("content", "editor_note"):
                parts.append(section.text + "\n")

        return "\n".join(parts)

    def parse_file(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse an HTML file.

        Args:
            file_path: Path to the HTML file.

        Returns:
            ParsedDocument with extracted sections.
        """
        path = Path(file_path)
        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        return self.parse_html(html_content, source_file=path.name)

    def extract_sections_by_number(
        self,
        doc: ParsedDocument,
        section_numbers: list[str] | None = None,
    ) -> Iterator[tuple[str, list[ParsedSection]]]:
        """
        Group sections by their section number.

        Args:
            doc: Parsed document.
            section_numbers: Optional list of section numbers to filter.

        Yields:
            Tuples of (section_number, list of content sections).
        """
        current_section = None
        current_content: list[ParsedSection] = []

        for section in doc.sections:
            if section.element_type == "section":
                # Yield previous section if exists
                if current_section and current_content:
                    if section_numbers is None or current_section in section_numbers:
                        yield current_section, current_content

                # Start new section
                current_section = section.section_number
                current_content = [section]
            elif section.element_type in ("content", "editor_note") and current_section:
                current_content.append(section)
            elif section.element_type in ("chapter", "subchapter"):
                # Include headers in content
                if current_section:
                    current_content.append(section)
                else:
                    # Standalone header before any section
                    pass

        # Yield last section
        if current_section and current_content:
            if section_numbers is None or current_section in section_numbers:
                yield current_section, current_content
